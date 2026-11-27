"""
oilfc/imaging.py  -  Price-chart images for the image-based CNN (owner: image CNN)

Implements the imaging method of Ren, Jiang, Ji & Zhai (2024, J. Forecasting 43:2809),
which modifies Jiang, Kelly & Xiu (2023, J. Finance) "(Re-)Imag(in)ing price trends":

  * one image = `window` trading days (paper: 20), 3 pixel columns per day
        column 3d     open tick  (1 px, at the open's row)
        column 3d + 1 high-low bar
        column 3d + 2 close tick
    so a 20-day image is 60 px wide; height is fixed (paper: 32 px)
  * the vertical axis is rescaled per image so the window's lowest low touches the
    bottom row and its highest high the top row
  * black background, white bars

Three styles (Figure 1 of the paper):
  "volume_intensity"  (a) the paper's method: no separate volume panel; each day's bars
                      are drawn with brightness proportional to volume / max volume in the
                      window (largest volume = brightest)
  "no_volume"         (b) same chart, all bars full white
  "jiang"             (c) Jiang et al.: full-white price bars on the top part of the image,
                      volume bars in the bottom fifth

Everything is vectorised over images, so ~10,000 images take well under a second.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STYLES = ("volume_intensity", "no_volume", "jiang")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def clean_ohlcv(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Basic repairs for Yahoo futures data (CL=F, BZ=F) before imaging.

    * drop rows without a close
    * open missing or exactly 0 (a Yahoo artefact) -> previous close
    * enforce low <= min(open, close) and high >= max(open, close)
    * volume <= 0 -> NaN (drawn at full brightness, see render_images)
    Negative prices (WTI, 2020-04-20) are kept: the per-image min/max scaling handles them.
    """
    df = df[OHLCV].copy().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    n0 = len(df)
    df = df.dropna(subset=["Close"])
    bad_open = df["Open"].isna() | (df["Open"] == 0)
    df.loc[bad_open, "Open"] = df["Close"].shift(1)[bad_open]
    df["Open"] = df["Open"].fillna(df["Close"])
    hi = df[["Open", "High", "Close"]].max(axis=1)
    lo = df[["Open", "Low", "Close"]].min(axis=1)
    fixed_hl = int(((df["High"] != hi) | (df["Low"] != lo)).sum())
    df["High"], df["Low"] = hi, lo
    df.loc[df["Volume"] <= 0, "Volume"] = np.nan
    if verbose:
        print(f"clean_ohlcv: {n0 - len(df)} rows without close dropped, {int(bad_open.sum())} opens "
              f"filled, {fixed_hl} high/low repaired, {int(df['Volume'].isna().sum())} days without volume")
    return df


def _rows(p: np.ndarray, lo: np.ndarray, span: np.ndarray, n_rows: int) -> np.ndarray:
    """Price -> pixel row (row 0 = top of the image)."""
    frac = (p - lo[:, None]) / span[:, None]
    return (n_rows - 1 - np.rint(frac * (n_rows - 1))).astype(np.int64)


def render_images(o, h, l, c, v, height: int = 32, style: str = "volume_intensity",
                  min_intensity: float = 0.1) -> np.ndarray:
    """Render a batch of windows. Inputs have shape (n_images, window).

    Returns uint8 array (n_images, height, 3 * window), 0 = black, 255 = white.
    `min_intensity` keeps very low-volume days visible in the volume_intensity style
    (the paper does not say how it treats them; set 0 for a pure linear scale).
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    o, h, l, c, v = (np.asarray(a, dtype=float) for a in (o, h, l, c, v))
    n, w = c.shape
    img = np.zeros((n, height, 3 * w), dtype=np.uint8)

    vol_rows = height // 5 if style == "jiang" else 0
    price_rows = height - vol_rows - (1 if vol_rows else 0)   # 1 px gap above the volume panel

    lo, hi = l.min(axis=1), h.max(axis=1)
    span = np.where(hi > lo, hi - lo, 1.0)
    r_o, r_h, r_l, r_c = (_rows(p, lo, span, price_rows) for p in (o, h, l, c))

    vmax = np.nanmax(np.where(np.isnan(v), -np.inf, v), axis=1)
    rel_v = np.where(np.isnan(v) | ~np.isfinite(vmax)[:, None], 1.0,
                     v / np.where(vmax > 0, vmax, 1.0)[:, None])
    if style == "volume_intensity":
        val = np.rint(255 * np.clip(rel_v, min_intensity, 1.0)).astype(np.uint8)
    else:
        val = np.full((n, w), 255, dtype=np.uint8)

    ii = np.arange(n)[:, None]
    dd = np.arange(w)[None, :]
    # high-low bars: mask over rows, shape (n, w, price_rows)
    rr = np.arange(price_rows)[None, None, :]
    bar = (rr >= r_h[..., None]) & (rr <= r_l[..., None])
    bar_val = np.where(bar, val[..., None], 0).astype(np.uint8)          # (n, w, rows)
    img[:, :price_rows, 1::3] = bar_val.transpose(0, 2, 1)
    # open and close ticks
    img[ii, r_o, 3 * dd] = val
    img[ii, r_c, 3 * dd + 2] = val

    if style == "jiang":
        vh = np.rint(np.clip(rel_v, 0, 1) * vol_rows).astype(np.int64)    # bar height in px
        rv = np.arange(vol_rows)[None, None, :]
        vbar = rv >= (vol_rows - vh[..., None])
        img[:, height - vol_rows:, 1::3] = np.where(vbar, 255, 0).astype(np.uint8).transpose(0, 2, 1)
    return img


def make_images(ohlcv: pd.DataFrame, window: int = 20, height: int = 32,
                style: str = "volume_intensity", min_intensity: float = 0.1
                ) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Sliding windows over a cleaned OHLCV frame.

    Image k covers rows k .. k+window-1; its date is the LAST day in the window, i.e.
    everything in the image is known at the close of that date.
    """
    if len(ohlcv) < window:
        raise ValueError("fewer rows than the window length")
    sw = lambda col: np.lib.stride_tricks.sliding_window_view(ohlcv[col].to_numpy(float), window)
    imgs = render_images(sw("Open"), sw("High"), sw("Low"), sw("Close"), sw("Volume"),
                         height=height, style=style, min_intensity=min_intensity)
    return imgs, pd.DatetimeIndex(ohlcv.index[window - 1:])
