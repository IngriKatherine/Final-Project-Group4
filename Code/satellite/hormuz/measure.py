"""Measure each detected vessel (length, width, heading) and convert size to oil capacity.

Length comes from the bright hull in the near-infrared band (B08): threshold the crop
against the surrounding water, keep the connected blob under the detection centre, and
use the blob's principal axes. For a uniform bar of length L the variance along its
axis is L^2 / 12, so L = sqrt(12 * lambda_1).
Wakes and very close neighbours inflate the estimate; treat it as a size class, not a
precise number.
"""
from __future__ import annotations

import numpy as np


def hull_size(nir: np.ndarray, cx: float, cy: float, crop: int = 64, k_mad: float = 4.0,
              res_m: float = 10.0) -> dict:
    """Return length_m, width_m, heading_deg (0-180, clockwise from north) and n_px."""
    from scipy import ndimage

    h, w = nir.shape
    half = crop // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    xs, ys = max(x0, 0), max(y0, 0)
    xe, ye = min(x0 + crop, w), min(y0 + crop, h)
    win = nir[ys:ye, xs:xe].astype(np.float32)
    nan = dict(length_m=np.nan, width_m=np.nan, heading_deg=np.nan, n_px=0)
    if win.size == 0:
        return nan
    valid = win > 0
    if valid.sum() < 50:
        return nan
    med = np.median(win[valid])
    mad = np.median(np.abs(win[valid] - med)) * 1.4826 + 1.0
    bright = valid & (win > med + k_mad * mad)
    labels, n = ndimage.label(bright)
    if n == 0:
        return nan
    # blob at the centre, or the nearest blob to the centre
    ccx, ccy = int(round(cx)) - xs, int(round(cy)) - ys
    ccx, ccy = np.clip(ccx, 0, win.shape[1] - 1), np.clip(ccy, 0, win.shape[0] - 1)
    lab = labels[ccy, ccx]
    if lab == 0:
        yy, xx = np.nonzero(labels)
        j = np.argmin((yy - ccy) ** 2 + (xx - ccx) ** 2)
        if (yy[j] - ccy) ** 2 + (xx[j] - ccx) ** 2 > 5 ** 2:
            return nan
        lab = labels[yy[j], xx[j]]
    yy, xx = np.nonzero(labels == lab)
    if len(yy) < 3:
        return dict(length_m=res_m * 1.5, width_m=res_m, heading_deg=np.nan, n_px=len(yy))
    coords = np.stack([xx, -yy]).astype(np.float64)          # x east, y north
    cov = np.cov(coords)
    evals, evecs = np.linalg.eigh(cov)                          # ascending
    l1, l2 = max(evals[1], 0), max(evals[0], 0)
    length = np.sqrt(12 * l1 + 1) * res_m                       # +1: a pixel has width
    width = np.sqrt(12 * l2 + 1) * res_m
    vx, vy = evecs[:, 1]
    heading = (np.degrees(np.arctan2(vx, vy)) + 360) % 180      # axis only: 0-180
    return dict(length_m=float(length), width_m=float(width), heading_deg=float(heading),
                n_px=int(len(yy)))


def dwt_from_length(length_m, cfg: dict):
    """Deadweight tonnes from length with a power law anchored on a VLCC."""
    s = cfg["supply"]
    L = np.asarray(length_m, dtype=float)
    return s["dwt_ref_tonnes"] * (L / s["dwt_ref_length_m"]) ** s["dwt_exponent"]


def size_class(length_m: float, cfg: dict) -> str:
    s = cfg["supply"]
    if not np.isfinite(length_m):
        return "unknown"
    if length_m < s["small_m"]:
        return "small"
    if length_m < s["large_m"]:
        return "medium"
    return "large"
