"""Data for the Chronos-anchored gated residual LSTM (issue #10).

Conventions (same as oilfc/data.py)
-----------------------------------
* Row t holds information known at the close of trading day t. The calendar is
  the set of days with a WTI spot print, so holidays do not create zero returns.
* Target for horizon h at origin t: r = log P_{t+h} - log P_t (h trading days).
* A sample belongs to a split by its TARGET date. For refit year Y:
  train targets <= end of Y-1-val_years, val targets in the year(s) before Y,
  evaluation targets in Y.
* Scalers are fit on training rows of each refit only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import path

FEATURES = ["r_wti", "r_brent", "brent_wti", "mom5", "mom20", "mom60", "log_rv20",
            "crack_ho", "log_ovx", "log_vix", "r_dxy", "r_xle", "d_ust10y", "d_be10y", "log_gpr"]
PRIOR_FEATURES = ["prior_q10", "prior_q50", "prior_q90", "log_sigma"]


# -----------------------------------------------------------------------------
# Levels: calendar alignment and nonpositive prices
# -----------------------------------------------------------------------------
def asof_on_calendar(s: pd.Series, cal: pd.DatetimeIndex, limit: int) -> pd.Series:
    """Last observation on or before each calendar day; NaN if older than `limit` calendar rows."""
    obs = s.index
    k = obs.searchsorted(cal, side="right") - 1
    vals = np.where(k >= 0, s.values[np.clip(k, 0, None)], np.nan)
    last_obs = obs[np.clip(k, 0, None)]
    age = np.arange(len(cal)) - cal.searchsorted(last_obs, side="left")
    vals[(k < 0) | (age > limit)] = np.nan
    return pd.Series(vals, index=cal)


def build_levels(cfg: dict) -> pd.DataFrame:
    """Align raw series on the WTI trading calendar. Adds `interp` = 1 where WTI was interpolated."""
    d = cfg["data"]
    raw = lambda name: pd.read_csv(path(cfg, "raw_dir", f"{name}.csv"), index_col=0, parse_dates=True)
    fred, yahoo, gpr = raw("fred"), raw("yahoo"), raw("gpr")

    # GPR of day d becomes usable on day d + lag (calendar days, incl. weekends)
    gpr = gpr.shift(d["gpr_lag_days"], freq="D")
    allx = pd.concat([fred.drop(columns="WTI"), yahoo, gpr], axis=1, sort=True)

    wti = fred["WTI"].dropna()
    cal = wti.index
    others = pd.DataFrame(index=cal)
    for c in allx.columns:
        others[c] = asof_on_calendar(allx[c].dropna(), cal, d["ffill_limit"])

    lv = pd.concat([wti.rename("WTI"), others], axis=1)
    bad = lv["WTI"] <= 0
    lv["interp"] = bad.astype(int)
    if bad.any():
        lv["WTI"] = lv["WTI"].mask(bad).interpolate(method="time")
    lv = lv.dropna()
    lv.index.name = "Date"
    return lv


# -----------------------------------------------------------------------------
# Features, volatility, targets
# -----------------------------------------------------------------------------
def make_features(lv: pd.DataFrame) -> pd.DataFrame:
    """15 inputs, each a function of rows <= t only."""
    lp = np.log(lv["WTI"])
    r = lp.diff()
    f = pd.DataFrame(index=lv.index)
    f["r_wti"] = r
    f["r_brent"] = np.log(lv["BRENT"]).diff()
    f["brent_wti"] = np.log(lv["BRENT"]) - lp
    for k in (5, 20, 60):
        f[f"mom{k}"] = lp - lp.shift(k)
    f["log_rv20"] = np.log(np.sqrt((r ** 2).rolling(20).mean()) + 1e-6)
    f["crack_ho"] = np.log(42.0 * lv["HEATOIL"]) - lp
    f["log_ovx"] = np.log(lv["OVX"])
    f["log_vix"] = np.log(lv["VIX"])
    f["r_dxy"] = np.log(lv["DXY"]).diff()
    f["r_xle"] = np.log(lv["XLE"]).diff()
    f["d_ust10y"] = lv["UST10Y"].diff()
    f["d_be10y"] = lv["BREAKEVEN10Y"].diff()
    f["log_gpr"] = np.log1p(lv["GPR"])
    return f[FEATURES]


def ewma_sigma(lv: pd.DataFrame, lam: float) -> pd.Series:
    """Daily ex-ante volatility from returns up to and including t (RiskMetrics recursion)."""
    r = np.log(lv["WTI"]).diff()
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).clip(lower=1e-4)


def targets(lv: pd.DataFrame, h: int) -> tuple[pd.Series, pd.Series]:
    lp = np.log(lv["WTI"])
    y = lp.shift(-h) - lp
    tdate = pd.Series(lv.index, index=lv.index).shift(-h)
    return y, tdate


def loss_mask(lv: pd.DataFrame, h: int, after_rows: int) -> np.ndarray:
    """False for origins whose inputs or target touch an interpolated WTI price."""
    flag = lv["interp"].values.astype(bool)
    ok = np.ones(len(lv), bool)
    for i in np.flatnonzero(flag):
        ok[max(0, i - h): i + after_rows + 1] = False
    return ok


# -----------------------------------------------------------------------------
# Refit splits and windows
# -----------------------------------------------------------------------------
@dataclass
class Split:
    X: np.ndarray            # (N, lookback, n_features), scaled
    prior: np.ndarray        # (N, 4): prior quantiles / sigma_h, log sigma_h (scaled)
    z: np.ndarray            # (N,): r / sigma_h
    y_raw: np.ndarray        # (N,): r
    sigma_h: np.ndarray      # (N,)
    m_raw: np.ndarray        # (N, 3): prior return quantiles
    base_price: np.ndarray
    true_price: np.ndarray
    origin_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex


@dataclass
class Refit:
    year: int
    horizon: int
    splits: dict[str, Split]
    features: list[str]
    x_scaler: StandardScaler
    p_scaler: StandardScaler


def refit_bounds(year: int, val_years: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    train_end = pd.Timestamp(f"{year - 1 - val_years}-12-31")
    val_end = pd.Timestamp(f"{year - 1}-12-31")
    eval_end = pd.Timestamp(f"{year}-12-31")
    return train_end, val_end, eval_end


def build_refit(cfg: dict, lv: pd.DataFrame, feats: pd.DataFrame, prior: pd.DataFrame | None,
                h: int, year: int, features: list[str] | None = None) -> Refit:
    """Windows for one refit year. `prior` holds Chronos return quantiles q10/q50/q90 per (origin, h)."""
    fr = cfg["framing"]
    L = fr["lookback"]
    features = features or FEATURES
    train_end, val_end, eval_end = refit_bounds(year, cfg["evaluation"]["val_years"])

    y, tdate = targets(lv, h)
    sig_h = ewma_sigma(lv, fr["ewma_lambda"]) * np.sqrt(h)
    price = lv["WTI"]
    fut_price = price.shift(-h)
    ok = loss_mask(lv, h, cfg["data"]["interp_exclusion_rows"])

    F = feats[features]
    first = int(np.argmax(F.notna().all(axis=1).values))       # first row with every feature
    x_scaler = StandardScaler().fit(F.loc[(F.index <= train_end)].iloc[first:].values)
    X_all = x_scaler.transform(F.fillna(0.0).values).astype(np.float32)

    if prior is not None:
        q = prior.xs(h, level="horizon")[["q10", "q50", "q90"]].reindex(lv.index)
        m = np.sort(q.values, axis=1)                          # guard against crossing
    else:
        m = np.zeros((len(lv), 3))
    p_raw = np.column_stack([m / sig_h.values[:, None], np.log(sig_h.values)])

    idx = np.arange(first + L - 1, len(lv))
    idx = idx[~np.isnan(y.values[idx]) & ~np.isnan(m[idx]).any(axis=1)]
    td = pd.DatetimeIndex(tdate.values[idx])
    masks = {
        "train": (td <= train_end) & ok[idx],
        "val": (td > train_end) & (td <= val_end) & ok[idx],
        "eval": (td > val_end) & (td <= eval_end),
    }
    p_scaler = StandardScaler().fit(p_raw[idx[masks["train"]]])
    P_all = p_scaler.transform(p_raw).astype(np.float32)

    splits = {}
    for name, msk in masks.items():
        ii = idx[msk]
        splits[name] = Split(
            X=np.stack([X_all[i - L + 1: i + 1] for i in ii]) if len(ii) else np.empty((0, L, len(features)), np.float32),
            prior=P_all[ii],
            z=(y.values[ii] / sig_h.values[ii]).astype(np.float32),
            y_raw=y.values[ii],
            sigma_h=sig_h.values[ii],
            m_raw=m[ii],
            base_price=price.values[ii],
            true_price=fut_price.values[ii],
            origin_dates=pd.DatetimeIndex(lv.index[ii]),
            target_dates=pd.DatetimeIndex(tdate.values[ii]),
        )
    if (splits["eval"].true_price <= 0).any() or (splits["eval"].base_price <= 0).any():
        raise ValueError(f"nonpositive WTI price in evaluation year {year}")
    return Refit(year, h, splits, list(features), x_scaler, p_scaler)


def load_cra_data(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lv = pd.read_csv(path(cfg, "processed_dir", "levels.csv"), index_col=0, parse_dates=True)
    return lv, make_features(lv)


def load_prior(cfg: dict) -> pd.DataFrame:
    f = path(cfg, "results_dir", "prior", "chronos_prior.csv")
    if not f.exists():
        raise FileNotFoundError(f"{f} not found - run 16_cra_chronos_prior.py first.")
    return pd.read_csv(f, parse_dates=["origin_date"]).set_index(["origin_date", "horizon"])
