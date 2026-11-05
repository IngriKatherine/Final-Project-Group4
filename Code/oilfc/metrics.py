"""Forecast accuracy metrics and the Diebold-Mariano test.

All metrics are computed on PRICE LEVELS (USD/bbl) in the test period, even
when the network predicts log returns, so every model is compared on the same scale.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat, eps: float = 1e-8) -> float:
    """Mean absolute percentage error in %. Unstable near zero prices (WTI, April 2020)."""
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(100 * np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))))


def directional_accuracy(y, yhat, base) -> float:
    """Share of forecasts that get the sign of the move (P_{t+h} - P_t) right."""
    y, yhat, base = map(np.asarray, (y, yhat, base))
    return float(np.mean(np.sign(y - base) == np.sign(yhat - base)))


def all_metrics(y, yhat, base=None) -> dict:
    out = {"RMSE": rmse(y, yhat), "MAE": mae(y, yhat), "MAPE": mape(y, yhat)}
    if base is not None:
        out["DirAcc"] = directional_accuracy(y, yhat, base)
    return out


def diebold_mariano(y, yhat_1, yhat_2, h: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    H0: equal predictive accuracy. A negative statistic means model 1 is more accurate.
    Uses a Newey-West variance with h-1 lags (forecast errors are MA(h-1) for h-step forecasts).
    """
    y, f1, f2 = map(np.asarray, (y, yhat_1, yhat_2))
    d = np.abs(y - f1) ** power - np.abs(y - f2) ** power
    n = len(d)
    d_bar = d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[: n - k] - d_bar)) / n for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return {"DM": np.nan, "p_value": np.nan}
    dm = d_bar / np.sqrt(var_d)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * dm
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return {"DM": float(stat), "p_value": float(p)}
