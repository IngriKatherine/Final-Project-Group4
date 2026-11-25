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
    """Share of forecasts that get the sign of the move (P_{t+h} - P_t) right.

    NaN for a forecaster that never predicts a move (the random walk): its "accuracy"
    would only count the days the price happened not to change.
    """
    y, yhat, base = map(np.asarray, (y, yhat, base))
    if np.all(yhat == base):
        return float("nan")
    return float(np.mean(np.sign(y - base) == np.sign(yhat - base)))


def all_metrics(y, yhat, base=None) -> dict:
    out = {"RMSE": rmse(y, yhat), "MAE": mae(y, yhat), "MAPE": mape(y, yhat)}
    if base is not None:
        out["DirAcc"] = directional_accuracy(y, yhat, base)
    return out


def oos_r2(y, yhat, yhat_bench) -> float:
    """Out-of-sample R^2 against a benchmark (Campbell & Thompson 2008): 1 - MSE / MSE_bench.

    Positive means the model beats the benchmark (usually the random walk), e.g. 0.02 = 2% lower MSE.
    """
    y, yhat, yb = map(np.asarray, (y, yhat, yhat_bench))
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - yb) ** 2))


def _newey_west_var(d: np.ndarray, h: int) -> float:
    """Long-run variance of the mean of d, Newey-West (Bartlett weights) with h-1 lags.

    The unweighted sum of autocovariances is not guaranteed positive and can badly understate
    the variance: for ARIMA vs. the random walk at h = 5 it gave DM = -3.5 (p < 0.001) for a
    0.05% MSE gain, against -1.2 (p = 0.25) with Bartlett weights at any lag length from 4 to 40.
    """
    n, d_bar = len(d), d.mean()
    gamma = [np.sum((d[k:] - d_bar) * (d[: n - k] - d_bar)) / n for k in range(h)]
    return (gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))) / n


def clark_west(y, yhat_small, yhat_large, h: int = 1) -> dict:
    """Clark-West (2007) test for NESTED models, e.g. random walk inside ARIMA, ARIMA inside ARIMAX.

    DM is biased against the larger model here: under H0 it still pays for estimating
    parameters that are truly zero. CW adds that noise term back:
        f_t = e_small^2 - (e_large^2 - (yhat_small - yhat_large)^2)
    H0: equal accuracy, H1: the larger model is more accurate (one-sided, standard normal).
    A positive statistic favours the larger model.
    """
    y, f1, f2 = map(np.asarray, (y, yhat_small, yhat_large))
    f = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)
    var_f = _newey_west_var(f, h)
    if var_f <= 0:
        return {"CW": np.nan, "CW_p_value": np.nan}
    stat = f.mean() / np.sqrt(var_f)
    return {"CW": float(stat), "CW_p_value": float(stats.norm.sf(stat))}


def diebold_mariano(y, yhat_1, yhat_2, h: int = 1, power: int = 2) -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    H0: equal predictive accuracy. A negative statistic means model 1 is more accurate.
    Uses a Newey-West (Bartlett) variance with h-1 lags (forecast errors are MA(h-1) for h-step forecasts).
    """
    y, f1, f2 = map(np.asarray, (y, yhat_1, yhat_2))
    d = np.abs(y - f1) ** power - np.abs(y - f2) ** power
    n = len(d)
    d_bar = d.mean()
    var_d = _newey_west_var(d, h)
    if var_d <= 0:
        return {"DM": np.nan, "p_value": np.nan}
    dm = d_bar / np.sqrt(var_d)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * dm
    p = 2 * stats.t.sf(np.abs(stat), df=n - 1)
    return {"DM": float(stat), "p_value": float(p)}
