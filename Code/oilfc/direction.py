"""
oilfc/direction.py  -  Directional (up/down) forecast evaluation (owner: image CNN)

The image CNN of Ren et al. (2024) is a classifier, so it cannot be scored with
RMSE. To compare it with the other models we turn every price forecast into a
direction forecast and score all of them on the same days:

    up  <=>  P_{t+h} >= P_t          (paper, eq. 5: Ret >= 0 counts as "up")

The direction is taken from the price CHANGE, not the simple return, because a
simple return has the wrong sign when the base price is negative (WTI 2020-04-20).

A random walk forecasts zero change, which under this convention is always "up",
so the random walk doubles as the "always up" benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def sign_forecast(pred_price, base_price) -> np.ndarray:
    return (np.asarray(pred_price, float) >= np.asarray(base_price, float)).astype(int)


def f1_balanced(true_up, pred_up) -> tuple[float, float, float]:
    """Paper eq. (8)-(9): mean of the F1 scores of the 'up' and 'down' classes."""
    t, p = np.asarray(true_up).astype(bool), np.asarray(pred_up).astype(bool)
    out = []
    for cls in (True, False):
        tp = np.sum((p == cls) & (t == cls))
        fp = np.sum((p == cls) & (t != cls))
        fn = np.sum((p != cls) & (t == cls))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return (out[0] + out[1]) / 2, out[0], out[1]


def pesaran_timmermann(true_up, pred_up) -> tuple[float, float]:
    """Pesaran & Timmermann (1992) test of directional accuracy vs. independence.

    H0: forecasts and outcomes are independent. One-sided p-value (good forecasts
    give a positive statistic). Undefined (NaN) when the forecast never changes sign,
    e.g. the random walk. Assumes serially independent errors, so it is only exact
    for h = 1; with overlapping h > 1 targets treat the p-value as optimistic.
    """
    y, x = np.asarray(true_up, float), np.asarray(pred_up, float)
    n = len(y)
    py, px, p_hat = y.mean(), x.mean(), np.mean(y == x)
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if v_hat - v_star <= 0 or px in (0.0, 1.0):
        return float("nan"), float("nan")
    s = (p_hat - p_star) / np.sqrt(v_hat - v_star)
    return float(s), float(stats.norm.sf(s))


def auc(true_up, score) -> float:
    """ROC AUC via the Mann-Whitney U statistic (no sklearn needed)."""
    y, s = np.asarray(true_up).astype(bool), np.asarray(score, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def direction_metrics(true_up, pred_up, prob_up=None) -> dict:
    t, p = np.asarray(true_up).astype(int), np.asarray(pred_up).astype(int)
    n = len(t)
    if n == 0:
        return {"n": 0}
    acc = float(np.mean(t == p))
    f1, f1_up, f1_down = f1_balanced(t, p)
    pt, pt_p = pesaran_timmermann(t, p)
    out = {
        "n": n,
        "Acc": acc,
        "F1": f1, "F1_up": f1_up, "F1_down": f1_down,
        "share_up_true": float(t.mean()),          # = accuracy of "always up" / random walk
        "share_up_pred": float(p.mean()),
        "binom_p_vs_50": float(stats.binomtest(int(np.sum(t == p)), n, 0.5, alternative="greater").pvalue),
        "PT_stat": pt, "PT_p": pt_p,
    }
    if prob_up is not None:
        out["AUC"] = auc(t, prob_up)
    return out
