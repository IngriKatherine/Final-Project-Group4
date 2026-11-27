"""Extra evaluation statistics for issue #10 (the shared ones live in metrics.py)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def pesaran_timmermann(y_move, f_move) -> dict:
    """Pesaran-Timmermann (1992) test of directional skill. One-sided, H1: skill.

    NaN when the forecast (or the outcome) never changes sign: the test is undefined.
    """
    x = np.asarray(y_move) > 0
    y = np.asarray(f_move) > 0
    n = len(x)
    px, py = x.mean(), y.mean()
    p_hat = np.mean(x == y)
    p_star = py * px + (1 - py) * (1 - px)
    v_hat = p_star * (1 - p_star) / n
    v_star = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
              + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    if v_hat - v_star <= 0:
        return {"PT": np.nan, "PT_p_value": np.nan, "hit_rate": float(p_hat), "n": n}
    s = (p_hat - p_star) / np.sqrt(v_hat - v_star)
    return {"PT": float(s), "PT_p_value": float(stats.norm.sf(s)), "hit_rate": float(p_hat), "n": n}


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (NaN entries are left out of the family)."""
    p = np.asarray(p_values, float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))
