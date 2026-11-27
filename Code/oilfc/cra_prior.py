"""Frozen Chronos-2 prior: quantiles of the h-step WTI log return at every origin.

The context for origin t is the log WTI price and the log covariate levels on
rows t-C+1 .. t, so the prior uses no information after the close of day t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def chronos_inputs(lv: pd.DataFrame, origins: np.ndarray, context: int, covariates: list[str]) -> list[dict]:
    lp = np.log(lv["WTI"].values)
    cov = {c: np.log(lv[c].values) for c in covariates}
    out = []
    for i in origins:
        lo = max(0, i - context + 1)
        out.append({"target": lp[lo: i + 1].astype(np.float32),
                    "past_covariates": {c: v[lo: i + 1].astype(np.float32) for c, v in cov.items()}})
    return out


def chronos_prior(lv: pd.DataFrame, cfg: dict, origins: np.ndarray, device: str = "cuda") -> pd.DataFrame:
    """Long frame: origin_date, horizon, q10, q50, q90 (log-return quantiles)."""
    import torch
    from chronos import Chronos2Pipeline

    c = cfg["chronos"]
    horizons = cfg["framing"]["horizons"]
    pipe = Chronos2Pipeline.from_pretrained(c["model"], device_map=device, dtype=torch.float32)
    lp = np.log(lv["WTI"].values)

    rows = []
    for start in range(0, len(origins), 4 * c["batch_size"]):
        chunk = origins[start: start + 4 * c["batch_size"]]
        q, _ = pipe.predict_quantiles(chronos_inputs(lv, chunk, c["context_length"], c["covariates"]),
                                      prediction_length=max(horizons), quantile_levels=c["quantiles"],
                                      batch_size=c["batch_size"])
        for i, qi in zip(chunk, q):
            qi = qi[0].float().cpu().numpy()             # (prediction_length, n_quantiles)
            for h in horizons:
                rows.append((lv.index[i], h, *(qi[h - 1] - lp[i])))
        print(f"   Chronos-2 origins {min(start + len(chunk), len(origins))}/{len(origins)}")
    return pd.DataFrame(rows, columns=["origin_date", "horizon", "q10", "q50", "q90"])
