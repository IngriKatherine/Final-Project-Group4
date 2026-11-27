"""
16_cra_chronos_prior.py  -  Cache the frozen Chronos-2 prior for every forecast origin (issue #10)

Zero-shot Chronos-2 with past-only covariates (cra_config.yaml: chronos), run
once per origin. The cached quantiles are both an input to the gated residual
LSTM and the standalone Chronos-2 benchmark.

Outputs
  results/cra/prior/chronos_prior.csv     origin_date, horizon, q10, q50, q90 (log returns)

Usage
  python 16_cra_chronos_prior.py
"""
from __future__ import annotations

import time

import numpy as np

from oilfc.cra_data import load_cra_data
from oilfc.cra_prior import chronos_prior
from oilfc.utils import CODE_DIR, get_device, load_config, path


def main():
    cfg = load_config(CODE_DIR / "cra_config.yaml")
    lv, _ = load_cra_data(cfg)
    origins = np.arange(cfg["framing"]["lookback"] + 60, len(lv))
    device = "cuda" if get_device().type == "cuda" else "cpu"
    print(f"Chronos-2 prior for {len(origins)} origins ({lv.index[origins[0]].date()} to "
          f"{lv.index[-1].date()}) on {device}")
    t0 = time.time()
    prior = chronos_prior(lv, cfg, origins, device)
    out = path(cfg, "results_dir", "prior", "chronos_prior.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    prior.to_csv(out, index=False)
    print(f"Saved {len(prior)} rows to {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
