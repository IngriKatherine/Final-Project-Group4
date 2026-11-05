"""
03_baselines.py  -  Naive random walk and ARIMA benchmarks (owner: data pipeline & baselines)

For every horizon h in config.yaml, on exactly the same test forecast origins the
RNNs use:
  * Random walk : P_hat_{t+h} = P_t   (hard to beat for oil prices; the key benchmark)
  * ARIMA(p,d,q) on log prices, rolling origin: parameters are refit every
    `refit_every` steps and the state is updated with each new observation.

Outputs
  results/predictions/baseline_{rw|arima}_h{h}.csv
  results/tables/baselines_metrics.csv

Usage
  python 03_baselines.py                 # all horizons
  python 03_baselines.py --skip-arima    # random walk only (fast)
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from oilfc.data import build_datasets, load_levels, predictions_frame
from oilfc.metrics import all_metrics
from oilfc.utils import load_config, path

warnings.filterwarnings("ignore")


def arima_rolling_forecasts(logp: pd.Series, origins: pd.DatetimeIndex, order, max_h: int,
                            refit_every: int) -> dict[pd.Timestamp, np.ndarray]:
    """Return {origin_date: array of max_h log-price forecasts} using only data <= origin."""
    pos = logp.index.get_indexer(origins)
    out, res, last_pos = {}, None, None
    for k, (origin, p) in enumerate(zip(origins, pos)):
        if res is None or k % refit_every == 0:
            res = ARIMA(logp.values[: p + 1], order=order).fit()
        else:
            res = res.extend(logp.values[last_pos + 1: p + 1])
        out[origin] = np.asarray(res.forecast(steps=max_h))
        last_pos = p
        if k % 100 == 0:
            print(f"   ARIMA origin {k}/{len(origins)}")
    return out
    # TODO: ARIMAX with exogenous drivers (as in Okasha & Zaher 2026) - pass `exog=`
    #       and use the realised exog values only up to the origin.
    # TODO: if baselines.auto is true, choose the order with pmdarima.auto_arima on train only.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arima", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    horizons = cfg["framing"]["horizons"]
    target = cfg["data"]["target"]
    levels = load_levels(cfg)
    logp = np.log(levels[target])
    rows = []

    arima_fc = None
    if not args.skip_arima:
        # Union of test origins over all horizons (splits use the TARGET date, so longer
        # horizons have origins that start a few days earlier).
        origins = pd.DatetimeIndex(sorted(set().union(*[
            build_datasets(cfg, "price_only", h, levels=levels).splits["test"].origin_dates for h in horizons])))
        print(f"Fitting rolling ARIMA{tuple(cfg['baselines']['arima_order'])} on {len(origins)} origins ...")
        arima_fc = arima_rolling_forecasts(logp, origins, tuple(cfg["baselines"]["arima_order"]),
                                           max(horizons), cfg["baselines"]["refit_every"])

    for h in horizons:
        test = build_datasets(cfg, "price_only", h, levels=levels).splits["test"]

        # Random walk
        rw = predictions_frame(test, test.base_price)
        rw.to_csv(path(cfg, "results_dir", "predictions", f"baseline_rw_h{h}.csv"), index=False)
        rows.append({"model": "random_walk", "features": "price_only", "horizon": h,
                     **all_metrics(rw.true_price, rw.pred_price, rw.base_price)})

        # ARIMA
        if arima_fc is not None:
            pred = np.exp([arima_fc[o][h - 1] for o in test.origin_dates])
            ar = predictions_frame(test, pred)
            ar.to_csv(path(cfg, "results_dir", "predictions", f"baseline_arima_h{h}.csv"), index=False)
            rows.append({"model": "arima", "features": "price_only", "horizon": h,
                         **all_metrics(ar.true_price, ar.pred_price, ar.base_price)})

    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results_dir", "tables", "baselines_metrics.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
