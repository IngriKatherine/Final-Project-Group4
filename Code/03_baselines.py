"""
03_baselines.py  -  Naive random walk, ARIMA and ARIMAX benchmarks (owner: data pipeline & baselines)

For every horizon h in config.yaml, on exactly the same test forecast origins the
RNNs use:
  * Random walk : P_hat_{t+h} = P_t   (hard to beat for oil prices; the key benchmark)
  * ARIMA(p,d,q) on log prices, rolling origin: parameters are refit every
    `refit_every` steps and the state is updated with each new observation.
  * ARIMAX(p,0,q) on daily log returns with the drivers' previous-day returns as
    exogenous inputs (as in Okasha & Zaher 2026). Same d as ARIMA, no drift, so the
    only difference from ARIMA is the drivers. Future driver moves are set to 0.

The ARIMA order is either `baselines.arima_order` or, with `baselines.auto: true`,
chosen by pmdarima.auto_arima (AIC) on TRAINING log prices only.

Outputs
  results/predictions/baseline_{rw|arima|arimax}_h{h}.csv
  results/tables/baselines_metrics.csv
  results/tables/baselines_arima_order.json

Usage
  python 03_baselines.py                 # all horizons
  python 03_baselines.py --skip-arima    # random walk only (fast)
  python 03_baselines.py --skip-arimax   # random walk + ARIMA
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from oilfc.data import build_datasets, load_levels, make_features, predictions_frame
from oilfc.metrics import all_metrics
from oilfc.utils import load_config, path, save_json

warnings.filterwarnings("ignore")


def choose_order(cfg: dict, logp: pd.Series) -> tuple[int, int, int]:
    """Fixed order from the config, or auto_arima on train-period log prices only."""
    b = cfg["baselines"]
    if not b.get("auto", False):
        return tuple(b["arima_order"])
    import pmdarima as pm
    train = logp.loc[: pd.Timestamp(cfg["split"]["train_end"])].values
    fit = pm.auto_arima(train, d=1, max_p=b.get("max_p", 5), max_q=b.get("max_q", 5),
                        seasonal=False, with_intercept=False, information_criterion="aic",
                        stepwise=True, suppress_warnings=True, error_action="ignore")
    print(f"auto_arima on train ({len(train)} obs) chose order {fit.order}, AIC {fit.aic():.1f}")
    return tuple(int(v) for v in fit.order)


def exog_frame(cfg: dict, levels: pd.DataFrame) -> pd.DataFrame:
    """Drivers' returns / differences lagged one day, aligned to the log-return index."""
    target = cfg["data"]["target"]
    feats = make_features(levels, cfg)
    cols = cfg["baselines"].get("exog", "all")
    cols = [c for c in feats.columns if c != target] if cols == "all" else list(cols)
    return feats[cols].shift(1).fillna(0.0)


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


def arimax_rolling_forecasts(logp: pd.Series, exog: pd.DataFrame, origins: pd.DatetimeIndex, order,
                             max_h: int, refit_every: int) -> dict[pd.Timestamp, np.ndarray]:
    """Same as arima_rolling_forecasts, but models log returns with lagged drivers as exog.

    Row t of `exog` holds the drivers' moves on day t-1, so the step-1 exog at origin t
    (row t+1) is already known at t. Steps 2..h use 0 for the unknown future moves.
    """
    p_, d, q = order
    if d != 1:
        raise ValueError(f"ARIMAX models daily log returns and needs d = 1, got order {order}")
    ret = logp.diff().iloc[1:]
    X = exog.reindex(ret.index).values
    pos = ret.index.get_indexer(origins)
    out, res, last_pos = {}, None, None
    for k, (origin, p) in enumerate(zip(origins, pos)):
        if res is None or k % refit_every == 0:
            res = ARIMA(ret.values[: p + 1], exog=X[: p + 1], order=(p_, d - 1, q), trend="n").fit()
        else:
            res = res.extend(ret.values[last_pos + 1: p + 1], exog=X[last_pos + 1: p + 1])
        x_future = np.zeros((max_h, X.shape[1]))
        if p + 1 < len(X):
            x_future[0] = X[p + 1]
        out[origin] = logp.loc[origin] + np.cumsum(res.forecast(steps=max_h, exog=x_future))
        last_pos = p
        if k % 100 == 0:
            print(f"   ARIMAX origin {k}/{len(origins)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-arima", action="store_true", help="random walk only")
    ap.add_argument("--skip-arimax", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    horizons = cfg["framing"]["horizons"]
    target = cfg["data"]["target"]
    refit_every = cfg["baselines"]["refit_every"]
    levels = load_levels(cfg)
    logp = np.log(levels[target])
    rows = []

    forecasts = {}
    if not args.skip_arima:
        order = choose_order(cfg, logp)
        save_json({"order": list(order), "auto": bool(cfg["baselines"].get("auto", False))},
                  path(cfg, "results_dir", "tables", "baselines_arima_order.json"))
        # Union of test origins over all horizons (splits use the TARGET date, so longer
        # horizons have origins that start a few days earlier).
        origins = pd.DatetimeIndex(sorted(set().union(*[
            build_datasets(cfg, "price_only", h, levels=levels).splits["test"].origin_dates for h in horizons])))
        print(f"Fitting rolling ARIMA{order} on {len(origins)} origins ...")
        forecasts["arima"] = ("price_only", arima_rolling_forecasts(logp, origins, order, max(horizons), refit_every))
        if not args.skip_arimax:
            exog = exog_frame(cfg, levels)
            print(f"Fitting rolling ARIMAX{order} with {exog.shape[1]} drivers: {', '.join(exog.columns)}")
            forecasts["arimax"] = ("all", arimax_rolling_forecasts(logp, exog, origins, order,
                                                                  max(horizons), refit_every))

    for h in horizons:
        test = build_datasets(cfg, "price_only", h, levels=levels).splits["test"]

        # Random walk
        rw = predictions_frame(test, test.base_price)
        rw.to_csv(path(cfg, "results_dir", "predictions", f"baseline_rw_h{h}.csv"), index=False)
        rows.append({"model": "random_walk", "features": "price_only", "horizon": h,
                     **all_metrics(rw.true_price, rw.pred_price, rw.base_price)})

        # ARIMA / ARIMAX
        for name, (features, fc) in forecasts.items():
            pred = np.exp([fc[o][h - 1] for o in test.origin_dates])
            df = predictions_frame(test, pred)
            df.to_csv(path(cfg, "results_dir", "predictions", f"baseline_{name}_h{h}.csv"), index=False)
            rows.append({"model": name, "features": features, "horizon": h,
                         **all_metrics(df.true_price, df.pred_price, df.base_price)})

    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results_dir", "tables", "baselines_metrics.csv"), index=False)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
