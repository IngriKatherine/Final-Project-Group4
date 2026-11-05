"""
06_run_experiments.py  -  Main experiment grid (owners: modeling + SHAP)

Research question: does SHAP-guided feature selection help where simply adding
more features does not?

Grid (from config.yaml -> experiments):
  model       in {lstm, gru}
  feature set in {price_only, all, shap_selected}
  horizon     in framing.horizons
  seed        in experiments.seeds

Finished runs are skipped, so the script can be stopped and restarted
(useful on the AWS GPU machine; run it inside tmux/screen).

Outputs
  results/tables/runs/*.json           one file per run (written by 04_train_rnn.py)
  results/tables/main_results_raw.csv  one row per run
  results/tables/main_results.csv      mean and std over seeds

Usage
  python 06_run_experiments.py
  python 06_run_experiments.py --models gru --horizons 1 5
"""
from __future__ import annotations

import argparse
import itertools

import pandas as pd

from oilfc.utils import import_script, load_config, path

train_script = import_script("04_train_rnn.py")


def flatten(summary: dict) -> dict:
    row = {k: summary[k] for k in ("run", "model", "features", "horizon", "seed", "n_params")}
    row["n_features"] = len(summary["feature_names"])
    row["feature_names"] = "|".join(summary["feature_names"])
    row["epochs"] = summary["history"]["epochs_run"]
    row.update({f"val_{k}": v for k, v in summary["val"].items()})
    row.update({f"test_{k}": v for k, v in summary["test"].items()})
    return row


def main():
    cfg = load_config()
    e = cfg["experiments"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=e["models"])
    ap.add_argument("--features", nargs="+", default=e["feature_sets"])
    ap.add_argument("--horizons", nargs="+", type=int, default=cfg["framing"]["horizons"])
    ap.add_argument("--seeds", nargs="+", type=int, default=e["seeds"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = []
    grid = list(itertools.product(args.models, args.features, args.horizons, args.seeds))
    for i, (m, f, h, s) in enumerate(grid, 1):
        print(f"[{i}/{len(grid)}]", end=" ")
        rows.append(flatten(train_script.run_single(cfg, m, f, h, s, force=args.force)))

    raw = pd.DataFrame(rows)
    raw.to_csv(path(cfg, "results_dir", "tables", "main_results_raw.csv"), index=False)

    metric_cols = [c for c in raw.columns if c.startswith(("val_", "test_"))]
    agg = raw.groupby(["model", "features", "horizon"])[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(path(cfg, "results_dir", "tables", "main_results.csv"), index=False)

    show = ["model", "features", "horizon", "test_RMSE_mean", "test_RMSE_std", "test_MAE_mean", "test_MAPE_mean"]
    print(agg[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
