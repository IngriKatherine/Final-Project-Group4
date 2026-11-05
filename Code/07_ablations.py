"""
07_ablations.py  -  Architecture and window ablations (owner: LSTM/GRU modeling)

One-factor-at-a-time: start from the defaults in config.yaml and vary a single
setting (hidden_size, num_layers, lookback, dropout) while keeping everything
else fixed. Model selection uses VALIDATION metrics; test metrics are reported
but must not be used to pick settings. Forecast-horizon effects come from the
main grid (06_run_experiments.py).

Outputs
  results/tables/ablations_raw.csv
  results/tables/ablations.csv          mean/std over seeds
  results/figures/ablation_{param}.png

Usage
  python 07_ablations.py --features shap_selected --horizon 5
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oilfc.utils import import_script, load_config, override, path

train_script = import_script("04_train_rnn.py")

PARAM_KEYS = {"hidden_size": "model.hidden_size", "num_layers": "model.num_layers",
              "dropout": "model.dropout", "lookback": "framing.lookback"}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=cfg["experiments"]["models"])
    ap.add_argument("--features", default="all")
    ap.add_argument("--horizon", type=int, default=cfg["framing"]["horizons"][0])
    ap.add_argument("--seeds", nargs="+", type=int, default=cfg["experiments"]["seeds"])
    args = ap.parse_args()

    rows = []
    for param, values in cfg["ablations"].items():
        for v in values:
            run_cfg = override(cfg, **{PARAM_KEYS[param]: v})
            for m in args.models:
                for s in args.seeds:
                    res = train_script.run_single(run_cfg, m, args.features, args.horizon, s,
                                                  tag=f"abl-{param}{v}", verbose=False)
                    rows.append({"param": param, "value": v, "model": m, "seed": s,
                                 "n_params": res["n_params"],
                                 **{f"val_{k}": x for k, x in res["val"].items()},
                                 **{f"test_{k}": x for k, x in res["test"].items()}})
                    print(f"{param}={v} {m} s{s}: val RMSE {res['val']['RMSE']:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv(path(cfg, "results_dir", "tables", "ablations_raw.csv"), index=False)
    agg = raw.groupby(["param", "value", "model"]).agg(
        val_RMSE=("val_RMSE", "mean"), val_RMSE_std=("val_RMSE", "std"),
        test_RMSE=("test_RMSE", "mean"), n_params=("n_params", "first")).reset_index()
    agg.to_csv(path(cfg, "results_dir", "tables", "ablations.csv"), index=False)

    for param, sub in agg.groupby("param"):
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for m, g in sub.groupby("model"):
            ax.errorbar(g["value"], g["val_RMSE"], yerr=g["val_RMSE_std"], marker="o", capsize=3, label=m.upper())
        ax.set_xlabel(param); ax.set_ylabel("validation RMSE (USD/bbl)")
        ax.set_title(f"Ablation: {param} (features={args.features}, h={args.horizon})")
        ax.legend(); fig.tight_layout()
        fig.savefig(path(cfg, "results_dir", "figures", f"ablation_{param}.png"), dpi=150)
        plt.close(fig)
    print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
