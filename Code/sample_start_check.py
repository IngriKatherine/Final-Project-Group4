"""
sample_start_check.py  -  Does dropping USD_BROAD (and so starting the sample earlier)
change forecast accuracy?  (owner: data pipeline & baselines, issue #1)

Two samples, identical to config.yaml in every other setting:
  sample2006_with_usd_broad      every series in config.yaml   -> starts 2006-01-02
  sample2003_without_usd_broad   USD_BROAD removed             -> starts 2003-01-02
DXY stays in both, so the dollar is always represented.

Each sample is trained with two input sets:
  WTI only     the oil price's own history; isolates the effect of extra years
  All series   every series in the sample; extra years AND the removed series together

The choice is made on VALIDATION metrics only (2019-2021). Test metrics are still
written to the per-run JSON files but are left out of the summaries on purpose, so
the test period does not influence which sample we use. config.yaml is never modified.

Outputs (all under Code/results/sample_start_check/)
  validation_summary.csv / .md    mean and std over seeds, one row per sample x inputs
  validation_by_run.csv           one row per training run
  <sample>/config_used.yaml       exact settings of that sample
  <sample>/processed/levels.csv   the data that sample was trained on
  <sample>/tables, predictions, models, figures   full outputs of every run

Usage (after 01_fetch_data.py)
  python sample_start_check.py
  python sample_start_check.py --models lstm gru --horizons 1 5 20
  python sample_start_check.py --force               # retrain even if results exist

Reproducing: same git commit + same data/raw/*.csv (see data/raw/download_log.json
for the download date; FRED and Yahoo data can be revised later) + same seeds.
CPU runs reproduce exactly; GPU runs can differ slightly between machines.
"""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path

import pandas as pd
import yaml

from oilfc.utils import CODE_DIR, import_script, load_config

prep_script = import_script("02_preprocess.py")
train_script = import_script("04_train_rnn.py")

REL = Path("results") / "sample_start_check"   # relative to Code/, like config.yaml paths
OUT = CODE_DIR / REL

# sample name -> series removed from config.yaml
SAMPLES = {
    "sample2006_with_usd_broad": [],
    "sample2003_without_usd_broad": ["USD_BROAD"],
}
INPUT_LABELS = {"price_only": "WTI only", "all": "All series"}
SUBDIRS = ["processed", "tables/runs", "figures", "predictions", "models", "shap"]
# MAPE is left out of the summary: the clipped April 2020 price (1 USD/bbl) sits in
# the validation period and dominates it. It is still in validation_by_run.csv.
SUMMARY_METRICS = ["val_RMSE", "val_MAE", "val_DirAcc"]


def sample_cfg(base: dict, name: str, drop: list[str]) -> dict:
    """Copy of config.yaml with some series removed and its own output folder."""
    cfg = copy.deepcopy(base)
    for col in drop:
        cfg["data"]["fred_series"].pop(col, None)
        cfg["data"]["yahoo_tickers"].pop(col, None)

    cfg["paths"]["processed_dir"] = (REL / name / "processed").as_posix()
    cfg["paths"]["results_dir"] = (REL / name).as_posix()

    # oilfc.utils.path() only creates the top folder, so create the subfolders
    # that 04_train_rnn.py writes into (predictions/, models/, ...).
    root = OUT / name
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    with open(root / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg


def main():
    base = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm"])
    ap.add_argument("--features", nargs="+", default=["price_only", "all"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1])
    ap.add_argument("--seeds", nargs="+", type=int, default=base["experiments"]["seeds"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = []
    for name, drop in SAMPLES.items():
        print(f"\n######## {name} (removing {drop or 'nothing'}) ########")
        cfg = sample_cfg(base, name, drop)
        levels = prep_script.main(cfg)
        info = {"sample": name,
                "sample_start": str(levels.index[0].date()),
                "sample_end": str(levels.index[-1].date()),
                "n_rows": len(levels)}
        print(info)

        grid = itertools.product(args.models, args.features, args.horizons, args.seeds)
        for m, fs, h, s in grid:
            res = train_script.run_single(cfg, m, fs, h, s, force=args.force, verbose=False)
            label = INPUT_LABELS.get(fs, fs)
            print(f"   {m.upper()} {label:10} h={h} seed={s}: val RMSE {res['val']['RMSE']:.4f}")
            rows.append({**info,
                         "inputs": label,
                         "n_inputs": len(res["feature_names"]),
                         "model": m.upper(),
                         "horizon": h,
                         "seed": s,
                         "epochs": res["history"]["epochs_run"],
                         **{f"val_{k}": round(v, 4) for k, v in res["val"].items()}})

    by_run = pd.DataFrame(rows)
    by_run.to_csv(OUT / "validation_by_run.csv", index=False)

    keys = ["sample", "sample_start", "n_rows", "inputs", "n_inputs", "model", "horizon"]
    summary = by_run.groupby(keys, sort=False)[SUMMARY_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().round(4)
    summary.to_csv(OUT / "validation_summary.csv", index=False)
    with open(OUT / "validation_summary.md", "w") as f:
        f.write("Validation period 2019-2021, errors in USD/bbl, mean and std over seeds.\n\n")
        f.write(summary.to_markdown(index=False))

    print(f"\nValidation results (mean and std over seeds), saved to {OUT}:")
    show = ["sample", "inputs", "n_inputs", "n_rows", "val_RMSE_mean", "val_RMSE_std"]
    print(summary[show].to_string(index=False))


if __name__ == "__main__":
    main()