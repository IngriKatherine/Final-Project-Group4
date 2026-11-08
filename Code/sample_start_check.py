"""
sample_start_check.py  -  Does dropping USD_BROAD (and so starting the sample earlier)
change forecast accuracy?  (owner: data pipeline & baselines, issue #1)

Two variants, identical to config.yaml in every other setting:
  with_usd_broad      every series in config.yaml      -> sample starts 2006
  without_usd_broad   USD_BROAD removed                -> sample starts 2003
DXY stays in both, so the dollar is always represented.

For each variant the script preprocesses the SAME raw files into its own folder and
trains the networks for every (model, feature set, horizon, seed). Feature set
price_only isolates the effect of the longer sample; "all" combines it with the
removed feature.

The choice is made on VALIDATION metrics only. Test metrics are still written to the
per-run JSON files but are left out of the summary on purpose, so the test period
does not influence which sample we use.

config.yaml is never modified.

Outputs
  results_sample_check/<variant>/config_used.yaml   exact settings of the variant
  results_sample_check/<variant>/processed/         levels.csv
  results_sample_check/<variant>/results/           runs, predictions, models, tables
  results_sample_check/summary_raw.csv              one row per run (validation only)
  results_sample_check/summary.csv                  mean and std over seeds

Usage (after 01_fetch_data.py; paths resolve against Code/ wherever you launch it)
  python sample_start_check.py
  python sample_start_check.py --models lstm gru --horizons 1 5 20
  python sample_start_check.py --force               # retrain even if results exist

Reproducing: same git commit + same data/raw/*.csv (see data/raw/download_log.json
for the download date; FRED and Yahoo data can be revised later) + same seeds.
GPU runs can differ slightly between machines; CPU runs are closer to exact.
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

REL = Path("results_sample_check")   # relative to Code/, as config.yaml paths are
OUT = CODE_DIR / REL
VARIANTS = {
    "with_usd_broad": [],
    "without_usd_broad": ["USD_BROAD"],
}
RESULT_SUBDIRS = ["tables/runs", "figures", "predictions", "models", "shap"]


def variant_cfg(base: dict, name: str, drop: list[str]) -> dict:
    """Copy of config.yaml with some series removed and its own output folders."""
    cfg = copy.deepcopy(base)
    for col in drop:
        cfg["data"]["fred_series"].pop(col, None)
        cfg["data"]["yahoo_tickers"].pop(col, None)

    cfg["paths"]["processed_dir"] = (REL / name / "processed").as_posix()
    cfg["paths"]["results_dir"] = (REL / name / "results").as_posix()

    # oilfc.utils.path() only creates the top folder, so create the subfolders
    # that 04_train_rnn.py writes into (predictions/, models/, ...).
    root = OUT / name
    (root / "processed").mkdir(parents=True, exist_ok=True)
    for sub in RESULT_SUBDIRS:
        (root / "results" / sub).mkdir(parents=True, exist_ok=True)

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
    for name, drop in VARIANTS.items():
        print(f"\n######## variant: {name} (dropping {drop or 'nothing'}) ########")
        cfg = variant_cfg(base, name, drop)
        levels = prep_script.main(cfg)
        sample = {"variant": name,
                  "start": str(levels.index[0].date()),
                  "end": str(levels.index[-1].date()),
                  "n_rows": len(levels),
                  "n_series": levels.shape[1]}
        print(sample)

        grid = itertools.product(args.models, args.features, args.horizons, args.seeds)
        for m, fs, h, s in grid:
            res = train_script.run_single(cfg, m, fs, h, s, force=args.force, verbose=False)
            print(f"   {m} {fs:11} h={h} seed={s}: val RMSE {res['val']['RMSE']:.3f}")
            rows.append({**sample, "model": m, "features": fs, "horizon": h, "seed": s,
                         "epochs": res["history"]["epochs_run"],
                         **{f"val_{k}": v for k, v in res["val"].items()}})

    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "summary_raw.csv", index=False)

    keys = ["variant", "start", "n_rows", "model", "features", "horizon"]
    metrics = [c for c in raw.columns if c.startswith("val_")]
    agg = raw.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT / "summary.csv", index=False)

    print("\nValidation results (mean and std over seeds):")
    show = keys + [c for c in agg.columns if c.startswith("val_RMSE")]
    print(agg[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
