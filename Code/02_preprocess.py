"""
02_preprocess.py  -  Clean and align the raw series (owner: data pipeline & baselines)

Steps
  1. Merge FRED + Yahoo and keep only the series listed in config.yaml
  2. Put everything on a common calendar (business days, or weekly if configured)
  3. Forward-fill short gaps (holidays / different exchange calendars), max 5 steps
  4. Apply the negative-price policy to WTI (2020-04-20 settled at -37.63 USD/bbl)
  5. Trim to the period where every series is available
  6. Save levels + descriptive statistics + correlation table for the report

Outputs
  data/processed/levels.csv            aligned price/rate levels (model input source)
  results/tables/descriptive_stats.csv
  results/tables/correlations.csv      correlation of daily log returns / differences
  results/figures/prices_overview.png

Usage
  python 02_preprocess.py

Other scripts can call main(cfg) with a modified config (see sample_start_check.py).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from oilfc.data import make_features
from oilfc.utils import load_config, path


def main(cfg: dict | None = None):
    cfg = cfg or load_config()
    d = cfg["data"]

    fred = pd.read_csv(path(cfg, "raw_dir", "fred.csv"), index_col=0, parse_dates=True)
    yahoo = pd.read_csv(path(cfg, "raw_dir", "yahoo.csv"), index_col=0, parse_dates=True)
    df = pd.concat([fred, yahoo], axis=1, sort=True)

    # 1. config.yaml decides the feature pool. Raw files may contain more series
    #    (e.g. after a series is removed from the config), so select explicitly.
    wanted = list(d["fred_series"]) + list(d["yahoo_tickers"])
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise ValueError(f"In config.yaml but not in {path(cfg, 'raw_dir')}: {missing}. "
                         "Rerun 01_fetch_data.py.")
    dropped = [c for c in df.columns if c not in wanted]
    if dropped:
        print(f"Not in config.yaml, ignored: {dropped}")
    df = df[wanted]

    # 2-3. common calendar + short forward fill
    df = df.resample(d["frequency"]).last() if d["frequency"] != "B" else df.asfreq("B")
    df = df.ffill(limit=5)

    # 4. negative / near-zero oil prices
    for col in ("WTI", "BRENT"):
        if col not in df.columns:
            continue
        bad = df[col] < d["min_price"]
        if bad.any():
            print(f"{col}: {bad.sum()} value(s) below {d['min_price']} -> policy '{d['negative_price_policy']}'")
            if d["negative_price_policy"] == "clip":
                df.loc[bad, col] = d["min_price"]
            elif d["negative_price_policy"] == "drop":
                df = df.loc[~bad]

    # 5. keep only rows where all series exist
    print("First valid date per series:")
    print(df.apply(lambda s: s.first_valid_index()).sort_values().to_string())
    before = len(df)
    df = df.dropna()
    print(f"Dropped {before - len(df)} rows with missing values; {len(df)} rows "
          f"from {df.index[0].date()} to {df.index[-1].date()}")

    df.index.name = "Date"
    df.to_csv(path(cfg, "processed_dir", "levels.csv"))

    # 6. tables and a quick figure for the report
    df.describe().T.to_csv(path(cfg, "results_dir", "tables", "descriptive_stats.csv"))
    make_features(df, cfg).corr().round(3).to_csv(path(cfg, "results_dir", "tables", "correlations.csv"))

    fig, axes = plt.subplots(len(df.columns), 1, figsize=(10, 1.6 * len(df.columns)), sharex=True)
    for ax, col in zip(axes, df.columns):
        ax.plot(df.index, df[col], lw=0.8)
        ax.set_ylabel(col, rotation=0, ha="right", fontsize=8)
        for b in (cfg["split"]["train_end"], cfg["split"]["val_end"]):
            ax.axvline(pd.Timestamp(b), color="grey", ls="--", lw=0.6)
    axes[0].set_title("Series levels (dashed lines: train | val | test boundaries)")
    fig.tight_layout()
    fig.savefig(path(cfg, "results_dir", "figures", "prices_overview.png"), dpi=150)
    plt.close(fig)
    print("Saved levels.csv and summary tables/figures.")
    return df


if __name__ == "__main__":
    main()