"""
15_cra_fetch_data.py  -  Raw data for the Chronos-anchored gated residual LSTM (issue #10)

Downloads the series in cra_config.yaml up to the latest available date:
  * FRED (incl. OVXCLS and DHOILNYH) and Yahoo Finance, reusing 01_fetch_data.py
  * Daily geopolitical risk index (Caldara & Iacoviello)

Then aligns everything on the WTI trading calendar (oilfc/cra_data.build_levels).

Outputs
  data/cra/raw/fred.csv, yahoo.csv, gpr.csv, download_log.json
  data/cra/processed/levels.csv

Usage
  python 15_cra_fetch_data.py
  python 15_cra_fetch_data.py --skip-download    # rebuild levels.csv only
"""
from __future__ import annotations

import argparse
import datetime as dt
import io

import pandas as pd
import requests

from oilfc.cra_data import build_levels
from oilfc.utils import CODE_DIR, import_script, load_config, path, save_json

fetch = import_script("01_fetch_data.py")


def fetch_gpr(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), engine="xlrd")
    date_col = "date" if "date" in df.columns else "DAY"
    dates = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d" if date_col == "DAY" else None)
    out = pd.DataFrame({"GPR": pd.to_numeric(df["GPRD"], errors="coerce").values}, index=dates)
    out.index.name = "Date"
    return out.sort_index()


def download(cfg: dict) -> None:
    d = cfg["data"]
    # end = today: Yahoo may return an intraday print for today, but the WTI calendar ends earlier
    start, end = d["start_date"], dt.date.today().isoformat()

    fred = fetch.fetch_fred(d["fred_series"], start, end)
    yahoo = fetch.fetch_yahoo(d["yahoo_tickers"], start, end)
    print("GPR    daily index")
    gpr = fetch_gpr(d["gpr_url"])

    for name, df in (("fred", fred), ("yahoo", yahoo), ("gpr", gpr)):
        df.to_csv(path(cfg, "raw_dir", f"{name}.csv"))
    save_json({"downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
               "gpr_url": d["gpr_url"],
               "fred": fetch.coverage(fred), "yahoo": fetch.coverage(yahoo), "gpr": fetch.coverage(gpr)},
              path(cfg, "raw_dir", "download_log.json"))
    print(f"Saved FRED {fred.shape}, Yahoo {yahoo.shape}, GPR {gpr.shape} to {path(cfg, 'raw_dir')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    cfg = load_config(CODE_DIR / "cra_config.yaml")
    if not args.skip_download:
        download(cfg)
    lv = build_levels(cfg)
    lv.to_csv(path(cfg, "processed_dir", "levels.csv"))
    print(f"levels.csv: {len(lv)} trading days, {lv.index[0].date()} to {lv.index[-1].date()}, "
          f"{int(lv['interp'].sum())} interpolated WTI price(s) on "
          f"{[str(x.date()) for x in lv.index[lv['interp'] == 1]]}")


if __name__ == "__main__":
    main()
