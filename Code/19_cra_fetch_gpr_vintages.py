"""
19_cra_fetch_gpr_vintages.py  -  Archived weekly vintages of the daily GPR index (issue #10 follow-up)

Each vintage file (data_gpr_daily_recent_YYYYMMDD.xls) is read in memory and
only its last `keep_days` days are stored, enough to build a point-in-time
GPR series and to measure revisions against the current file.

Outputs
  data/cra/raw/gpr_vintages.csv     vintage, date, GPR

Usage
  python 19_cra_fetch_gpr_vintages.py
"""
from __future__ import annotations

import io
import re
import time

import pandas as pd
import requests

from oilfc.utils import CODE_DIR, load_config, path


def list_vintages(api_url: str) -> list[str]:
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    names = [x["name"] for x in r.json()]
    return sorted(n for n in names if re.fullmatch(r"data_gpr_daily_recent_\d{8}\.xls", n))


def read_vintage(url: str, keep_days: int) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), engine="xlrd").dropna(subset=["DAY"])
    dates = pd.to_datetime(df["DAY"].astype(int).astype(str), format="%Y%m%d")
    out = pd.DataFrame({"date": dates, "GPR": pd.to_numeric(df["GPRD"], errors="coerce")}).dropna()
    return out[out.date > out.date.max() - pd.Timedelta(days=keep_days)]


def main():
    cfg = load_config(CODE_DIR / "cra_config.yaml")
    g = load_config(CODE_DIR / "cra_followup.yaml")["gpr"]
    names = list_vintages(g["vintage_api"])
    print(f"{len(names)} vintages: {names[0]} .. {names[-1]}")
    rows, t0 = [], time.time()
    for k, n in enumerate(names):
        v = pd.Timestamp(re.search(r"(\d{8})", n).group(1))
        rows.append(read_vintage(f"{g['vintage_raw']}/{n}", g["keep_days"]).assign(vintage=v))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(names)}  ({time.time() - t0:.0f} s)")
    out = pd.concat(rows, ignore_index=True)[["vintage", "date", "GPR"]]
    f = path(cfg, "raw_dir", "gpr_vintages.csv")
    out.to_csv(f, index=False)
    print(f"Saved {len(out)} rows from {len(names)} vintages to {f}")


if __name__ == "__main__":
    main()
