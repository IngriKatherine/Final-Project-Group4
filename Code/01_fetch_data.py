"""
01_fetch_data.py  -  Download raw data (owner: data pipeline & baselines)

Downloads oil prices and candidate drivers listed in config.yaml:
  * FRED series via the public CSV endpoint (no API key required)
  * Yahoo Finance tickers via yfinance

Outputs
  data/raw/fred.csv         one column per FRED series (renamed to config keys)
  data/raw/yahoo.csv        one column per Yahoo ticker (adjusted close)
  data/raw/ohlcv_{name}.csv Open/High/Low/Close/Volume for data.ohlcv_tickers
                            (input to the image CNN, 09_image_cnn.py)
  data/raw/download_log.json  what was downloaded, when, and date coverage

Usage
  python 01_fetch_data.py
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import requests

from oilfc.utils import load_config, path, save_json

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def fetch_fred(series: dict[str, str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for name, sid in series.items():
        print(f"FRED   {sid:<14} -> {name}")
        r = requests.get(FRED_CSV.format(sid=sid), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["Date", name]
        df["Date"] = pd.to_datetime(df["Date"])
        df[name] = pd.to_numeric(df[name], errors="coerce")  # FRED uses "." for missing
        frames.append(df.set_index("Date"))
    out = pd.concat(frames, axis=1).sort_index()
    return out.loc[start:end]


def fetch_yahoo(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    frames = []
    for name, tk in tickers.items():
        print(f"Yahoo  {tk:<14} -> {name}")
        df = yf.download(tk, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            print(f"   WARNING: no data for {tk}")
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):      # newer yfinance returns a MultiIndex
            close = close.iloc[:, 0]
        frames.append(close.rename(name))
    out = pd.concat(frames, axis=1).sort_index()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.index.name = "Date"
    return out


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily Open/High/Low/Close/Volume. Futures need no adjustment, so auto_adjust=False
    keeps the raw exchange prices (CL=F closes at -37.63 on 2020-04-20)."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):      # newer yfinance
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df.sort_index()


def coverage(df: pd.DataFrame) -> dict:
    return {c: {"first": str(df[c].first_valid_index().date()),
                "last": str(df[c].last_valid_index().date()),
                "n_obs": int(df[c].notna().sum())} for c in df.columns}


def main():
    cfg = load_config()
    d = cfg["data"]
    start, end = d["start_date"], d["end_date"]

    fred = fetch_fred(d["fred_series"], start, end)
    fred.to_csv(path(cfg, "raw_dir", "fred.csv"))

    yahoo = fetch_yahoo(d["yahoo_tickers"], start, end)
    yahoo.to_csv(path(cfg, "raw_dir", "yahoo.csv"))

    log = {
        "downloaded_at": dt.datetime.now().isoformat(timespec="seconds"),
        "start": start, "end": end,
        "fred": coverage(fred), "yahoo": coverage(yahoo), "ohlcv": {},
    }

    # Full OHLCV for the image CNN. Kept out of yahoo.csv so 02_preprocess.py is unaffected.
    for name, tk in d.get("ohlcv_tickers", {}).items():
        print(f"OHLCV  {tk:<14} -> ohlcv_{name}.csv")
        ohlcv = fetch_ohlcv(tk, start, end)
        ohlcv.to_csv(path(cfg, "raw_dir", f"ohlcv_{name}.csv"))
        log["ohlcv"][name] = {"ticker": tk, **coverage(ohlcv[["Close"]])["Close"],
                              "zero_volume_days": int((ohlcv["Volume"] <= 0).sum())}
    save_json(log, path(cfg, "raw_dir", "download_log.json"))
    print(f"\nSaved {fred.shape} FRED and {yahoo.shape} Yahoo rows to {path(cfg, 'raw_dir')}")


if __name__ == "__main__":
    main()
