"""s07 - Calibrate the satellite series against AIS transits and export price-model features.

1. Download IMF PortWatch daily transits for the Strait of Hormuz (AIS-based: tankers
   counted and their cargo tonnage estimated). Free ArcGIS REST API, no key.
2. Calibrate: satellite snapshot of tanker capacity in the strait (s06 transit_dwt, tonnes)
   -> flow (tonnes/day) with one constant k, fitted ONLY on dates up to
   portwatch.calibration_end (2025). Physically k = 24 h x speed / corridor length.
   Then million barrels/day = tonnes/day x 7.33 / 1e6.
3. Compare with AIS in 2026. AIS misses ships that switch transponders off ("dark"),
   so a satellite estimate ABOVE the AIS line in 2026 is the evidence that the images
   add information AIS doesn't have.
4. Export business-day features for the group's price model (causal: on day t only
   images taken on or before t are used):
       HZ_SAT_FLOW_MBD     satellite oil flow through Hormuz, mean of images in the last 10 days
       HZ_SAT_ANCHOR_MBBL  tanker capacity waiting at anchor (stuck supply)
       HZ_SAT_LARGE_N      large ships in the transit zones
       HZ_AIS_FLOW_MBD     PortWatch AIS tanker flow, 7-day mean (baseline feature)
       HZ_OBS_AGE_D        days since the last usable image
   -> ../data/processed/satellite_features.csv  (+ data/series/features_weekly.csv)

Run
    python s07_calibrate_and_export.py
    python s07_calibrate_and_export.py --ais-lag 3    # PortWatch publishes with a few days' delay
    python s07_calibrate_and_export.py --portwatch-csv data/series/portwatch_hormuz_daily.csv   # offline re-run
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import requests

from hormuz.common import P, get_logger, load_cfg, resolve, save_json

log = get_logger("s07")
CONFLICT = pd.Timestamp("2026-02-28")


def fetch_portwatch(cfg) -> pd.DataFrame:
    pw = cfg["portwatch"]
    rows, offset = [], 0
    while True:
        r = requests.get(pw["url"], params={
            "where": f"portname LIKE '%{pw['name_contains']}%'", "outFields": "*", "f": "json",
            "orderByFields": "date", "resultOffset": offset, "resultRecordCount": 1000}, timeout=60)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(js["error"])
        feats = js.get("features", [])
        rows += [f["attributes"] for f in feats]
        if len(feats) < 1000 and not js.get("exceededTransferLimit"):
            break
        offset += len(feats)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("PortWatch returned nothing - check portwatch.name_contains")
    d = df["date"]
    df["date"] = pd.to_datetime(d, unit="ms", utc=True) if pd.api.types.is_numeric_dtype(d) \
        else pd.to_datetime(d, utc=True, format="mixed")
    df["date"] = df["date"].dt.tz_localize(None).dt.normalize()
    for c in ("n_tanker", "capacity_tanker", "n_total"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log.info("PortWatch: %d days for %s, %s .. %s", len(df), df["portname"].iloc[0],
             df["date"].min().date(), df["date"].max().date())
    return df.sort_values("date").drop_duplicates("date")


def fit_k(x: np.ndarray, y: np.ndarray, fallback: float, min_weeks: int = 8) -> dict:
    """Least squares through the origin y = k x, plus diagnostics.
    With fewer than `min_weeks` usable weeks, fall back to the physical constant."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if m.sum() < min_weeks or (x * x).sum() == 0:
        log.warning("only %d calibration weeks - using the physical constant k = %.1f/day "
                    "(supply.fallback_k_per_day); download more 2023-2025 scenes", m.sum(), fallback)
        return dict(k=float(fallback), r2=np.nan, pearson=np.nan, n=int(m.sum()), fallback=True)
    k = float((x * y).sum() / (x * x).sum())
    pred = k * x
    ss_res, ss_tot = ((y - pred) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return dict(k=k, r2=float(1 - ss_res / ss_tot), pearson=float(np.corrcoef(x, y)[0, 1]),
                n=int(m.sum()), fallback=False)


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ais-lag", type=int, default=0, help="days PortWatch data arrive late (for honest backtests)")
    ap.add_argument("--portwatch-csv", default=None,
                    help="use a saved PortWatch table (date, n_tanker, capacity_tanker, n_total) instead of the API")
    args = ap.parse_args()
    bpt = cfg["supply"]["barrels_per_tonne"]
    series = P(cfg, "series_dir")

    sat = pd.read_csv(series / "supply_by_scene.csv", parse_dates=["date"])
    if args.portwatch_csv:
        pw = pd.read_csv(args.portwatch_csv)
        pw["date"] = pd.to_datetime(pw["date"]).dt.normalize()
        log.info("PortWatch from %s: %d days", args.portwatch_csv, len(pw))
    else:
        pw = fetch_portwatch(cfg)
        pw.to_csv(series / "portwatch_hormuz_daily.csv", index=False)
    pwd = pw.set_index("date")[["n_tanker", "capacity_tanker", "n_total"]].asfreq("D")

    # --- calibration on weekly means (both series are noisy day to day) ---------------
    w = pd.DataFrame({"sat_dwt": sat.set_index("date")["transit_dwt"].resample("W-FRI").mean(),
                      "ais_t": pwd["capacity_tanker"].resample("W-FRI").mean(),
                      "ais_n": pwd["n_tanker"].resample("W-FRI").mean()}).dropna(subset=["sat_dwt"])
    cal_end = pd.Timestamp(cfg["portwatch"]["calibration_end"])
    cal = w[w.index <= cal_end]
    fit = fit_k(cal["sat_dwt"].values, cal["ais_t"].values, cfg["supply"]["fallback_k_per_day"])
    log.info("calibration (weeks <= %s): k = %.2f /day, R2 = %.2f, r = %.2f, n = %d weeks",
             cal_end.date(), fit["k"], fit["r2"], fit["pearson"], fit["n"])
    post = w[w.index >= CONFLICT]
    if len(post) > 3:
        fit["r_2026"] = float(post[["sat_dwt", "ais_t"]].corr().iloc[0, 1])
    w["sat_flow_mbd"] = w["sat_dwt"] * fit["k"] * bpt / 1e6
    w["ais_flow_mbd"] = w["ais_t"] * bpt / 1e6
    w["gap_mbd"] = w["sat_flow_mbd"] - w["ais_flow_mbd"]
    for period, mask in (("2023-2025", w.index < CONFLICT), ("since 28 Feb 2026", w.index >= CONFLICT)):
        sub = w[mask]
        if len(sub):
            log.info("%-18s satellite %.2f mb/d   AIS %.2f mb/d   gap %.2f mb/d", period,
                     sub["sat_flow_mbd"].mean(), sub["ais_flow_mbd"].mean(), sub["gap_mbd"].mean())
    save_json(fit, series / "calibration.json")
    w.to_csv(series / "satellite_vs_ais_weekly.csv")

    # --- causal business-day features --------------------------------------------------
    s = sat.set_index("date").sort_index()
    s["flow_mbd"] = s["transit_dwt"] * fit["k"] * bpt / 1e6
    daily = pd.DataFrame(index=pd.date_range(s.index.min(), max(s.index.max(), pwd.index.max()), freq="D"))
    valid = s["flow_mbd"].dropna()
    daily["HZ_SAT_FLOW_MBD"] = valid.reindex(daily.index).rolling("10D", min_periods=1).mean()
    daily["HZ_SAT_ANCHOR_MBBL"] = s["anchor_mbbl"].dropna().reindex(daily.index).rolling("10D", min_periods=1).mean()
    daily["HZ_SAT_LARGE_N"] = s["transit_n_large"].dropna().reindex(daily.index).rolling("10D", min_periods=1).mean()
    last_obs = pd.Series(valid.index, index=valid.index).reindex(daily.index).ffill()
    daily["HZ_OBS_AGE_D"] = (daily.index - last_obs).dt.days
    daily[["HZ_SAT_FLOW_MBD", "HZ_SAT_ANCHOR_MBBL", "HZ_SAT_LARGE_N"]] = \
        daily[["HZ_SAT_FLOW_MBD", "HZ_SAT_ANCHOR_MBBL", "HZ_SAT_LARGE_N"]].ffill()
    ais = (pwd["capacity_tanker"] * bpt / 1e6).rolling("7D", min_periods=3).mean()
    daily["HZ_AIS_FLOW_MBD"] = ais.shift(args.ais_lag).reindex(daily.index).ffill()

    # Business days over the whole satellite period (2026 included even if the group's
    # levels.csv still stops in 2025), with the group's prices joined where available.
    feats = daily[daily.index.dayofweek < 5].copy()
    levels_file = resolve(cfg["export"]["levels_csv"])
    if levels_file.exists():
        lv = pd.read_csv(levels_file, index_col=0, parse_dates=True)
        feats = feats.join(lv, how="left")
        log.info("joined prices from %s (it ends %s)", levels_file.name, lv.index.max().date())
        if lv.index.max() < feats.index.max() - pd.Timedelta(days=30):
            log.warning("levels.csv stops before the satellite series: set data.end_date in "
                        "Code/config.yaml to 2026-11-24 and rerun 01/02 to get 2026 prices")
    else:
        log.warning("%s not found - exporting satellite features only", levels_file)
    out = resolve(cfg["export"]["out_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.dropna(how="all").to_csv(out)
    feats.resample("W-FRI").last().to_csv(series / "features_weekly.csv")
    log.info("features -> %s", out)
    ais_weekly = (pwd["capacity_tanker"] * bpt / 1e6).resample("W-FRI").mean()
    plots(cfg, w, fit, levels_file, ais_weekly)


def plots(cfg, w: pd.DataFrame, fit: dict, levels_file, ais_weekly: pd.Series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = P(cfg, "figures_dir")
    pre, post = w[w.index < CONFLICT], w[w.index >= CONFLICT]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(pre["sat_dwt"] / 1e6, pre["ais_t"] / 1e6, s=12, label="2023-2025 (fit)")
    ax.scatter(post["sat_dwt"] / 1e6, post["ais_t"] / 1e6, s=12, c="red", label="since 28 Feb 2026")
    xx = np.linspace(0, w["sat_dwt"].max() / 1e6, 10)
    ax.plot(xx, fit["k"] * xx, "k--", lw=1,
            label=f"k = {fit['k']:.2f}/day" + ("" if fit.get("fallback") else f", R² = {fit['r2']:.2f}"))
    ax.set(xlabel="satellite: tanker-size DWT in strait at image time (Mt)",
           ylabel="AIS (PortWatch): tanker tonnage transiting per day (Mt)", title="Weekly means")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "calibration_scatter.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ais_weekly = ais_weekly[ais_weekly.index >= w.index.min() - pd.Timedelta(days=7)]
    ax.plot(ais_weekly.index, ais_weekly.values, lw=1.2, alpha=0.8, label="AIS transits (IMF PortWatch)")
    ax.plot(w.index, w["sat_flow_mbd"], "o-", ms=3, lw=1.8, label="Satellite CNN estimate (Sentinel-2)")
    ax.axvline(CONFLICT, color="red", ls="--", lw=1)
    ax.set(ylabel="million barrels / day (tanker cargo capacity)",
           title="Oil moving through the Strait of Hormuz, weekly")
    if levels_file.exists():
        lv = pd.read_csv(levels_file, index_col=0, parse_dates=True)
        col = "BRENT" if "BRENT" in lv else lv.columns[0]
        b = lv[col].resample("W-FRI").last()
        b = b[b.index >= w.index.min() - pd.Timedelta(days=7)]
        ax2 = ax.twinx()
        ax2.plot(b.index, b.values, color="black", lw=1, alpha=0.6, label=f"{col} (right)")
        ax2.set_ylabel(f"{col} USD/bbl")
        ax2.legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "satellite_vs_ais_flow.png", dpi=130)
    plt.close(fig)
    log.info("figures -> %s", figs)


if __name__ == "__main__":
    main()
