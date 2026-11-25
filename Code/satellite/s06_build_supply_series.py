"""s06 - Turn ship detections into a "supply of oil through Hormuz" time series.

Per image date and zone:
    n_large    ships >= 150 m (oil-tanker size: Aframax, Suezmax, VLCC, also LNG/containers)
    n_medium   60-150 m (product tankers, feeders)
    dwt_large  sum of estimated deadweight of the large ships (tonnes), from hull length
Cloud correction: each zone's count is divided by the share of its usual water that was
visible (visible water px / the zone's 90th-percentile visible water). Zones with less
than `supply.min_coverage` visible are set to missing for that date.

Series written to data/series/
    supply_by_scene.csv    one row per date (every 2-5 days)
        transit_n_large, transit_dwt, transit_mbbl  -> oil-size ships MOVING through the strait
        anchor_n_large, anchor_mbbl                 -> oil-size ships WAITING at anchor
        coverage_transit, coverage_anchor
    supply_weekly.csv      weekly (Friday) means, the frequency the price model uses
and figures/supply_series.png

transit_mbbl is a SNAPSHOT (million barrels of tanker capacity inside the strait at image
time). s07 converts it into a flow (million barrels/day) by calibrating against AIS
transit data before 2026.

Run
    python s06_build_supply_series.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hormuz.common import P, get_logger, load_cfg

log = get_logger("s06")


def build(cfg) -> pd.DataFrame:
    s = cfg["supply"]
    det = pd.read_csv(P(cfg, "detections_csv")) if P(cfg, "detections_csv").exists() else pd.DataFrame()
    cov = pd.read_csv(P(cfg, "coverage_csv"))
    # visible share relative to what is normal for the zone (land inside polygons cancels out)
    ref = cov.groupby("zone")["water_px"].quantile(0.9).rename("ref_px")
    cov = cov.join(ref, on="zone")
    cov["coverage"] = (cov["water_px"] / cov["ref_px"]).clip(0, 1)

    if det.empty:
        det = pd.DataFrame(columns=["date", "zone", "size_class", "dwt_est"])
    det = det[det["zone"] != "other"]
    agg = det.assign(large=det["size_class"].eq("large"), medium=det["size_class"].eq("medium")) \
        .groupby(["date", "zone"]).agg(n_large=("large", "sum"), n_medium=("medium", "sum"),
                                       dwt_large=("dwt_est", "sum")).reset_index()
    z = cov.merge(agg, on=["date", "zone"], how="left").fillna({"n_large": 0, "n_medium": 0, "dwt_large": 0})
    ok = z["coverage"] >= s["min_coverage"]
    for c in ("n_large", "n_medium", "dwt_large"):
        z[c + "_adj"] = np.where(ok, z[c] / z["coverage"].clip(lower=1e-6), np.nan)

    out = []
    for date, g in z.groupby("date"):
        row = {"date": date}
        for typ, label in (("transit", "transit"), ("anchorage", "anchor")):
            gg = g[g["zone_type"] == typ]
            valid = gg["coverage"] >= s["min_coverage"]
            row[f"coverage_{label}"] = float(gg["coverage"].mean()) if len(gg) else np.nan
            if len(gg) == 0 or not valid.all():          # a hidden zone would bias the total down
                row.update({f"{label}_n_large": np.nan, f"{label}_n_medium": np.nan, f"{label}_dwt": np.nan})
                continue
            row[f"{label}_n_large"] = gg["n_large_adj"].sum()
            row[f"{label}_n_medium"] = gg["n_medium_adj"].sum()
            row[f"{label}_dwt"] = gg["dwt_large_adj"].sum()
        for zname, zz in g.groupby("zone"):                 # per-zone detail (routes shift in 2026)
            row[f"{zname}_n_large"] = float(zz["n_large_adj"].iloc[0])
        out.append(row)
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date"])
    bpt = s["barrels_per_tonne"]
    for label in ("transit", "anchor"):
        if f"{label}_dwt" in df:
            df[f"{label}_mbbl"] = df[f"{label}_dwt"] * bpt / 1e6
    return df.sort_values("date").reset_index(drop=True)


def plot(df: pd.DataFrame, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for a, col, title in ((ax[0], "transit_mbbl", "Tanker-size capacity moving through the strait (snapshot, million bbl)"),
                          (ax[1], "anchor_mbbl", "Tanker-size capacity waiting at Larak/Qeshm anchorage (million bbl)")):
        if col not in df:
            continue
        a.plot(df["date"], df[col], ".", ms=4, alpha=0.5, label="each image")
        w = df.set_index("date")[col].resample("W-FRI").mean()
        a.plot(w.index, w.values, "-", lw=1.8, label="weekly mean")
        a.axvline(pd.Timestamp("2026-02-28"), color="red", ls="--", lw=1)
        a.set_title(title, fontsize=10)
        a.legend(fontsize=8)
    ax[0].text(pd.Timestamp("2026-03-05"), ax[0].get_ylim()[1] * 0.9, "war starts 28 Feb 2026", color="red", fontsize=8)
    ax[1].set_xlabel("Sentinel-2 image date")
    fig.tight_layout()
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, dpi=130)
    plt.close(fig)


def main():
    cfg = load_cfg()
    df = build(cfg)
    out = P(cfg, "series_dir")
    df.to_csv(out / "supply_by_scene.csv", index=False)
    num = df.set_index("date").select_dtypes("number")
    weekly = num.resample("W-FRI").mean()
    weekly["n_images"] = num["transit_n_large"].resample("W-FRI").count()
    weekly.to_csv(out / "supply_weekly.csv")
    plot(df, P(cfg, "figures_dir") / "supply_series.png")
    log.info("%d image dates (%d with valid transit zones) -> %s", len(df), df["transit_n_large"].notna().sum(), out)
    pre = df[df["date"] < "2026-02-28"]["transit_n_large"].mean()
    post = df[df["date"] >= "2026-02-28"]["transit_n_large"].mean()
    log.info("mean large ships in transit zones: before 28 Feb 2026 = %.1f, after = %.1f", pre, post)


if __name__ == "__main__":
    main()
