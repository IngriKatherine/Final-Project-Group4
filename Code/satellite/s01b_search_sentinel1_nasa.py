"""s01b (optional) - Sentinel-1 radar scenes of the Strait from NASA's ASF DAAC.

NASA's Alaska Satellite Facility (ASF DAAC) distributes the full ESA Sentinel-1 archive.
Radar sees through cloud and at night, so it complements Sentinel-2. It needs a free
NASA Earthdata Login (https://urs.earthdata.nasa.gov) exported as environment variables:
    EARTHDATA_USERNAME, EARTHDATA_PASSWORD

Modes
    search  list GRD scenes over the AOI -> data/sentinel1/s1_scenes.csv
    rtc     submit them to NASA HyP3 for free radiometric terrain correction (GeoTIFF,
            10/20/30 m), wait, download, unzip into data/sentinel1/rtc/

The Sentinel-2 detector in this folder does NOT work on radar; these files are for a
radar detector (e.g. trained on xView3) or for checking cloudy dates by eye.

Run
    pip install asf_search hyp3_sdk
    python s01b_search_sentinel1_nasa.py search --start 2026-01-01
    python s01b_search_sentinel1_nasa.py rtc --limit 20 --resolution 20
"""
from __future__ import annotations

import argparse
import os
import zipfile

import pandas as pd

from hormuz.common import P, get_logger, load_cfg

log = get_logger("s01b")


def aoi_wkt(cfg) -> str:
    lon0, lat0, lon1, lat1 = cfg["aoi"]["bbox"]
    return f"POLYGON(({lon0} {lat0},{lon1} {lat0},{lon1} {lat1},{lon0} {lat1},{lon0} {lat0}))"


def search(cfg, start, end) -> pd.DataFrame:
    import asf_search as asf

    res = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=aoi_wkt(cfg),
                         start=start, end=end, processingLevel=asf.PRODUCT_TYPE.GRD_HD,
                         beamMode=asf.BEAMMODE.IW)
    rows = [dict(granule=r.properties["sceneName"], start=r.properties["startTime"],
                 orbit=r.properties.get("pathNumber"), direction=r.properties.get("flightDirection"),
                 platform=r.properties.get("platform"), url=r.properties.get("url")) for r in res]
    df = pd.DataFrame(rows).sort_values("start") if rows else pd.DataFrame()
    out = P(cfg, "sentinel1_dir") / "s1_scenes.csv"
    df.to_csv(out, index=False)
    log.info("%d Sentinel-1 GRD scenes -> %s", len(df), out)
    return df


def rtc(cfg, limit: int, resolution: int):
    import hyp3_sdk

    df = pd.read_csv(P(cfg, "sentinel1_dir") / "s1_scenes.csv").tail(limit)
    hyp3 = hyp3_sdk.HyP3(username=os.environ["EARTHDATA_USERNAME"],
                         password=os.environ["EARTHDATA_PASSWORD"])
    jobs = hyp3_sdk.Batch()
    for g in df["granule"]:
        jobs += hyp3.submit_rtc_job(g, name="hormuz", resolution=resolution, scale="power",
                                    radiometry="gamma0", dem_matching=False)
    log.info("submitted %d RTC jobs; waiting (can take 30-60 min)", len(jobs))
    jobs = hyp3.watch(jobs)
    out = P(cfg, "sentinel1_dir") / "rtc"
    out.mkdir(parents=True, exist_ok=True)
    for z in jobs.download_files(out):
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
    log.info("downloaded to %s", out)


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["search", "rtc"])
    ap.add_argument("--start", default=cfg["dates"]["start"])
    ap.add_argument("--end", default=cfg["dates"]["end"])
    ap.add_argument("--limit", type=int, default=20, help="rtc: newest N scenes")
    ap.add_argument("--resolution", type=int, default=20, choices=[10, 20, 30])
    a = ap.parse_args()
    search(cfg, a.start, a.end) if a.mode == "search" else rtc(cfg, a.limit, a.resolution)


if __name__ == "__main__":
    main()
