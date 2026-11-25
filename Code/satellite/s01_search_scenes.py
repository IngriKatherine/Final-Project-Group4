"""s01 - Find every Sentinel-2 L2A scene over the Strait of Hormuz.

Source: Element 84 Earth Search STAC API, which indexes the Sentinel-2 Cloud-Optimized
GeoTIFFs on the AWS Registry of Open Data. Free, no account, and fastest from an AWS
machine (the data sit in us-west-2).

Output
    data/stac/scenes.jsonl   full STAC items (s02 reads these)
    data/stac/scenes.csv     one row per item: id, datetime, cloud cover, tile, baseline

Run
    python s01_search_scenes.py                       # dates from config
    python s01_search_scenes.py --start 2025-06-01 --end 2026-11-24
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from hormuz.common import P, get_logger, load_cfg

log = get_logger("s01")


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=cfg["dates"]["start"])
    ap.add_argument("--end", default=cfg["dates"]["end"])
    ap.add_argument("--max-cloud", type=float, default=cfg["stac"]["max_cloud"])
    args = ap.parse_args()

    from pystac_client import Client

    client = Client.open(cfg["stac"]["url"])
    search = client.search(collections=[cfg["stac"]["collection"]], bbox=cfg["aoi"]["bbox"],
                           datetime=f"{args.start}/{args.end}",
                           query={"eo:cloud_cover": {"lt": args.max_cloud}})
    items = list(search.items())
    log.info("found %d items %s .. %s (cloud < %s%%)", len(items), args.start, args.end, args.max_cloud)
    if not items:
        log.warning("nothing found - check the STAC url/collection in config_satellite.yaml")
        return

    from hormuz.raster import BANDS, asset_key
    try:
        found = {b: asset_key(items[0], b, cfg) for b in BANDS + ["scl"]}
        log.info("band -> asset: %s", found)
    except KeyError as e:
        log.error("%s", e)
        log.error("add the right names under stac.assets in config_satellite.yaml")
        return

    with open(P(cfg, "scenes_jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict()) + "\n")
    rows = [dict(id=it.id, datetime=it.datetime, date=it.datetime.date().isoformat(),
                 cloud=it.properties.get("eo:cloud_cover"),
                 tile=it.properties.get("s2:mgrs_tile") or it.properties.get("grid:code"),
                 platform=it.properties.get("platform"),
                 baseline=it.properties.get("s2:processing_baseline")) for it in items]
    df = pd.DataFrame(rows).sort_values("datetime")
    df.to_csv(P(cfg, "scenes_csv"), index=False)
    log.info("%d distinct dates, tiles: %s", df["date"].nunique(), sorted(df["tile"].dropna().unique()))
    log.info("dates per year:\n%s", df.groupby(df["date"].str[:4])["date"].nunique().to_string())
    latest = pd.Timestamp(df["date"].max())
    if latest < pd.Timestamp(args.end) - pd.Timedelta(days=30):
        log.warning("newest image is %s, more than 30 days before --end %s: collection %r may have "
                    "stopped updating - try the other one in config_satellite.yaml (stac.collection)",
                    latest.date(), args.end, cfg["stac"]["collection"])
    else:
        log.info("newest image: %s (collection %s)", latest.date(), cfg["stac"]["collection"])


if __name__ == "__main__":
    main()
