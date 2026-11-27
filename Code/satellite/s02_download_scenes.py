"""s02 - Download one cloud-masked image of the Strait per date.

For every date found by s01, read only the AOI window of each tile (bands B04, B03, B02,
B08 at 10 m and the SCL cloud classification), reproject onto one fixed UTM grid,
mosaic the tiles of that date, and save
    data/scenes/S2_YYYYMMDD.tif   (5 bands uint8: red, green, blue, nir, scl)
plus a row in data/scenes/scene_index.csv with how much clear water was visible.

Each file is ~10,000 x 9,500 px; deflate keeps most dates at 40-120 MB. Each worker
needs ~1.5 GB RAM. 2023-2026 is ~250 dates. To start small, pass --start 2025-10-01.

Resumable: dates that already have a file are skipped.

Run
    python s02_download_scenes.py --workers 4
    python s02_download_scenes.py --start 2026-01-01 --max-dates 10      # quick test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from hormuz.common import P, aoi_grid, get_logger, load_cfg
from hormuz.raster import mosaic_date, valid_water, write_scene

log = get_logger("s02")


def load_items(cfg):
    import pystac

    by_date = defaultdict(list)
    with open(P(cfg, "scenes_jsonl")) as f:
        for line in f:
            it = pystac.Item.from_dict(json.loads(line))
            by_date[it.datetime.date().isoformat()].append(it)
    return dict(sorted(by_date.items()))


def process_date(date: str, items: list, cfg: dict, grid) -> dict:
    out = P(cfg, "scene_dir") / f"S2_{date.replace('-', '')}.tif"
    rgbn, scl = mosaic_date(items, cfg, grid)
    has_data = rgbn[0] > 0
    water = valid_water(rgbn, scl, cfg["inference"]["ndwi_min"])
    cloud = np.isin(scl, [3, 8, 9, 10]) & has_data
    row = dict(date=date, file=out.name, n_tiles=len(items),
               data_frac=float(has_data.mean()),
               cloud_frac=float(cloud.sum() / max(has_data.sum(), 1)),
               clear_water_px=int(water.sum()),
               items=";".join(i.id for i in items))
    if has_data.mean() < 0.05:
        row["status"] = "empty"
        return row
    write_scene(out, rgbn, scl, grid, tags=dict(date=date, items=row["items"],
                                                 scale_max_reflectance=cfg["image"]["scale_max_reflectance"]))
    row["status"] = "ok"
    return row


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=cfg["image"]["workers"])
    ap.add_argument("--max-dates", type=int, default=None)
    args = ap.parse_args()

    grid = aoi_grid(cfg)
    log.info("AOI grid %d x %d px at %s m (%s)", grid.width, grid.height, cfg["aoi"]["resolution"], grid.crs)
    by_date = load_items(cfg)
    dates = [d for d in by_date if (args.start is None or d >= args.start) and (args.end is None or d <= args.end)]
    index_file = P(cfg, "scene_index")
    done = set(pd.read_csv(index_file)["date"]) if index_file.exists() else set()
    todo = [d for d in dates if d not in done]
    if args.max_dates:
        todo = todo[: args.max_dates]
    log.info("%d dates in range, %d already done, %d to download", len(dates), len(dates) - len(todo), len(todo))

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(process_date, d, by_date[d], cfg, grid): d for d in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            d = futs[fut]
            try:
                row = fut.result()
            except Exception as e:          # network hiccups: log and continue, rerun later
                log.error("%s failed: %s", d, e)
                continue
            rows.append(row)
            log.info("[%d/%d] %s %s cloud=%.0f%% tiles=%d", k, len(todo), d, row["status"],
                     100 * row["cloud_frac"], row["n_tiles"])
            if len(rows) % 10 == 0:
                _append(index_file, rows)
                rows = []
    _append(index_file, rows)
    log.info("index: %s", index_file)


def _append(file, rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    if file.exists():
        df = pd.concat([pd.read_csv(file), df]).drop_duplicates("date", keep="last")
    df.sort_values("date").to_csv(file, index=False)


if __name__ == "__main__":
    main()
