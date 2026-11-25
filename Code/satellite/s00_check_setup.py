"""s00 - Check the machine and every external data source BEFORE the long runs.

Takes ~1 minute (plus ~170 MB if --weights). Prints PASS / WARN / FAIL per check:
  1. Python packages            2. GPU                       3. free disk
  4. config: AOI grid and zones (inside the AOI, valid polygons, area)
  5. Earth Search: newest Sentinel-2 image over the AOI, band names, and a real
     512 x 512 read of one band (proves anonymous COG reads work from this machine)
  6. IMF PortWatch API: the Hormuz chokepoint and its newest date
  7. Ai2 training data: the tar is reachable, and its size
  8. (--weights) download the COCO Faster R-CNN weights and run one forward pass

Run
    python s00_check_setup.py
    python s00_check_setup.py --weights
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import time
import traceback

import numpy as np

from hormuz.common import SAT_DIR, aoi_grid, load_cfg, zones_lonlat

AI2_TAR = "https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar"
results = []


def report(name, status, msg):
    results.append((name, status, msg))
    print(f"[{status:4}] {name}: {msg}", flush=True)


def check(name):
    def deco(fn):
        def run(*a, **k):
            try:
                fn(*a, **k)
            except Exception as e:                                  # one failure never stops the rest
                report(name, "FAIL", f"{type(e).__name__}: {e}")
                traceback.print_exc(limit=1)
        return run
    return deco


@check("packages")
def packages():
    need = ["torch", "torchvision", "rasterio", "pyproj", "shapely", "pystac", "pystac_client",
            "scipy", "matplotlib", "pandas", "yaml", "requests", "PIL"]
    missing = [m for m in need if importlib.util.find_spec(m) is None]
    if missing:
        report("packages", "FAIL", f"missing {missing}: pip install -r requirements_satellite.txt")
        return
    import rasterio
    import torch
    import torchvision
    opt = [m for m in ("torchgeo", "asf_search", "hyp3_sdk") if importlib.util.find_spec(m) is None]
    report("packages", "PASS", f"torch {torch.__version__}, torchvision {torchvision.__version__}, "
           f"rasterio {rasterio.__version__} (GDAL {rasterio.__gdal_version__})"
           + (f"; optional not installed: {opt}" if opt else ""))


@check("gpu")
def gpu():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report("gpu", "PASS", f"{p.name}, {p.total_memory / 1e9:.0f} GB")
    else:
        report("gpu", "WARN", "no CUDA GPU: training s04 and detection s05 will be very slow on CPU")


@check("disk")
def disk():
    free = shutil.disk_usage(SAT_DIR).free / 1e9
    report("disk", "PASS" if free > 150 else "WARN" if free > 60 else "FAIL",
           f"{free:.0f} GB free at {SAT_DIR} (plan ~150 GB: scenes 10-30 GB, Ai2 tar + chips, checkpoints)")


@check("config")
def config(cfg):
    from shapely.geometry import box
    g = aoi_grid(cfg)
    aoi = box(*cfg["aoi"]["bbox"])
    bad = []
    for name, (typ, poly) in zones_lonlat(cfg).items():
        if not poly.is_valid:
            bad.append(f"{name} is not a valid polygon (edges cross?)")
        elif not aoi.contains(poly):
            bad.append(f"{name} extends outside aoi.bbox")
        if typ not in ("transit", "anchorage"):
            bad.append(f"{name} has type {typ!r} (use transit or anchorage)")
    types = {t for t, _ in zones_lonlat(cfg).values()}
    if "transit" not in types:
        bad.append("no zone of type transit")
    msg = f"AOI grid {g.width} x {g.height} px ({g.width * g.height * 5 / 1e6:.0f} MB per date before compression); " \
          f"zones: {', '.join(cfg['zones'])}"
    report("config", "FAIL" if bad else "PASS", "; ".join(bad) if bad else msg)


@check("earth-search")
def earth_search(cfg):
    import pandas as pd
    from pystac_client import Client
    from rasterio.transform import from_origin

    from hormuz.common import Grid
    from hormuz.raster import BANDS, _scale_offset, asset_key, read_band

    t0 = time.time()
    client = Client.open(cfg["stac"]["url"])
    end = pd.Timestamp.now().normalize()
    items = list(client.search(collections=[cfg["stac"]["collection"]], bbox=cfg["aoi"]["bbox"],
                               datetime=f"{(end - pd.Timedelta(days=45)).date()}/{end.date()}",
                               max_items=50).items())
    if not items:
        report("earth-search", "FAIL", f"no {cfg['stac']['collection']} items in the last 45 days over the AOI: "
               "the collection may be frozen - try the other stac.collection value")
        return
    items.sort(key=lambda it: it.datetime, reverse=True)
    it = items[0]
    keys = {b: asset_key(it, b, cfg) for b in BANDS + ["scl"]}
    age = (end - pd.Timestamp(it.datetime.date())).days
    # read a 512 x 512 window (5 km) in the middle of the AOI from the newest item
    g = aoi_grid(cfg)
    cx = g.transform.c + g.width * 5
    cy = g.transform.f - g.height * 5
    small = Grid(g.crs, from_origin(cx - 2560, cy + 2560, 10, 10), 512, 512)
    for cand in items:                     # newest item that actually covers the AOI centre
        dn = read_band(cand.assets[asset_key(cand, "red", cfg)].href, small)
        if (dn > 0).mean() > 0.5:
            it = cand
            break
    scale, offset = _scale_offset(it, asset_key(it, "red", cfg))
    refl = np.median(dn[dn > 0]) * scale + offset if (dn > 0).any() else np.nan
    status = "PASS" if (dn > 0).mean() > 0.5 and age <= 20 else "WARN"
    report("earth-search", status,
           f"{len(items)} items in 45 days; newest {it.datetime.date()} ({age} d ago), tile cloud "
           f"{it.properties.get('eo:cloud_cover')}%; assets {keys}; 512x512 read in {time.time() - t0:.1f}s, "
           f"{(dn > 0).mean():.0%} valid, median red reflectance {refl:.3f} (open water ~0.01-0.06)")


@check("portwatch")
def portwatch(cfg):
    import pandas as pd
    import requests
    pw = cfg["portwatch"]
    r = requests.get(pw["url"], params={"where": f"portname LIKE '%{pw['name_contains']}%'", "outFields": "*",
                                        "f": "json", "orderByFields": "date DESC", "resultRecordCount": 1},
                     timeout=60)
    r.raise_for_status()
    js = r.json()
    if "error" in js or not js.get("features"):
        report("portwatch", "FAIL", f"no Hormuz rows: {js.get('error', js)}"[:300])
        return
    a = js["features"][0]["attributes"]
    d = a["date"]
    d = pd.to_datetime(d, unit="ms") if isinstance(d, (int, float)) else pd.to_datetime(d)
    report("portwatch", "PASS", f"{a.get('portname')} ({a.get('portid')}): newest {d.date()}, "
           f"n_tanker={a.get('n_tanker')}, capacity_tanker={a.get('capacity_tanker')}")


@check("ai2-data")
def ai2_data():
    import requests
    r = requests.head(AI2_TAR, timeout=30, allow_redirects=True)
    size = int(r.headers.get("Content-Length", 0)) / 1e9
    report("ai2-data", "PASS" if r.ok else "FAIL", f"HTTP {r.status_code}, {size:.1f} GB -> {AI2_TAR}")


@check("weights")
def weights(cfg):
    import torch

    from hormuz.detector import build_model
    t0 = time.time()
    model = build_model(cfg, "coco").eval()
    with torch.no_grad():
        out = model([torch.rand(3, 512, 512)])
    report("weights", "PASS", f"COCO Faster R-CNN loaded and ran in {time.time() - t0:.0f}s "
           f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, {len(out[0]['boxes'])} boxes on noise)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", action="store_true", help="also download and test the COCO weights")
    args = ap.parse_args()
    cfg = load_cfg()
    packages()
    gpu()
    disk()
    config(cfg)
    earth_search(cfg)
    portwatch(cfg)
    ai2_data()
    if args.weights:
        weights(cfg)
    n_fail = sum(s == "FAIL" for _, s, _ in results)
    n_warn = sum(s == "WARN" for _, s, _ in results)
    print(f"\n{len(results)} checks: {len(results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} fail")
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
