"""Build a small synthetic Strait of Hormuz inside a COPY of Code/satellite (never the real one).

Creates, under <root>/data/:
  * fake Sentinel-2 tiles (UTM 40N GeoTIFFs, +1000 offset like processing baseline 05.10) for
    8 dates, two tiles per date, a land strip, a cloud on one date, 18 ships per image before
    28 Feb 2026 and 3 after;
  * STAC items pointing at those tiles (data/stac/scenes.jsonl), as s01 would write;
  * 60 windows in the Ai2 rslearn layout (image.png + label/data.geojson + metadata.json with
    bounds-offset point labels), as the real tar contains.
It also shrinks the AOI and zones in that copy's config_satellite.yaml.

Usage: python tests/synthetic_world.py <copy-of-satellite-folder>
"""

import json, sys, random
from pathlib import Path
import numpy as np, yaml, rasterio
from rasterio.transform import from_origin
from PIL import Image
import pystac, datetime as dt
from pyproj import Transformer

ROOT = Path(sys.argv[1])           # .../test_env/satellite
cfgf = ROOT / "config_satellite.yaml"
cfg = yaml.safe_load(cfgf.read_text())
cfg["aoi"]["bbox"] = [56.40, 26.50, 56.52, 26.62]
cfg["zones"] = {
  "omani_tss": {"type": "transit", "polygon": [[56.41,26.52],[56.51,26.52],[56.51,26.555],[56.41,26.555]]},
  "iranian_route": {"type": "transit", "polygon": [[56.41,26.565],[56.51,26.565],[56.51,26.59],[56.41,26.59]]},
  "larak_qeshm_anchorage": {"type": "anchorage", "polygon": [[56.41,26.598],[56.51,26.598],[56.51,26.615],[56.41,26.615]]},
}
cfg["detector"].update(min_size=512, max_size=512, num_workers=2, batch_size=4, max_val_chips=200)
cfg["labels"]["hormuz_export_n"] = 12
cfg["export"]["levels_csv"] = "../data/processed/levels.csv"
cfgf.write_text(yaml.safe_dump(cfg, sort_keys=False))

rng = np.random.default_rng(0)
def draw_ship(arrs, cx, cy, L, W, ang, vals):
    h, w = arrs[0].shape
    yy, xx = np.mgrid[0:h, 0:w]
    c, s = np.cos(ang), np.sin(ang)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    m = (np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)
    for a, val in zip(arrs, vals):
        a[m] = val
    return m

# ---- synthetic Sentinel-2 tiles in UTM 40N (2 tiles split the AOI west/east) ----
tr = Transformer.from_crs("EPSG:4326", "EPSG:32640", always_xy=True)
xs, ys = tr.transform([56.39, 56.53], [26.49, 26.63])
x0, y1 = xs[0], ys[1]
W_px, H_px = int((xs[1] - xs[0]) / 10), int((ys[1] - ys[0]) / 10)
tiles_dir = ROOT / "data" / "fake_tiles"; tiles_dir.mkdir(parents=True, exist_ok=True)
items = []
dates = ["2025-11-03", "2025-12-08", "2026-01-12", "2026-02-11", "2026-03-18", "2026-05-02", "2026-06-21", "2026-08-10"]
for di, date in enumerate(dates):
    n_ships = 18 if date < "2026-02-28" else 3
    water_dn = {"red": 1300, "green": 1400, "blue": 1450, "nir": 1080}
    bands = {b: np.full((H_px, W_px), v, np.uint16) for b, v in water_dn.items()}
    scl = np.full((H_px, W_px), 6, np.uint8)
    # land strip in the south (Musandam-like)
    bands["red"][-120:] = 3500; bands["green"][-120:] = 3300; bands["blue"][-120:] = 3000; bands["nir"][-120:] = 4200; scl[-120:] = 5
    for b in bands: bands[b] = (bands[b] + rng.normal(0, 25, bands[b].shape)).clip(1, 65535).astype(np.uint16)
    if date == "2026-01-12":   # a cloud over part of it
        bands["red"][100:400, 100:500] = 6000; scl[100:400, 100:500] = 8
    for k in range(n_ships):
        cx, cy = rng.uniform(60, W_px - 60), rng.uniform(60, H_px - 200)
        L = rng.choice([8, 15, 25, 30, 33]); Wd = max(2, L / 6)
        draw_ship([bands["red"], bands["green"], bands["blue"], bands["nir"]], cx, cy, L, Wd, rng.uniform(0, np.pi), [3200, 3300, 3400, 3600])
    half = W_px // 2
    day_items = []
    for t, (c0, c1) in enumerate([(0, half + 40), (half - 40, W_px)]):
        tf = from_origin(x0 + c0 * 10, y1, 10, 10)
        hrefs = {}
        for name, arr in list(bands.items()) + [("scl", scl)]:
            f = tiles_dir / f"{date}_t{t}_{name}.tif"
            with rasterio.open(f, "w", driver="GTiff", width=c1 - c0, height=H_px, count=1, dtype=arr.dtype, crs="EPSG:32640", transform=tf, nodata=0) as dst:
                dst.write(arr[:, c0:c1], 1)
            hrefs[name] = str(f)
        item = pystac.Item(id=f"S2X_{date}_T{t}", geometry=None, bbox=None,
                           datetime=dt.datetime.fromisoformat(date + "T07:00:00+00:00"),
                           properties={"eo:cloud_cover": 5, "s2:processing_baseline": "05.10", "s2:mgrs_tile": f"40RC{t}"})
        for name, h in hrefs.items():
            item.add_asset(name, pystac.Asset(href=h))
        items.append(item)
jsonl = ROOT / "data" / "stac" / "scenes.jsonl"; jsonl.parent.mkdir(parents=True, exist_ok=True)
with open(jsonl, "w") as f:
    for it in items: f.write(json.dumps(it.to_dict()) + "\n")

# ---- synthetic Ai2 rslearn dataset ----
ai2 = ROOT / "data" / "ai2_sentinel2_vessels" / "windows" / "default"
for i in range(60):
    size = int(rng.choice([400, 700]))
    img = (25 + rng.normal(0, 2, (size, size, 3))).clip(0, 255).astype(np.uint8)
    chans = [img[..., c] for c in range(3)]
    bx, by = 100000 + i * 1000, 200000 + i * 1000
    feats = []
    for k in range(int(rng.integers(0, 6))):
        cx, cy = rng.uniform(20, size - 20), rng.uniform(20, size - 20)
        L = rng.choice([8, 15, 25, 33]); draw_ship(chans, cx, cy, L, max(2, L / 6), rng.uniform(0, np.pi), [180, 185, 190])
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [bx + cx, by + cy]}, "properties": {"category": "vessel"}})
    img = np.stack(chans, -1)
    w = ai2 / f"win_{i:03d}"
    (w / "layers" / "sentinel2" / "R_G_B").mkdir(parents=True, exist_ok=True)
    (w / "layers" / "label").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(w / "layers" / "sentinel2" / "R_G_B" / "image.png")
    (w / "layers" / "label" / "data.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    (w / "metadata.json").write_text(json.dumps({"group": "default", "name": w.name, "bounds": [bx, by, bx + size, by + size], "options": {"split": "val" if i % 6 == 0 else "train"}}))
print("synthetic world ready:", len(items), "items,", len(dates), "dates")
