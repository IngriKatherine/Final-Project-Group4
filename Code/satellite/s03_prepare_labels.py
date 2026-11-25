"""s03 - Build the labelled training chips for the CNN.

Sub-commands
    ai2      Convert the Allen Institute for AI (Ai2) Sentinel-2 vessel dataset into
             512 x 512 PNG chips + YOLO labels  ->  data/labels/ai2/{train,val}/
             (43k patches, 37k ship labels, hand-annotated, public).
             Download first (about tens of GB, check free disk):
               wget https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar
               tar xf sentinel2_vessels.tar -C data/ai2_sentinel2_vessels
    gfw      (optional) Turn an external table of ship detections with lat/lon/time
             (e.g. Global Fishing Watch Sentinel-2 detections) into labelled chips cut
             from OUR downloaded Hormuz scenes  ->  data/labels/gfw_hormuz/{train,val}/
    stats    Per-channel mean/std of the training chips (used by the ssl4eo backbone) and a
             check that Ai2 chips and our Hormuz scenes have similar brightness.
    preview  Draw the boxes on 16 random chips per source -> figures/labels_preview_<src>.png
             ALWAYS look at this before training.

Run
    python s03_prepare_labels.py ai2 --inspect      # print one window's files, change nothing
    python s03_prepare_labels.py ai2 --workers 8
    python s03_prepare_labels.py preview
    python s03_prepare_labels.py stats
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from hormuz.common import P, aoi_grid, get_logger, load_cfg, lonlat_to_pixel, save_json
from hormuz.detector import read_yolo, write_yolo

log = get_logger("s03")
VESSEL_WORDS = ("vessel", "ship", "boat")


# -----------------------------------------------------------------------------
# Ai2 rslearn dataset -> YOLO chips
# -----------------------------------------------------------------------------
def _find_image(window: Path) -> Path | None:
    for pat in ("layers/sentinel2*/R_G_B/image.png", "layers/*/R_G_B/image.png", "layers/**/image.png"):
        hits = [p for p in window.glob(pat) if "label" not in p.parts]
        if hits:
            return sorted(hits)[0]
    return None


def _points(geojson: Path, bounds, w: int, h: int) -> tuple[np.ndarray, list]:
    """Feature centres in image pixels. Ai2 stores them in projection pixel units, offset by the
    window bounds; if they already fall inside the image we keep them as they are."""
    from shapely.geometry import shape

    feats = json.loads(geojson.read_text()).get("features", [])
    pts, cats = [], []
    for f in feats:
        cat = str((f.get("properties") or {}).get("category", "vessel")).lower()
        cats.append(cat)
        if not any(wd in cat for wd in VESSEL_WORDS):
            continue
        c = shape(f["geometry"]).centroid
        pts.append((c.x, c.y))
    pts = np.asarray(pts, float).reshape(-1, 2)
    if len(pts) and bounds is not None:
        inside_raw = ((pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)).mean()
        if inside_raw < 0.5:
            pts = pts - np.array([bounds[0], bounds[1]], float)
    return pts, cats


def _split_of(window: Path, meta: dict, val_fraction: float) -> str:
    split = (meta.get("options") or {}).get("split")
    if split in ("train", "val", "test"):
        return "val" if split == "test" else split
    hsh = int(hashlib.md5(window.name.encode()).hexdigest(), 16) % 1000
    return "val" if hsh < val_fraction * 1000 else "train"


def _convert_window(args) -> dict:
    from PIL import Image

    geojson, out_root, tile, box_px, val_fraction = args
    window = geojson.parent.parent.parent
    meta_file = window / "metadata.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    img_file = _find_image(window)
    if img_file is None:
        return dict(window=window.name, status="no_image")
    img = np.asarray(Image.open(img_file).convert("RGB"))
    h, w = img.shape[:2]
    pts, cats = _points(geojson, meta.get("bounds"), w, h)
    split = _split_of(window, meta, val_fraction)
    n_chips = n_boxes = 0
    for ty in range(0, max(h, 1), tile):
        for tx in range(0, max(w, 1), tile):
            chip = np.zeros((tile, tile, 3), np.uint8)              # zero-pad edges
            sub = img[ty:ty + tile, tx:tx + tile]
            chip[:sub.shape[0], :sub.shape[1]] = sub
            if sub.size == 0 or (sub.max() == 0):
                continue
            sel = (pts[:, 0] >= tx) & (pts[:, 0] < tx + sub.shape[1]) & \
                  (pts[:, 1] >= ty) & (pts[:, 1] < ty + sub.shape[0]) if len(pts) else np.zeros(0, bool)
            p = pts[sel] - [tx, ty] if len(pts) else np.zeros((0, 2))
            boxes = np.concatenate([p - box_px / 2, p + box_px / 2], 1) if len(p) else np.zeros((0, 4))
            stem = f"{window.parent.name}_{window.name}_{ty}_{tx}".replace(" ", "_")
            Image.fromarray(chip).save(out_root / split / "images" / f"{stem}.png")
            write_yolo(out_root / split / "labels" / f"{stem}.txt", boxes, tile, tile)
            n_chips += 1
            n_boxes += len(boxes)
    return dict(window=window.name, status="ok", split=split, chips=n_chips, boxes=n_boxes,
                categories=";".join(sorted(set(cats))))


def cmd_ai2(cfg, args):
    root = P(cfg, "ai2_raw")
    geojsons = sorted(root.glob("**/layers/label/data.geojson"))
    log.info("found %d labelled windows under %s", len(geojsons), root)
    if not geojsons:
        log.error("no layers/label/data.geojson found - did you extract the tar into %s ?", root)
        return
    if args.inspect:
        g = geojsons[0]
        win = g.parent.parent.parent
        log.info("example window: %s", win)
        for p in sorted(win.rglob("*"))[:40]:
            log.info("  %s", p.relative_to(win))
        meta = win / "metadata.json"
        if meta.exists():
            log.info("metadata.json: %s", meta.read_text()[:600])
        log.info("data.geojson (first 600 chars): %s", g.read_text()[:600])
        return
    out = P(cfg, "labels_dir") / "ai2"
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    if args.limit:
        geojsons = geojsons[: args.limit]
    lab = cfg["labels"]
    jobs = [(g, out, lab["tile"], lab["ai2_box_px"], lab["val_fraction"]) for g in geojsons]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_convert_window, jobs, chunksize=32))
    df = pd.DataFrame(rows)
    df.to_csv(out / "conversion_log.csv", index=False)
    ok = df[df["status"] == "ok"]
    log.info("windows ok=%d no_image=%d", len(ok), (df["status"] == "no_image").sum())
    log.info("chips per split:\n%s", ok.groupby("split")[["chips", "boxes"]].sum().to_string())
    cats = sorted(set(";".join(ok["categories"].fillna("")).split(";")) - {""})
    log.info("label categories seen: %s (kept those containing %s)", cats, VESSEL_WORDS)


# -----------------------------------------------------------------------------
# External detections (e.g. GFW Sentinel-2) -> chips from our Hormuz scenes
# -----------------------------------------------------------------------------
def cmd_gfw(cfg, args):
    from PIL import Image

    from hormuz.raster import read_scene

    f = Path(args.csv) if args.csv else P(cfg, "gfw_csv")
    df = pd.read_csv(f)
    log.info("columns in %s: %s", f.name, list(df.columns))
    need = [args.lat_col, args.lon_col, args.time_col]
    if any(c not in df.columns for c in need):
        log.error("set --lat-col/--lon-col/--time-col to the column names above")
        return
    df["date"] = pd.to_datetime(df[args.time_col], utc=True, errors="coerce").dt.date.astype(str)
    lon0, lat0, lon1, lat1 = cfg["aoi"]["bbox"]
    df = df[df[args.lon_col].between(lon0, lon1) & df[args.lat_col].between(lat0, lat1)]
    log.info("%d detections inside the AOI", len(df))
    idx = pd.read_csv(P(cfg, "scene_index"))
    idx = idx[idx["status"] == "ok"]
    grid = aoi_grid(cfg)
    tile, out = cfg["labels"]["tile"], P(cfg, "labels_dir") / "gfw_hormuz"
    dates = sorted(set(idx["date"]) & set(df["date"]))
    log.info("%d dates have both a downloaded scene and detections", len(dates))
    rng = random.Random(0)
    n = 0
    for k, d in enumerate(dates):
        split = "val" if k % 5 == 4 else "train"            # every 5th date held out
        rgbn, _, _ = read_scene(P(cfg, "scene_dir") / f"S2_{d.replace('-', '')}.tif")
        rgb = rgbn[:3].transpose(1, 2, 0)
        sub = df[df["date"] == d]
        cols, rows = lonlat_to_pixel(grid, sub[args.lon_col].values, sub[args.lat_col].values)
        side = np.full(len(sub), float(cfg["labels"]["ai2_box_px"]))
        if args.length_col and args.length_col in sub:
            side = np.clip(sub[args.length_col].astype(float).values / cfg["aoi"]["resolution"], 6, 45)
        centres = list(zip(cols, rows))
        for cx, cy in centres[: args.max_chips_per_date]:
            x0 = int(np.clip(cx - tile // 2 + rng.randint(-150, 150), 0, grid.width - tile))
            y0 = int(np.clip(cy - tile // 2 + rng.randint(-150, 150), 0, grid.height - tile))
            chip = rgb[y0:y0 + tile, x0:x0 + tile]
            if (chip[..., 0] == 0).mean() > 0.2:
                continue
            inside = (cols >= x0) & (cols < x0 + tile) & (rows >= y0) & (rows < y0 + tile)
            px, py, s = cols[inside] - x0, rows[inside] - y0, side[inside]
            boxes = np.stack([px - s / 2, py - s / 2, px + s / 2, py + s / 2], 1)
            stem = f"{d}_{y0}_{x0}"
            (out / split / "images").mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.ascontiguousarray(chip)).save(out / split / "images" / f"{stem}.png")
            write_yolo(out / split / "labels" / f"{stem}.txt", boxes, tile, tile)
            n += 1
    log.info("wrote %d chips to %s", n, out)


# -----------------------------------------------------------------------------
# Stats and preview
# -----------------------------------------------------------------------------
def _sample_images(cfg, n=400):
    from PIL import Image

    root = P(cfg, "labels_dir")
    out = {}
    for src in sorted(p.name for p in root.iterdir() if p.is_dir()):
        files = sorted((root / src / "train" / "images").glob("*.png"))
        random.Random(0).shuffle(files)
        if files:
            out[src] = [np.asarray(Image.open(f).convert("RGB")) for f in files[:n]]
    return out


def cmd_stats(cfg, args):
    from hormuz.raster import read_scene, valid_water

    stats = {}
    for src, imgs in _sample_images(cfg).items():
        sub = [im[::4, ::4].reshape(-1, 3) for im in imgs]                 # every 4th pixel is plenty
        px = np.concatenate([p[p.max(1) > 0] for p in sub]).astype(np.float32) / 255.0
        stats[src] = dict(mean=px.mean(0).round(4).tolist(), std=px.std(0).round(4).tolist(),
                          median=np.median(px, 0).round(4).tolist(), n_images=len(imgs))
        log.info("%-12s mean=%s std=%s median=%s", src, stats[src]["mean"], stats[src]["std"], stats[src]["median"])
    idx_file = P(cfg, "scene_index")
    if idx_file.exists():
        idx = pd.read_csv(idx_file)
        idx = idx[idx["status"] == "ok"].sort_values("cloud_frac").head(3)
        meds = []
        for f in idx["file"]:
            rgbn, scl, _ = read_scene(P(cfg, "scene_dir") / f)
            wmask = valid_water(rgbn, scl)
            meds.append([float(np.median(rgbn[i][wmask])) / 255 for i in range(3)])
        stats["hormuz_water_median"] = np.mean(meds, 0).round(4).tolist() if meds else None
        log.info("Hormuz open-water median RGB (0-1): %s  -> compare with the medians above. "
                 "If they differ by >2x, change image.scale_max_reflectance and re-run s02, "
                 "or rely on the brightness augmentation + Hormuz fine-tuning.", stats["hormuz_water_median"])
    pooled = [s for k, s in stats.items() if isinstance(s, dict)]
    if pooled:
        stats["train_mean"] = np.mean([s["mean"] for s in pooled], 0).round(4).tolist()
        stats["train_std"] = np.mean([s["std"] for s in pooled], 0).round(4).tolist()
    save_json(stats, P(cfg, "labels_dir") / "stats.json")


def cmd_preview(cfg, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    root = P(cfg, "labels_dir")
    for src in sorted(p.name for p in root.iterdir() if p.is_dir()):
        files = sorted((root / src / "train" / "images").glob("*.png"))
        files = [f for f in files if (f.parent.parent / "labels" / (f.stem + ".txt")).stat().st_size > 0] or files
        random.Random(1).shuffle(files)
        if not files:
            continue
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, f in zip(axes.ravel(), files[:16]):
            im = np.asarray(Image.open(f).convert("RGB"))
            ax.imshow(np.clip(im.astype(float) * 2.5, 0, 255).astype(np.uint8))   # brighten water scenes
            for x0, y0, x1, y1 in read_yolo(f.parent.parent / "labels" / (f.stem + ".txt"), *im.shape[1::-1]):
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1))
            ax.set_title(f.stem[:28], fontsize=7)
            ax.axis("off")
        out = P(cfg, "figures_dir") / f"labels_preview_{src}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        log.info("wrote %s", out)


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ai2")
    a.add_argument("--inspect", action="store_true")
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--limit", type=int, default=None, help="convert only the first N windows")
    g = sub.add_parser("gfw")
    g.add_argument("--csv", default=None)
    g.add_argument("--lat-col", default="lat")
    g.add_argument("--lon-col", default="lon")
    g.add_argument("--time-col", default="timestamp")
    g.add_argument("--length-col", default=None)
    g.add_argument("--max-chips-per-date", type=int, default=20)
    sub.add_parser("stats")
    sub.add_parser("preview")
    args = ap.parse_args()
    {"ai2": cmd_ai2, "gfw": cmd_gfw, "stats": cmd_stats, "preview": cmd_preview}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
