"""s03b - Hand-label chips from OUR Strait of Hormuz scenes (the in-domain labels).

export  Cut ~300 chips (512 x 512) from the downloaded scenes, half before and half
        after the 28 Feb 2026 closure, centred on likely ships so labelling time isn't
        spent on empty sea. Each chip comes with PRE-ANNOTATIONS (YOLO .txt):
          * from a trained detector if you pass --ckpt (best: run s04 on Ai2 first), or
          * from simple bright-blob candidates otherwise.
        -> data/hormuz_to_label/images, /labels, manifest.csv
        Correct the boxes in any YOLO-capable tool (Label Studio, CVAT, Roboflow, or the
        lightweight `labelImg`). Rule: one box around each hull (not the wake), every ship
        longer than ~3 px, including dark/AIS-off ships, anchored ships and small craft.
import  Copy the corrected chips into data/labels/hormuz/{train,val,test}, split BY DATE
        (whole dates go to one split, so neighbouring chips of one image never leak).

Run
    python s03b_hormuz_chips.py export --n 300 --ckpt runs/ai2_coco/best.pt
    # ... label ...
    python s03b_hormuz_chips.py import --from data/hormuz_to_label
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from hormuz.common import P, aoi_grid, get_logger, load_cfg, zone_masks
from hormuz.detector import read_yolo, write_yolo
from hormuz.raster import read_scene, valid_water

log = get_logger("s03b")


def blob_candidates(nir: np.ndarray, water: np.ndarray, k: float = 6.0, min_px: int = 3):
    """Bright compact objects on visible water -> (N, 4) boxes. A cheap pre-annotation."""
    from scipy import ndimage

    w = nir[water]
    if w.size < 1000:
        return np.zeros((0, 4))
    med = np.median(w)
    mad = np.median(np.abs(w - med)) * 1.4826 + 1
    near_water = ndimage.binary_dilation(water, iterations=3)
    bright = (nir > med + k * mad) & near_water
    lab, n = ndimage.label(bright)
    boxes = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh * ww >= min_px and max(hh, ww) <= 60:
            boxes.append([sl[1].start - 1, sl[0].start - 1, sl[1].stop + 1, sl[0].stop + 1])
    return np.asarray(boxes, float).reshape(-1, 4)


def cmd_export(cfg, args):
    from PIL import Image

    idx = pd.read_csv(P(cfg, "scene_index"))
    idx = idx[(idx["status"] == "ok") & (idx["cloud_frac"] < 0.3)].sort_values("date")
    split_date = cfg["labels"]["hormuz_split_date"]
    groups = [idx[idx["date"] < split_date], idx[idx["date"] >= split_date]]
    rng = random.Random(0)
    grid = aoi_grid(cfg)
    zones = zone_masks(cfg, grid)
    zone_any = np.any(np.stack(list(zones.values())), 0)
    tile = cfg["labels"]["tile"]
    out = P(cfg, "hormuz_export")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    model = device = None
    if args.ckpt:
        import torch

        from hormuz.detector import build_model, predict_scene
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = build_model(cfg, ck["backbone"], ck.get("stats"))
        model.load_state_dict(ck["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    rows = []
    per_group = args.n // 2
    for g in groups:
        if g.empty:
            continue
        dates = g["date"].tolist()
        rng.shuffle(dates)
        dates = dates[: max(1, min(len(dates), args.max_dates // 2))]
        per_date = int(np.ceil(per_group / len(dates)))
        for d in dates:
            rgbn, scl, _ = read_scene(P(cfg, "scene_dir") / f"S2_{d.replace('-', '')}.tif")
            water = valid_water(rgbn, scl, cfg["inference"]["ndwi_min"])
            # pre-annotate EVERY visible ship (a chip must be labelled completely, or the model
            # learns to ignore ships), but centre the chips on ships inside the zones
            if model is not None:
                boxes, _ = predict_scene(model, rgbn[:3].transpose(1, 2, 0), device,
                                         min_score=0.3, skip_mask=water)
            else:
                boxes = blob_candidates(rgbn[3], water)
            centres = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
            # where to centre chips: the detector's ships, plus bright-object candidates so a
            # weak first model still leads us to ships (pre-annotations stay the detector's)
            cand = centres
            if model is not None:
                bb = blob_candidates(rgbn[3], water)
                cand = np.concatenate([centres, np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]])
            in_zone = zone_any[cand[:, 1].astype(int).clip(0, grid.height - 1),
                               cand[:, 0].astype(int).clip(0, grid.width - 1)] if len(cand) else np.zeros(0, bool)
            seeds = cand[in_zone]
            # 70% of chips around candidates, 30% random visible water inside the zones (negatives)
            wy, wx = np.nonzero(water[::16, ::16] & zone_any[::16, ::16])
            picks = []
            for _ in range(per_date):
                if len(seeds) and rng.random() < 0.7:
                    cx, cy = seeds[rng.randrange(len(seeds))]
                elif len(wy):
                    j = rng.randrange(len(wy))
                    cx, cy = wx[j] * 16, wy[j] * 16
                else:
                    continue
                x0 = int(np.clip(cx - tile // 2 + rng.randint(-120, 120), 0, grid.width - tile))
                y0 = int(np.clip(cy - tile // 2 + rng.randint(-120, 120), 0, grid.height - tile))
                picks.append((x0, y0))
            for x0, y0 in picks:
                chip = rgbn[:3, y0:y0 + tile, x0:x0 + tile].transpose(1, 2, 0)
                if (chip[..., 0] == 0).mean() > 0.2:
                    continue
                sel = (centres[:, 0] >= x0) & (centres[:, 0] < x0 + tile) & \
                      (centres[:, 1] >= y0) & (centres[:, 1] < y0 + tile) if len(centres) else np.zeros(0, bool)
                b = boxes[sel] - [x0, y0, x0, y0] if len(centres) else np.zeros((0, 4))
                stem = f"hz_{d}_{y0}_{x0}"
                Image.fromarray(np.ascontiguousarray(chip)).save(out / "images" / f"{stem}.png")
                write_yolo(out / "labels" / f"{stem}.txt", b, tile, tile)
                rows.append(dict(chip=stem, date=d, x0=x0, y0=y0, pre_boxes=len(b),
                                 period="before" if d < split_date else "after"))
            log.info("%s: %d chips", d, len(picks))
    (out / "classes.txt").write_text("vessel\n")
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    log.info("exported %d chips to %s (pre-annotations from %s)", len(rows), out,
             "the detector" if model is not None else "bright-blob candidates")


def cmd_import(cfg, args):
    src = P(cfg, "hormuz_export") if args.src is None else Path(args.src)
    man = pd.read_csv(src / "manifest.csv")
    dates = sorted(man["date"].unique())
    rng = random.Random(0)
    rng.shuffle(dates)
    n = len(dates)
    test_d = set(dates[: max(1, round(n * 0.2))])
    val_d = set(dates[max(1, round(n * 0.2)): max(2, round(n * 0.35))])
    out = P(cfg, "labels_dir") / "hormuz"
    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for _, r in man.iterrows():
        split = "test" if r["date"] in test_d else "val" if r["date"] in val_d else "train"
        img = src / "images" / f"{r['chip']}.png"
        lab = src / "labels" / f"{r['chip']}.txt"
        if not img.exists():
            continue
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / split / "images" / img.name)
        tile = cfg["labels"]["tile"]
        boxes = read_yolo(lab, tile, tile)
        write_yolo(out / split / "labels" / lab.name, boxes, tile, tile)
        counts.setdefault(split, [0, 0])
        counts[split][0] += 1
        counts[split][1] += len(boxes)
    for s, (c, b) in sorted(counts.items()):
        log.info("%-5s chips=%d boxes=%d", s, c, b)


def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=cfg["labels"]["hormuz_export_n"])
    e.add_argument("--max-dates", type=int, default=30)
    e.add_argument("--ckpt", default=None)
    i = sub.add_parser("import")
    i.add_argument("--from", dest="src", default=None)
    args = ap.parse_args()
    cmd_export(cfg, args) if args.cmd == "export" else cmd_import(cfg, args)


if __name__ == "__main__":
    main()
