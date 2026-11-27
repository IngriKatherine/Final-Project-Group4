"""s05 - Run the trained CNN over every downloaded Hormuz image.

For each date:
  * sliding-window detection (512 px tiles, 64 px overlap, NMS across tiles) over the
    tiles that touch a zone and contain visible water;
  * drop detections on land or under cloud (checked in a ring around the centre: the
    hull itself is bright in near-infrared, so it is never "water" on its own);
  * measure each hull (length, width, heading) from the near-infrared band;
  * assign the zone (omani_tss, iranian_route, larak_qeshm_anchorage, or other);
  * record, per zone, how much water was actually visible (cloud/no-data reduce it).

Outputs (appended per date, so the script can be stopped and restarted)
    data/detections/detections.csv   one row per ship
    data/detections/coverage.csv     one row per date x zone
    figures/quicklooks/YYYY-MM-DD.png   (with --quicklook) downsampled image + boxes

Run
    python s05_detect_scenes.py --ckpt runs/hormuz_ft/best.pt --quicklook
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hormuz.common import P, aoi_grid, get_logger, load_cfg, pixel_to_lonlat, zone_masks
from hormuz.detector import build_model, predict_scene
from hormuz.measure import dwt_from_length, hull_size, size_class
from hormuz.raster import SCL_CLOUD, read_scene, valid_water

log = get_logger("s05")


def ring_fraction(mask: np.ndarray, cx: float, cy: float, r_out: int = 12, r_in: int = 4) -> float:
    """Share of True pixels in a square ring around (cx, cy)."""
    h, w = mask.shape
    x, y = int(round(cx)), int(round(cy))
    y0, y1, x0, x1 = max(y - r_out, 0), min(y + r_out + 1, h), max(x - r_out, 0), min(x + r_out + 1, w)
    win = mask[y0:y1, x0:x1].copy()
    iy, ix = y - y0, x - x0
    win[max(iy - r_in, 0):iy + r_in + 1, max(ix - r_in, 0):ix + r_in + 1] = False
    n = (y1 - y0) * (x1 - x0) - min(2 * r_in + 1, y1 - y0) * min(2 * r_in + 1, x1 - x0)
    return float(win.sum()) / max(n, 1)


def quicklook(file, rgbn, boxes, kinds, zones):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = 4
    rgb = rgbn[:3, ::f, ::f].transpose(1, 2, 0).astype(float)
    rgb = np.clip(rgb * 3.0, 0, 255).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 9.5))
    ax.imshow(rgb)
    for name, m in zones.items():
        ax.contour(m[::f, ::f], levels=[0.5], colors="white", linewidths=0.6)
    colors = {"large": "red", "medium": "orange", "small": "cyan", "unknown": "yellow"}
    for (x0, y0, x1, y1), k in zip(boxes, kinds):
        ax.add_patch(plt.Rectangle((x0 / f, y0 / f), max((x1 - x0) / f, 2), max((y1 - y0) / f, 2),
                                   fill=False, ec=colors.get(k, "yellow"), lw=0.8))
    ax.set_title(f"{file.stem}: {sum(k == 'large' for k in kinds)} large, {len(kinds)} total")
    ax.axis("off")
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file, dpi=100)
    plt.close(fig)


def main():
    cfg = load_cfg()
    inf, meas = cfg["inference"], cfg["measure"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min-score", type=float, default=inf["min_score"])
    ap.add_argument("--quicklook", action="store_true")
    ap.add_argument("--all-water", action="store_true", help="detect over the whole AOI, not only the zones")
    ap.add_argument("--redo", action="store_true", help="ignore previous outputs")
    args = ap.parse_args()

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(cfg, ck["backbone"], ck.get("stats"))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    log.info("model %s (backbone %s, epoch %s) on %s", args.ckpt, ck["backbone"], ck.get("epoch"), device)

    grid = aoi_grid(cfg)
    zones = zone_masks(cfg, grid)
    ztype = {n: z["type"] for n, z in cfg["zones"].items()}
    zone_any = np.any(np.stack(list(zones.values())), 0)
    zone_id = np.zeros((grid.height, grid.width), np.int16)          # 0 = other
    names = list(zones)
    for i, n in enumerate(names, 1):
        zone_id[zones[n] & (zone_id == 0)] = i

    det_file, cov_file = P(cfg, "detections_csv"), P(cfg, "coverage_csv")
    if args.redo:
        det_file.unlink(missing_ok=True)
        cov_file.unlink(missing_ok=True)
    done = set(pd.read_csv(cov_file)["date"]) if cov_file.exists() else set()
    idx = pd.read_csv(P(cfg, "scene_index"))
    idx = idx[idx["status"] == "ok"].sort_values("date")
    todo = [r for _, r in idx.iterrows() if r["date"] not in done]
    log.info("%d dates to process (%d done)", len(todo), len(done))

    for k, r in enumerate(todo, 1):
        rgbn, scl, _ = read_scene(P(cfg, "scene_dir") / r["file"])
        water = valid_water(rgbn, scl, inf["ndwi_min"])
        cloud = np.isin(scl, list(SCL_CLOUD))
        search = water if args.all_water else (water & zone_any)
        boxes, scores = predict_scene(model, rgbn[:3].transpose(1, 2, 0), device, tile=inf["tile"],
                                      stride=inf["stride"], min_score=args.min_score,
                                      nms_iou=inf["nms_iou"], skip_mask=search)
        rows, kept_boxes, kinds = [], [], []
        if len(boxes):
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            lon, lat = pixel_to_lonlat(grid, cx, cy)
            for j in range(len(boxes)):
                wfrac = ring_fraction(water | cloud, cx[j], cy[j])        # sea around it (cloud may hide sea)
                cfrac = ring_fraction(cloud, cx[j], cy[j])
                if wfrac < 0.4 or cfrac > 0.5:
                    continue                                              # land, port, or cloud edge
                m = hull_size(rgbn[3], cx[j], cy[j], meas["crop"], meas["k_mad"], cfg["aoi"]["resolution"])
                zi = int(zone_id[int(round(cy[j])) % grid.height, int(round(cx[j])) % grid.width])
                zone = names[zi - 1] if zi else "other"
                L = m["length_m"]
                sc = size_class(L, cfg)
                rows.append(dict(date=r["date"], x=round(float(cx[j]), 1), y=round(float(cy[j]), 1),
                                 lon=round(float(lon[j]), 5), lat=round(float(lat[j]), 5),
                                 score=round(float(scores[j]), 3),
                                 box_w=round(float(boxes[j, 2] - boxes[j, 0]), 1),
                                 box_h=round(float(boxes[j, 3] - boxes[j, 1]), 1),
                                 length_m=round(L, 1) if np.isfinite(L) else np.nan,
                                 width_m=round(m["width_m"], 1) if np.isfinite(m["width_m"]) else np.nan,
                                 heading_deg=round(m["heading_deg"], 1) if np.isfinite(m["heading_deg"]) else np.nan,
                                 size_class=sc, zone=zone, zone_type=ztype.get(zone, "other"),
                                 dwt_est=round(float(dwt_from_length(L, cfg)), 0) if sc == "large" else 0.0))
                kept_boxes.append(boxes[j])
                kinds.append(sc)
        cov = [dict(date=r["date"], zone=n, zone_type=ztype[n], zone_px=int(zones[n].sum()),
                    water_px=int((water & zones[n]).sum()),
                    cloud_px=int((cloud & zones[n]).sum())) for n in names]
        if rows:
            pd.DataFrame(rows).to_csv(det_file, mode="a", header=not det_file.exists(), index=False)
        pd.DataFrame(cov).to_csv(cov_file, mode="a", header=not cov_file.exists(), index=False)
        n_large = sum(1 for x in rows if x["size_class"] == "large" and x["zone"] != "other")
        log.info("[%d/%d] %s  ships=%d  large-in-zones=%d", k, len(todo), r["date"], len(rows), n_large)
        if args.quicklook:
            quicklook(P(cfg, "figures_dir") / "quicklooks" / f"{r['date']}.png", rgbn,
                      np.asarray(kept_boxes).reshape(-1, 4), kinds, zones)


if __name__ == "__main__":
    main()
