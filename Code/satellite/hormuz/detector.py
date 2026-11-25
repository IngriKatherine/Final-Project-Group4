"""Vessel detector: torchvision Faster R-CNN (ResNet-50 + FPN) with pretrained weights.

Ways to initialise it (config detector.backbone, or --backbone):
    coco     whole detector pretrained on COCO (torchvision weights)      <- default
    imagenet only the ResNet-50 backbone pretrained on ImageNet
    ssl4eo   ResNet-50 pretrained on Sentinel-2 RGB with MoCo (SSL4EO-S12, via torchgeo)
    scratch  no pretraining at all (the control in the transfer-learning comparison)

Ships are 1-35 px long at 10 m, so the anchor sizes are smaller than torchvision's
defaults and 512-px chips are upsampled to 800 px. The number of anchors per location
stays 3, so the pretrained RPN head weights still load.

Label format: YOLO text files, one line per box: "class cx cy w h" (normalised 0-1).
Class 0 = vessel. Images: 8-bit RGB PNG chips.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
def build_model(cfg: dict, backbone: str | None = None, stats: dict | None = None):
    import torch.nn as nn
    from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, FastRCNNPredictor
    from torchvision.models.detection.rpn import RPNHead

    d = cfg["detector"]
    backbone = backbone or d["backbone"]
    sizes = tuple((int(s),) for s in d["anchor_sizes"])
    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes))
    common = dict(min_size=d["min_size"], max_size=d["max_size"], box_detections_per_img=300,
                  rpn_pre_nms_top_n_test=2000, rpn_post_nms_top_n_test=1000,
                  box_score_thresh=0.05)
    mean = (stats or {}).get("mean", IMAGENET_MEAN)
    std = (stats or {}).get("std", IMAGENET_STD)

    if backbone in ("coco", "imagenet", "scratch"):
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
        kw = {"coco": dict(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
              "imagenet": dict(weights=None, weights_backbone=ResNet50_Weights.IMAGENET1K_V2, num_classes=91),
              "scratch": dict(weights=None, weights_backbone=None, num_classes=91)}[backbone]
        model = fasterrcnn_resnet50_fpn_v2(trainable_backbone_layers=5, image_mean=IMAGENET_MEAN,
                                           image_std=IMAGENET_STD, **kw, **common)
        model.rpn.anchor_generator = anchors
    elif backbone == "ssl4eo":
        from torchvision.models import resnet50
        from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
        try:
            from torchgeo.models import ResNet50_Weights as GeoWeights
        except ImportError as e:
            raise ImportError("backbone=ssl4eo needs torchgeo: pip install torchgeo") from e
        net = resnet50(weights=None)
        state = GeoWeights.SENTINEL2_RGB_MOCO.get_state_dict(progress=True)
        missing, unexpected = net.load_state_dict(state, strict=False)
        print(f"SSL4EO weights loaded; missing={len(missing)} unexpected={len(unexpected)}")
        body = _resnet_fpn_extractor(net, trainable_layers=5, norm_layer=nn.BatchNorm2d)
        rpn_head = RPNHead(body.out_channels, 3, conv_depth=2)
        box_head = FastRCNNConvFCHead((body.out_channels, 7, 7), [256, 256, 256, 256], [1024],
                                      norm_layer=nn.BatchNorm2d)
        # SSL4EO was trained on reflectance, not ImageNet colours: normalise with our own
        # training-set statistics (written by s03_prepare_labels.py stats).
        model = FasterRCNN(body, num_classes=2, rpn_anchor_generator=anchors, rpn_head=rpn_head,
                           box_head=box_head, image_mean=mean, image_std=std, **common)
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)   # background + vessel
    return model


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def read_yolo(file: Path, w: int, h: int) -> np.ndarray:
    """YOLO txt -> (N, 4) xyxy pixel boxes."""
    if not file.exists():
        return np.zeros((0, 4), np.float32)
    rows = [l.split() for l in file.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32)
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def write_yolo(file: Path, boxes: np.ndarray, w: int, h: int) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x0, y0, x1, y1 in np.asarray(boxes).reshape(-1, 4):
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        lines.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                     f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    file.write_text("\n".join(lines) + ("\n" if lines else ""))


def list_chips(labels_dir: Path, sources: list[str], split: str) -> list[tuple[Path, Path]]:
    pairs = []
    for src in sources:
        img_dir = labels_dir / src / split / "images"
        for img in sorted(img_dir.glob("*.png")):
            pairs.append((img, labels_dir / src / split / "labels" / (img.stem + ".txt")))
    return pairs


def augment(img: np.ndarray, boxes: np.ndarray, rng: random.Random):
    """Geometric + radiometric augmentation that is valid for nadir satellite chips.

    Ships have no preferred heading, so flips and 90-degree rotations are free labels.
    Brightness/contrast/gamma jitter covers differences in sun angle, haze and scaling.
    """
    h, w = img.shape[:2]
    b = boxes.copy()
    if rng.random() < 0.5:                                   # horizontal flip
        img = img[:, ::-1]
        b[:, [0, 2]] = w - b[:, [2, 0]]
    if rng.random() < 0.5:                                   # vertical flip
        img = img[::-1]
        b[:, [1, 3]] = h - b[:, [3, 1]]
    if rng.random() < 0.5 and h == w:                        # 90-degree rotation (clockwise)
        img = np.rot90(img, k=-1)
        b = np.stack([h - b[:, 3], b[:, 0], h - b[:, 1], b[:, 2]], 1) if len(b) else b
    img = img.astype(np.float32) / 255.0
    img = img * rng.uniform(0.8, 1.25) + rng.uniform(-0.05, 0.05)        # contrast, brightness
    img = np.clip(img, 0, 1) ** rng.uniform(0.8, 1.25)                     # gamma
    if rng.random() < 0.3:
        img = img + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.01, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), b.astype(np.float32)


class ChipDataset:
    """(image tensor CHW float 0-1, target dict) pairs for torchvision detection models."""

    def __init__(self, pairs: list[tuple[Path, Path]], train: bool, seed: int = 0):
        self.pairs, self.train, self.seed = pairs, train, seed
        self._rng, self._worker = None, None

    @property
    def rng(self) -> random.Random:
        """One RNG per DataLoader worker; otherwise every worker would replay the same augmentations."""
        import torch
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else -1
        if self._rng is None or self._worker != wid:
            base = info.seed if info is not None else self.seed
            self._rng, self._worker = random.Random(base), wid
        return self._rng

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import torch
        from PIL import Image

        img_file, lab_file = self.pairs[i]
        img = np.asarray(Image.open(img_file).convert("RGB"))
        h, w = img.shape[:2]
        boxes = read_yolo(lab_file, w, h)
        if self.train:
            img, boxes = augment(img, boxes, self.rng)
        else:
            img = img.astype(np.float32) / 255.0
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones(len(boxes), dtype=torch.int64),
                  "image_id": torch.tensor(i)}
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), target


def collate(batch):
    return tuple(zip(*batch))


# -----------------------------------------------------------------------------
# Metrics: detections are matched to labels by centre distance (labels from points
# have no true box size, so IoU would be misleading)
# -----------------------------------------------------------------------------
def _centres(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b).reshape(-1, 4)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)


def match(pred_boxes, scores, gt_boxes, max_dist: float):
    """Greedy matching by score. Returns (is_tp per prediction sorted by score, scores sorted)."""
    order = np.argsort(-np.asarray(scores))
    pc = _centres(pred_boxes)[order]
    gc = _centres(gt_boxes)
    used = np.zeros(len(gc), bool)
    tp = np.zeros(len(pc), bool)
    for k, c in enumerate(pc):
        if len(gc) == 0:
            break
        d = np.hypot(*(gc - c).T)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= max_dist:
            used[j], tp[k] = True, True
    return tp, np.asarray(scores)[order]


def detection_metrics(results: list[dict], max_dist: float, score_thr: float) -> dict:
    """results: [{'boxes','scores','gt'}] per image -> AP, P, R, F1, count errors."""
    all_tp, all_sc, n_gt = [], [], 0
    count_err, tp_at, fp_at, fn_at = [], 0, 0, 0
    for r in results:
        tp, sc = match(r["boxes"], r["scores"], r["gt"], max_dist)
        all_tp.append(tp)
        all_sc.append(sc)
        n_gt += len(r["gt"])
        keep = sc >= score_thr
        tp_at += int(tp[keep].sum())
        fp_at += int((~tp[keep]).sum())
        fn_at += len(r["gt"]) - int(tp[keep].sum())
        count_err.append(int(keep.sum()) - len(r["gt"]))
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    sc = np.concatenate(all_sc) if all_sc else np.zeros(0)
    order = np.argsort(-sc)
    tp = tp[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    # all-point interpolated AP
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    p = tp_at / max(tp_at + fp_at, 1)
    r = tp_at / max(tp_at + fn_at, 1)
    ce = np.asarray(count_err, float)
    return dict(ap=ap, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9),
                count_mae=float(np.mean(np.abs(ce))) if len(ce) else np.nan,
                count_bias=float(np.mean(ce)) if len(ce) else np.nan,
                n_images=len(results), n_gt=n_gt)


@np.errstate(all="ignore")
def evaluate(model, loader, device, max_dist: float, score_thr: float) -> dict:
    import torch

    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in loader:
            outs = model([im.to(device) for im in images])
            for o, t in zip(outs, targets):
                results.append(dict(boxes=o["boxes"].cpu().numpy(), scores=o["scores"].cpu().numpy(),
                                    gt=t["boxes"].numpy()))
    return detection_metrics(results, max_dist, score_thr)


# -----------------------------------------------------------------------------
# Inference on a whole scene
# -----------------------------------------------------------------------------
def predict_scene(model, rgb_u8: np.ndarray, device, tile: int = 512, stride: int = 448,
                  min_score: float = 0.5, nms_iou: float = 0.3, batch: int = 8,
                  skip_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window detection over an (H, W, 3) uint8 image -> (boxes xyxy, scores).

    skip_mask: boolean (H, W); tiles with no True pixel (e.g. no visible water) are skipped.
    """
    import torch
    from torchvision.ops import nms

    model.eval()
    h, w = rgb_u8.shape[:2]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) + ([h - tile] if h > tile and (h - tile) % stride else [])
    xs = list(range(0, max(w - tile, 0) + 1, stride)) + ([w - tile] if w > tile and (w - tile) % stride else [])
    windows = [(y, x) for y in ys for x in xs
               if skip_mask is None or skip_mask[y:y + tile, x:x + tile].any()]
    all_b, all_s = [], []
    with torch.no_grad():
        for k in range(0, len(windows), batch):
            chunk = windows[k:k + batch]
            ims = [torch.from_numpy(np.ascontiguousarray(
                rgb_u8[y:y + tile, x:x + tile].transpose(2, 0, 1))).float().div(255).to(device)
                for y, x in chunk]
            for (y, x), o in zip(chunk, model(ims)):
                keep = o["scores"] >= min_score
                b = o["boxes"][keep].cpu().numpy()
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                all_b.append(b)
                all_s.append(o["scores"][keep].cpu().numpy())
    if not all_b or sum(len(b) for b in all_b) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    b = torch.as_tensor(np.concatenate(all_b))
    s = torch.as_tensor(np.concatenate(all_s))
    keep = nms(b, s, nms_iou).numpy()
    return b.numpy()[keep], s.numpy()[keep]
