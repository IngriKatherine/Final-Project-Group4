"""s04 - Fine-tune a pretrained Faster R-CNN to find ships in Sentinel-2 chips.

Transfer learning in two stages (run both, compare in the report):
    1. Ai2 chips (37k hand-labelled ships worldwide), starting from a pretrained network
         python s04_train_detector.py --sources ai2 --backbone coco --tag ai2_coco
    2. Fine-tune that model on our hand-labelled Hormuz chips (in-domain)
         python s04_train_detector.py --sources hormuz --init runs/ai2_coco/best.pt \
                --epochs 15 --lr 0.003 --tag hormuz_ft
Backbone ablation (H: domain pretraining beats ImageNet/COCO):
    --backbone coco | imagenet | ssl4eo | scratch   (ssl4eo needs `pip install torchgeo`)

Evaluation: centre-distance matching (10 px = 100 m). Reports AP, precision, recall, F1 and
the per-chip count error, because a COUNT is what the supply series needs. The best epoch
(by validation AP) is saved to runs/<tag>/best.pt; the test split (if any) is scored once
at the end with that checkpoint.

Quick smoke test on a laptop CPU:
    python s04_train_detector.py --sources ai2 --epochs 1 --max-train 64 --max-val 32 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from hormuz.common import P, get_logger, load_cfg, save_json, set_seed
from hormuz.detector import ChipDataset, build_model, collate, evaluate, list_chips

log = get_logger("s04")


def main():
    cfg = load_cfg()
    d = cfg["detector"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["ai2"], help="folders under data/labels/")
    ap.add_argument("--backbone", default=d["backbone"], choices=["coco", "imagenet", "ssl4eo", "scratch"])
    ap.add_argument("--init", default=None, help="checkpoint to start from (fine-tuning)")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch-size", type=int, default=d["batch_size"])
    ap.add_argument("--optimizer", default=d["optimizer"], choices=["sgd", "adamw"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=d.get("max_val_chips"),
                    help="validation chips scored each epoch (random fixed subset)")
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = P(cfg, "runs_dir") / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = P(cfg, "labels_dir")

    stats_file = labels_dir / "stats.json"
    stats = None
    if args.backbone == "ssl4eo":
        if not stats_file.exists():
            raise SystemExit("run `python s03_prepare_labels.py stats` first (ssl4eo needs channel stats)")
        s = json.loads(stats_file.read_text())
        stats = dict(mean=s["train_mean"], std=s["train_std"])

    init = None
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        args.backbone, stats = init["backbone"], init.get("stats")
        log.info("fine-tuning from %s (backbone=%s)", args.init, args.backbone)
    model = build_model(cfg, args.backbone, stats)
    if init is not None:
        model.load_state_dict(init["model"])
    model.to(device)

    train_pairs = list_chips(labels_dir, args.sources, "train")
    val_pairs = list_chips(labels_dir, args.sources, "val")
    test_pairs = list_chips(labels_dir, args.sources, "test")
    rng = np.random.default_rng(args.seed)
    if args.max_train and len(train_pairs) > args.max_train:
        train_pairs = [train_pairs[i] for i in rng.choice(len(train_pairs), args.max_train, replace=False)]
    if args.max_val and len(val_pairs) > args.max_val:
        val_pairs = [val_pairs[i] for i in rng.choice(len(val_pairs), args.max_val, replace=False)]
    log.info("chips: train=%d val=%d test=%d (sources %s) on %s", len(train_pairs), len(val_pairs),
             len(test_pairs), args.sources, device)
    if not train_pairs or not val_pairs:
        raise SystemExit("no chips found - run s03 / s03b first")

    nw = d["num_workers"]
    train_dl = DataLoader(ChipDataset(train_pairs, train=True, seed=args.seed), batch_size=args.batch_size,
                          shuffle=True, num_workers=nw, collate_fn=collate, drop_last=True,
                          persistent_workers=nw > 0)
    val_dl = DataLoader(ChipDataset(val_pairs, train=False), batch_size=args.batch_size,
                        num_workers=nw, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=d["weight_decay"])
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=d["weight_decay"])
    steps = args.epochs * len(train_dl)
    warm = min(500, max(1, steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(           # linear warm-up, then cosine decay
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for k, (images, targets) in enumerate(train_dl):
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{kk: v.to(device) for kk, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append({kk: v.item() for kk, v in loss_dict.items()} | {"total": loss.item()})
            if k % 50 == 0:
                log.info("epoch %d step %d/%d loss %.3f lr %.2e", epoch, k, len(train_dl), float(loss),
                         opt.param_groups[0]["lr"])
        m = evaluate(model, val_dl, device, d["match_distance_px"], d["score_threshold"])
        row = {"epoch": epoch, "seconds": round(time.time() - t0, 1)} | \
              {f"train_{kk}": float(np.mean([l[kk] for l in losses])) for kk in losses[0]} | \
              {f"val_{kk}": v for kk, v in m.items()}
        history.append(row)
        log.info("epoch %d  train loss %.3f  val AP %.3f  P %.3f  R %.3f  F1 %.3f  count MAE %.2f",
                 epoch, row["train_total"], m["ap"], m["precision"], m["recall"], m["f1"], m["count_mae"])
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        ck = dict(model=model.state_dict(), backbone=args.backbone, stats=stats, epoch=epoch,
                  sources=args.sources, val=m, args=vars(args))
        torch.save(ck, run_dir / "last.pt")
        if m["ap"] > best:
            best = m["ap"]
            torch.save(ck, run_dir / "best.pt")

    result = {"best_val_ap": best, "args": vars(args), "n_train": len(train_pairs), "n_val": len(val_pairs)}
    best_ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ck["model"])
    result["best_epoch"] = best_ck["epoch"]
    result["val"] = best_ck["val"]
    if test_pairs:
        test_dl = DataLoader(ChipDataset(test_pairs, train=False), batch_size=args.batch_size,
                             num_workers=nw, collate_fn=collate)
        result["test"] = evaluate(model, test_dl, device, d["match_distance_px"], d["score_threshold"])
        log.info("TEST (scored once): %s", {k: round(v, 3) if isinstance(v, float) else v
                                            for k, v in result["test"].items()})
    save_json(result, run_dir / "metrics.json")
    plot_history(pd.DataFrame(history), run_dir / "training_curves.png", args.tag)
    log.info("best val AP %.3f at epoch %d -> %s", best, result["best_epoch"], run_dir / "best.pt")


def plot_history(h: pd.DataFrame, file, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h["epoch"], h["train_total"], marker="o")
    ax[0].set(title=f"{tag}: training loss", xlabel="epoch", ylabel="loss")
    for k in ("val_ap", "val_f1", "val_precision", "val_recall"):
        ax[1].plot(h["epoch"], h[k], marker="o", label=k[4:])
    ax[1].set(title="validation", xlabel="epoch", ylim=(0, 1))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(file, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
