"""End-to-end smoke test of Code/satellite on a synthetic Strait of Hormuz.

Copies the code to a temporary folder, builds a fake world there (tests/synthetic_world.py),
and runs every step exactly as the README does: s02 -> s03 -> s04 -> s03b -> s04 fine-tune
-> s05 -> s06 -> s07. Your real data/, runs/ and config are never touched.

Modes
    full   (default when a CUDA GPU is present) COCO-pretrained detector, enough epochs to
           learn the synthetic ships. Also checks that the CNN really learned (val AP >= 0.5)
           and that the supply series drops after 28 Feb 2026. Needs internet for the COCO
           weights (~170 MB, cached afterwards). ~3-5 min on an A10G.
    quick  (CPU) untrained-from-scratch detector, 1 short epoch: checks that the code runs,
           not that it learns. ~10-15 min on a laptop CPU.

Run (from Code/satellite)
    python tests/run_smoke_test.py               # picks the mode
    python tests/run_smoke_test.py --mode quick --keep
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SAT = HERE.parent
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, msg: str = ""):
    checks.append((name, bool(ok), msg))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {msg}", flush=True)


def run(step: str, args: list[str], cwd: Path, timeout: int = 3600) -> bool:
    t0 = time.time()
    print(f"\n=== {step}: python {' '.join(args)}", flush=True)
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=timeout)
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-6:])
    print(tail)
    check(step, p.returncode == 0, f"({time.time() - t0:.0f}s)" + ("" if p.returncode == 0 else f" exit {p.returncode}"))
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-3000:])
    return p.returncode == 0


def fake_portwatch(file: Path):
    d = pd.date_range("2023-01-01", "2026-11-20", freq="D")
    r = np.random.default_rng(0)
    n = np.where(d >= "2026-02-28", r.poisson(3, len(d)), r.poisson(40, len(d)))
    pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "portname": "Strait of Hormuz", "n_tanker": n,
                  "capacity_tanker": n * 48000.0, "n_total": 2 * n}).to_csv(file, index=False)


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "quick"], default="full" if torch.cuda.is_available() else "quick")
    ap.add_argument("--workdir", default=None, help="where to build the test copy (default: a temp folder)")
    ap.add_argument("--keep", action="store_true", help="keep the test folder to inspect figures")
    a = ap.parse_args()

    base = Path(a.workdir) if a.workdir else Path(tempfile.mkdtemp(prefix="hormuz_smoke_"))
    root = base / "satellite"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(SAT, root, ignore=shutil.ignore_patterns("data", "runs", "figures", "__pycache__", "*.pt"))
    print(f"mode={a.mode}  test copy: {root}")
    full = a.mode == "full"

    ok = run("synthetic world", [str(root / "tests" / "synthetic_world.py"), str(root)], root)
    ok = ok and run("s02 download (fake tiles)", ["s02_download_scenes.py", "--workers", "2"], root)
    if ok:
        idx = pd.read_csv(root / "data/scenes/scene_index.csv")
        check("s02 8 dates mosaicked", (idx["status"] == "ok").sum() == 8, f"({(idx['status'] == 'ok').sum()})")
        check("s02 cloud found on 2026-01-12", idx.set_index("date").loc["2026-01-12", "cloud_frac"] > 0.02)
    ok = ok and run("s03 ai2 convert", ["s03_prepare_labels.py", "ai2", "--workers", "2"], root)
    ok = ok and run("s03 preview", ["s03_prepare_labels.py", "preview"], root)
    ok = ok and run("s03 stats", ["s03_prepare_labels.py", "stats"], root)

    train = ["s04_train_detector.py", "--sources", "ai2", "--tag", "ai2_test"]
    train += ["--backbone", "coco", "--epochs", "10", "--lr", "0.01"] if full else \
             ["--backbone", "scratch", "--epochs", "1", "--max-train", "8", "--max-val", "8",
              "--optimizer", "adamw", "--lr", "3e-4"]
    ok = ok and run("s04 train on Ai2-format chips", train, root, timeout=7200)
    if ok:
        m = json.loads((root / "runs/ai2_test/metrics.json").read_text())
        check("s04 wrote best.pt + metrics", (root / "runs/ai2_test/best.pt").exists(), f"val AP {m['val']['ap']:.2f}")
        if full:
            check("s04 CNN learned the synthetic ships (val AP >= 0.5)", m["val"]["ap"] >= 0.5, f"AP {m['val']['ap']:.2f}")

    ok = ok and run("s03b export Hormuz chips (model pre-annotations)",
                    ["s03b_hormuz_chips.py", "export", "--n", "24", "--max-dates", "8", "--ckpt", "runs/ai2_test/best.pt"], root)
    ok = ok and run("s03b import (split by date)", ["s03b_hormuz_chips.py", "import"], root)
    ok = ok and run("s03 preview incl. Hormuz chips", ["s03_prepare_labels.py", "preview"], root)
    if ok:
        man = pd.read_csv(root / "data/hormuz_to_label/manifest.csv")
        check("s03b chips cover both periods", set(man["period"]) == {"before", "after"}, f"({len(man)} chips)")
    ft = ["s04_train_detector.py", "--sources", "hormuz", "--init", "runs/ai2_test/best.pt", "--tag", "hz_test",
          "--batch-size", "2"]
    ft += ["--epochs", "6", "--lr", "0.003"] if full else ["--epochs", "1", "--max-train", "4", "--lr", "0.001"]
    ok = ok and run("s04 fine-tune on Hormuz chips", ft, root, timeout=7200)
    if ok:
        m = json.loads((root / "runs/hz_test/metrics.json").read_text())
        check("s04 scored the date-held-out test split", "test" in m,
              f"test AP {m.get('test', {}).get('ap', float('nan')):.2f}")

    ok = ok and run("s05 detect all dates", ["s05_detect_scenes.py", "--ckpt", "runs/hz_test/best.pt", "--quicklook"],
                    root, timeout=7200)
    if ok:
        cov = pd.read_csv(root / "data/detections/coverage.csv")
        check("s05 coverage rows (8 dates x 3 zones)", len(cov) == 24, f"({len(cov)})")
        det_file = root / "data/detections/detections.csv"
        n = len(pd.read_csv(det_file)) if det_file.exists() else 0
        if full:
            check("s05 found ships", n > 20, f"({n} detections)")
        ok = run("s05 resume skips finished dates", ["s05_detect_scenes.py", "--ckpt", "runs/hz_test/best.pt"], root)
    ok = ok and run("s06 supply series", ["s06_build_supply_series.py"], root)
    if ok and full:
        s = pd.read_csv(root / "data/series/supply_by_scene.csv", parse_dates=["date"])
        pre = s[s["date"] < "2026-02-28"]["transit_n_large"].mean()
        post = s[s["date"] >= "2026-02-28"]["transit_n_large"].mean()
        check("s06 supply falls after 28 Feb 2026", pre > post, f"(large ships in transit: {pre:.1f} -> {post:.1f})")
    fake_portwatch(root / "data" / "portwatch_fake.csv")
    (base / "data" / "processed").mkdir(parents=True, exist_ok=True)
    ok = ok and run("s07 calibrate + export", ["s07_calibrate_and_export.py", "--portwatch-csv",
                                                "data/portwatch_fake.csv", "--ais-lag", "3"], root)
    if ok:
        f = base / "data/processed/satellite_features.csv"
        feats = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame()
        cols = {"HZ_SAT_FLOW_MBD", "HZ_SAT_ANCHOR_MBBL", "HZ_SAT_LARGE_N", "HZ_AIS_FLOW_MBD", "HZ_OBS_AGE_D"}
        check("s07 features file has all columns", cols <= set(feats.columns), f"({len(feats)} business days)")
        for fig in ("supply_series.png", "calibration_scatter.png", "satellite_vs_ais_flow.png",
                    "labels_preview_ai2.png", "labels_preview_hormuz.png"):
            check(f"figure {fig}", (root / "figures" / fig).exists())

    n_fail = sum(not c[1] for c in checks)
    print(f"\n{len(checks)} checks: {len(checks) - n_fail} pass, {n_fail} fail   (mode {a.mode})")
    if a.keep or n_fail:
        print(f"test folder kept: {root}  (figures/ has the previews and series)")
    else:
        shutil.rmtree(base, ignore_errors=True)
    raise SystemExit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
