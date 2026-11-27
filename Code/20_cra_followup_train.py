"""
20_cra_followup_train.py  -  Train the Chronos-feature LSTM (issue #10 follow-up)

Same refits, seeds, samples and loss mask as 17_cra_train.py, with two changes:
the model (oilfc/cra_followup_model.py) and point-in-time GPR (oilfc/cra_gpr.py).
Each run also predicts with the Chronos block masked, to measure how much the
model relies on it.

Outputs (stem = e.g. cfl_h5_y2023_s0)
  results/cra/followup/predictions/{stem}.csv   incl. pred_price_noprior
  results/cra/followup/runs/{stem}.json
  results/cra/followup/models/{stem}.pt         (git-ignored)

Usage
  python 20_cra_followup_train.py
  python 20_cra_followup_train.py --horizons 5 --force
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json

import numpy as np
import torch

from oilfc.cra_data import build_refit, load_prior
from oilfc.cra_followup_model import build_followup_model
from oilfc.cra_gpr import load_followup_data
from oilfc.cra_model import predict_cra, train_cra
from oilfc.metrics import all_metrics
from oilfc.models import count_parameters
from oilfc.utils import CODE_DIR, get_device, import_script, load_config, load_json, path, save_json, set_seed

frame = import_script("17_cra_train.py").frame


def config_hash(cfg: dict, fu: dict) -> str:
    keys = {k: cfg[k] for k in ("framing", "evaluation", "data", "chronos")}
    keys.update({k: fu[k] for k in ("model", "training", "gpr")})
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def masked(split):
    return dataclasses.replace(split, prior=np.zeros_like(split.prior))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if a matching run exists")
    args = ap.parse_args()

    cfg = load_config(CODE_DIR / "cra_config.yaml")
    fu = load_config(CODE_DIR / "cra_followup.yaml")
    run_cfg = {**cfg, "paths": {**cfg["paths"], **fu["paths"]}, "model": fu["model"], "training": fu["training"]}
    ev = cfg["evaluation"]
    device = get_device()
    lv, feats = load_followup_data(cfg, fu)
    prior = load_prior(cfg)
    chash = config_hash(cfg, fu)
    for sub in ("predictions", "runs", "models"):
        path(run_cfg, "results_dir", sub).mkdir(parents=True, exist_ok=True)

    for h in args.horizons or cfg["framing"]["horizons"]:
        for year in ev["test_years"] + ev["holdout_years"]:
            refit = build_refit(cfg, lv, feats, prior, h, year)
            tr, va, te = (refit.splits[k] for k in ("train", "val", "eval"))
            n = {k: len(s.z) for k, s in refit.splits.items()}
            print(f"h={h} year={year}: train {n['train']} val {n['val']} eval {n['eval']}")
            for seed in ev["seeds"]:
                stem = f"cfl_h{h}_y{year}_s{seed}"
                summary_file = path(run_cfg, "results_dir", "runs", f"{stem}.json")
                if summary_file.exists() and not args.force:
                    old = load_json(summary_file)
                    if old.get("config_hash") == chash and old.get("status") == "complete":
                        continue
                set_seed(seed)
                model = build_followup_model(fu, n_features=len(refit.features))
                model, hist = train_cra(model, tr, va, run_cfg, device, seed)
                q_ev, g_ev, w_ev = predict_cra(model, te, device)
                q_va, g_va, _ = predict_cra(model, va, device)
                q_ev0, _, _ = predict_cra(model, masked(te), device)
                q_va0, _, _ = predict_cra(model, masked(va), device)
                df_ev = frame(te, q_ev[:, 1], q_ev, g_ev)
                df_ev["pred_price_noprior"] = te.base_price * np.exp(q_ev0[:, 1])
                df_va, df_va0 = frame(va, q_va[:, 1]), frame(va, q_va0[:, 1])
                df_ev.to_csv(path(run_cfg, "results_dir", "predictions", f"{stem}.csv"), index=False)
                torch.save(model.state_dict(), path(run_cfg, "results_dir", "models", f"{stem}.pt"))
                summary = {"run": stem, "model": "cfl", "horizon": h, "year": year, "seed": seed,
                           "config_hash": chash, "status": "complete", "n": n,
                           "n_params": count_parameters(model), "history": hist,
                           "val": all_metrics(df_va.true_price, df_va.pred_price, df_va.base_price),
                           "val_noprior": all_metrics(df_va0.true_price, df_va0.pred_price, df_va0.base_price),
                           "eval": all_metrics(df_ev.true_price, df_ev.pred_price, df_ev.base_price),
                           "gate_eval_mean": float(g_ev.mean()), "gate_val_mean": float(g_va.mean()),
                           "vsn_weights": dict(zip(refit.features, map(float, w_ev)))}
                save_json(summary, summary_file)
                print(f"   {stem}: epochs {hist['epochs_run']:3d}  val RMSE {summary['val']['RMSE']:.3f} "
                      f"(masked {summary['val_noprior']['RMSE']:.3f})  eval RMSE {summary['eval']['RMSE']:.3f}  "
                      f"gate {summary['gate_eval_mean']:.2f}  ({hist['train_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
