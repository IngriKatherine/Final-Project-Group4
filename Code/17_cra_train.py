"""
17_cra_train.py  -  Train the original LSTM and the gated residual LSTM under one protocol (issue #10)

For every horizon, evaluation year (test_years + holdout_years) and seed:
  * lstm : the original price-only LSTM (config.yaml model + training, oilfc.train.train_model)
  * cra  : the Chronos-anchored gated residual LSTM (cra_config.yaml)
Both are refit each year on identical samples, scalers and loss mask
(oilfc/cra_data.build_refit), so differences come from the model only.

Outputs (stem = e.g. cra_h5_y2023_s0)
  results/cra/predictions/{stem}.csv
  results/cra/runs/{stem}.json
  results/cra/models/{stem}.pt          (git-ignored)

Usage
  python 17_cra_train.py
  python 17_cra_train.py --models cra --horizons 5 --force
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from oilfc.cra_data import build_refit, load_cra_data, load_prior
from oilfc.cra_model import build_cra_model, predict_cra, train_cra
from oilfc.metrics import all_metrics
from oilfc.models import build_model, count_parameters
from oilfc.train import predict, train_model
from oilfc.utils import CODE_DIR, get_device, load_config, load_json, override, path, save_json, set_seed


def config_hash(cfg: dict, base: dict, model: str) -> str:
    keys = {"framing": cfg["framing"], "evaluation": cfg["evaluation"], "data": cfg["data"]}
    keys.update({"model": base["model"], "training": base["training"]} if model == "lstm" else
                {"model": cfg["model"], "training": cfg["training"], "chronos": cfg["chronos"]})
    return hashlib.sha1(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12]


def frame(split, pred_r, q=None, gate=None) -> pd.DataFrame:
    df = pd.DataFrame({"origin_date": split.origin_dates, "target_date": split.target_dates,
                       "base_price": split.base_price, "true_price": split.true_price,
                       "pred_price": split.base_price * np.exp(pred_r)})
    if q is not None:
        df["q10_price"] = split.base_price * np.exp(q[:, 0])
        df["q90_price"] = split.base_price * np.exp(q[:, 2])
        df["gate"] = gate
    return df


def run_lstm(base: dict, refit, seed: int, device):
    """Original LSTM: price-only input, standardized log-return target, MSE, Adam + plateau."""
    cfg = override(base, **{"model.type": "lstm"})
    j = refit.features.index("r_wti")
    tr, va, ev = (refit.splits[k] for k in ("train", "val", "eval"))
    mu, sd = tr.y_raw.mean(), tr.y_raw.std()
    loader = lambda s, shuffle: DataLoader(
        TensorDataset(torch.from_numpy(np.ascontiguousarray(s.X[:, :, [j]])),
                      torch.from_numpy(((s.y_raw - mu) / sd).astype(np.float32))),
        batch_size=cfg["training"]["batch_size"], shuffle=shuffle)
    model = build_model(cfg, n_features=1)
    model, hist = train_model(model, loader(tr, True), loader(va, False), cfg, device, verbose=False)
    pr = lambda s: predict(model, np.ascontiguousarray(s.X[:, :, [j]]), device) * sd + mu
    return model, hist, frame(ev, pr(ev)), frame(va, pr(va)), {}


def run_cra(cfg: dict, refit, seed: int, device):
    tr, va, ev = (refit.splits[k] for k in ("train", "val", "eval"))
    model = build_cra_model(cfg, n_features=len(refit.features))
    model, hist = train_cra(model, tr, va, cfg, device, seed)
    q_ev, g_ev, w_ev = predict_cra(model, ev, device)
    q_va, g_va, _ = predict_cra(model, va, device)
    extra = {"gate_eval_mean": float(g_ev.mean()), "gate_eval_p90": float(np.quantile(g_ev, 0.9)),
             "vsn_weights": dict(zip(refit.features, map(float, w_ev)))}
    return model, hist, frame(ev, q_ev[:, 1], q_ev, g_ev), frame(va, q_va[:, 1], q_va, g_va), extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm", "cra"], choices=["lstm", "cra"])
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="retrain even if a matching run exists")
    args = ap.parse_args()

    cfg = load_config(CODE_DIR / "cra_config.yaml")
    base = load_config()
    ev = cfg["evaluation"]
    years = ev["test_years"] + ev["holdout_years"]
    device = get_device()
    lv, feats = load_cra_data(cfg)
    prior = load_prior(cfg)
    for sub in ("predictions", "runs", "models"):
        path(cfg, "results_dir", sub).mkdir(parents=True, exist_ok=True)

    for h in args.horizons or cfg["framing"]["horizons"]:
        for year in years:
            refit = build_refit(cfg, lv, feats, prior, h, year)
            n = {k: len(s.z) for k, s in refit.splits.items()}
            print(f"h={h} year={year}: train {n['train']} val {n['val']} eval {n['eval']} "
                  f"({refit.splits['eval'].origin_dates[0].date()} .. {refit.splits['eval'].origin_dates[-1].date()})")
            for model_name in args.models:
                chash = config_hash(cfg, base, model_name)
                for seed in ev["seeds"]:
                    stem = f"{model_name}_h{h}_y{year}_s{seed}"
                    summary_file = path(cfg, "results_dir", "runs", f"{stem}.json")
                    if summary_file.exists() and not args.force:
                        old = load_json(summary_file)
                        if old.get("config_hash") == chash and old.get("status") == "complete":
                            continue
                    set_seed(seed)
                    runner = run_lstm if model_name == "lstm" else run_cra
                    model, hist, df_ev, df_va, extra = runner(base if model_name == "lstm" else cfg,
                                                              refit, seed, device)
                    df_ev.to_csv(path(cfg, "results_dir", "predictions", f"{stem}.csv"), index=False)
                    torch.save(model.state_dict(), path(cfg, "results_dir", "models", f"{stem}.pt"))
                    summary = {"run": stem, "model": model_name, "horizon": h, "year": year, "seed": seed,
                               "config_hash": chash, "status": "complete", "n": n,
                               "n_params": count_parameters(model), "history": hist,
                               "val": all_metrics(df_va.true_price, df_va.pred_price, df_va.base_price),
                               "eval": all_metrics(df_ev.true_price, df_ev.pred_price, df_ev.base_price),
                               **extra}
                    save_json(summary, summary_file)
                    print(f"   {stem}: epochs {hist['epochs_run']:3d}  val RMSE {summary['val']['RMSE']:.3f}  "
                          f"eval RMSE {summary['eval']['RMSE']:.3f}  ({hist['train_seconds']:.0f} s)")


if __name__ == "__main__":
    main()
