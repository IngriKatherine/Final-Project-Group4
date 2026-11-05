"""
04_train_rnn.py  -  Train one LSTM or GRU forecaster (owner: LSTM/GRU modeling)

One run = one (model, feature set, horizon, seed). Other scripts import
`run_single` from here to loop over many runs.

Outputs (stem = e.g. lstm_all_h5_s0)
  results/models/{stem}.pt                    weights + feature names + scaler stats
  results/predictions/{stem}.csv              test-period price forecasts
  results/tables/runs/{stem}.json             metrics, history, settings

Usage
  python 04_train_rnn.py --model lstm --features all --horizon 1 --seed 0
  python 04_train_rnn.py --model gru  --features price_only --horizon 20
  python 04_train_rnn.py --model lstm --features shap_selected --horizon 5   # after 05_shap_analysis.py
"""
from __future__ import annotations

import argparse

import torch

from oilfc.data import build_datasets, predictions_frame, to_price, torch_loader
from oilfc.metrics import all_metrics
from oilfc.models import build_model, count_parameters
from oilfc.train import predict, train_model
from oilfc.utils import get_device, load_config, load_json, override, path, run_name, save_json, set_seed


def run_single(cfg: dict, model_type: str, feature_set, horizon: int, seed: int,
               tag: str = "", force: bool = False, verbose: bool = True) -> dict:
    fs_name = feature_set if isinstance(feature_set, str) else "custom"
    stem = run_name(model_type, fs_name, horizon, seed, tag)
    summary_file = path(cfg, "results_dir", "tables", "runs") / f"{stem}.json"
    if summary_file.exists() and not force:
        if verbose:
            print(f"[skip] {stem} already done")
        return load_json(summary_file)

    cfg = override(cfg, **{"model.type": model_type})
    set_seed(seed)
    device = get_device()
    prep = build_datasets(cfg, feature_set, horizon)
    tr, va, te = prep.splits["train"], prep.splits["val"], prep.splits["test"]
    if verbose:
        print(f"\n=== {stem} | features={prep.feature_names} | "
              f"train {len(tr.y)}  val {len(va.y)}  test {len(te.y)} | device {device}")

    t = cfg["training"]
    model = build_model(cfg, n_features=len(prep.feature_names))
    model, history = train_model(
        model,
        torch_loader(tr, t["batch_size"], shuffle=True, num_workers=t["num_workers"]),
        torch_loader(va, t["batch_size"], shuffle=False),
        cfg, device, verbose=verbose,
    )

    # Test-set forecasts in price space
    tf = cfg["framing"]["target_transform"]
    pred_price = to_price(prep.inverse_y(predict(model, te.X, device)), te.base_price, tf)
    preds = predictions_frame(te, pred_price)
    preds.to_csv(path(cfg, "results_dir", "predictions", f"{stem}.csv"), index=False)
    val_price = to_price(prep.inverse_y(predict(model, va.X, device)), va.base_price, tf)

    torch.save({
        "state_dict": model.state_dict(),
        "model_cfg": cfg["model"],
        "feature_names": prep.feature_names,
        "lookback": prep.lookback,
        "horizon": horizon,
        "x_mean": prep.x_scaler.mean_, "x_scale": prep.x_scaler.scale_,
        "y_mean": prep.y_scaler.mean_, "y_scale": prep.y_scaler.scale_,
    }, path(cfg, "results_dir", "models", f"{stem}.pt"))

    summary = {
        "run": stem, "model": model_type, "features": fs_name, "horizon": horizon, "seed": seed,
        "tag": tag, "feature_names": prep.feature_names, "n_params": count_parameters(model),
        "lookback": prep.lookback, "model_cfg": cfg["model"], "training_cfg": t,
        "val": all_metrics(va.true_price, val_price, va.base_price),
        "test": all_metrics(te.true_price, pred_price, te.base_price),
        "history": history,
    }
    save_json(summary, summary_file)
    if verbose:
        print(f"    val  {summary['val']}\n    test {summary['test']}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru"])
    ap.add_argument("--features", default="all", help="price_only | all | shap_selected")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="retrain even if results exist")
    args = ap.parse_args()
    run_single(load_config(), args.model, args.features, args.horizon, args.seed, force=args.force)


if __name__ == "__main__":
    main()
