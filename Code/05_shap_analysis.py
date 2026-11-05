"""
05_shap_analysis.py  -  SHAP feature importance and feature selection (owner: SHAP analysis)

For each horizon h:
  1. Train (or load) the ALL-features LSTM for every seed in experiments.seeds
  2. Explain it with SHAP:
       explainer = "gradient" -> shap.GradientExplainer on the RNN itself (expected gradients)
       explainer = "tree"     -> LightGBM surrogate + TreeExplainer, the route used by
                                 Okasha & Zaher (2026) for marine gas oil
     Background samples come from TRAIN, explained samples from VALIDATION.
     The test period is never touched, so selection does not leak test information.
  3. Aggregate |SHAP| over samples and time lags -> one importance score per feature,
     averaged over seeds
  4. Select exogenous drivers (top_k or cumulative share); the target's own history
     is always kept

Outputs
  results/shap/importance_h{h}.csv           mean |SHAP| per feature (+ share, rank, per-seed)
  results/shap/lag_importance_h{h}.csv       mean |SHAP| per (lag, feature)
  results/shap/selected_features_h{h}.json   used by feature_set = "shap_selected"
  results/figures/shap_bar_h{h}.png, shap_lag_heatmap_h{h}.png, shap_beeswarm_h{h}.png

Usage
  python 05_shap_analysis.py
  python 05_shap_analysis.py --explainer tree

Note: cuDNN cannot back-propagate through an RNN in eval mode, so SHAP runs on CPU.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from oilfc.data import build_datasets
from oilfc.models import RNNForecaster
from oilfc.utils import import_script, load_config, path, run_name, save_json, set_seed

train_script = import_script("04_train_rnn.py")


class _TwoD(nn.Module):
    """SHAP expects outputs of shape (batch, n_outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


def load_model(cfg, stem) -> RNNForecaster:
    ckpt = torch.load(path(cfg, "results_dir", "models", f"{stem}.pt"), map_location="cpu", weights_only=False)
    m = ckpt["model_cfg"]
    model = RNNForecaster(len(ckpt["feature_names"]), m["type"], m["hidden_size"], m["num_layers"],
                          m["dropout"], m.get("bidirectional", False))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def gradient_shap(model, X_bg, X_ex) -> np.ndarray:
    """Return SHAP values with shape (n_explain, lookback, n_features)."""
    explainer = shap.GradientExplainer(_TwoD(model), torch.from_numpy(X_bg))
    sv = explainer.shap_values(torch.from_numpy(X_ex))
    sv = np.asarray(sv[0] if isinstance(sv, list) else sv)
    return sv[..., 0] if sv.ndim == 4 else sv


def tree_shap(X_tr, y_tr, X_ex) -> np.ndarray:
    """LightGBM surrogate on flattened windows; SHAP values reshaped to (n, lookback, n_features)."""
    import lightgbm as lgb

    n, L, F = X_tr.shape
    gbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, verbose=-1)
    gbm.fit(X_tr.reshape(n, -1), y_tr)
    sv = shap.TreeExplainer(gbm).shap_values(X_ex.reshape(len(X_ex), -1))
    return np.asarray(sv).reshape(len(X_ex), L, F)


def select_features(importance: pd.Series, target: str, cfg) -> list[str]:
    s = cfg["shap"]
    exo = importance.drop(index=target, errors="ignore").sort_values(ascending=False)
    if s["selection_rule"] == "top_k":
        return list(exo.index[: s["top_k"]])
    share = (exo / exo.sum()).cumsum()
    return list(share.index[: int(np.searchsorted(share.values, s["cumulative_threshold"]) + 1)])


def plots(cfg, h, importance, lag_imp, sv_sum, X_last, names):
    figs = path(cfg, "results_dir", "figures")

    fig, ax = plt.subplots(figsize=(6, 0.35 * len(names) + 1))
    importance.sort_values().plot.barh(ax=ax)
    ax.set_xlabel("mean |SHAP| (scaled target units)")
    ax.set_title(f"Feature importance, h={h}")
    fig.tight_layout(); fig.savefig(figs / f"shap_bar_h{h}.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(names) + 1))
    im = ax.imshow(lag_imp.T.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("time step in window (right = most recent)")
    ax.set_title(f"|SHAP| by lag and feature, h={h}")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(figs / f"shap_lag_heatmap_h{h}.png", dpi=150); plt.close(fig)

    shap.summary_plot(sv_sum, X_last, feature_names=names, show=False)
    plt.title(f"SHAP summed over lags, h={h}")
    plt.tight_layout(); plt.savefig(figs / f"shap_beeswarm_h{h}.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explainer", choices=["gradient", "tree"], default=None)
    args = ap.parse_args()
    cfg = load_config()
    explainer = args.explainer or cfg["shap"]["explainer"]
    target = cfg["data"]["target"]
    rng = np.random.default_rng(cfg["seed"])

    for h in cfg["framing"]["horizons"]:
        prep = build_datasets(cfg, "all", h)
        names = prep.feature_names
        tr, va = prep.splits["train"], prep.splits["val"]
        bg = tr.X[rng.choice(len(tr.X), min(cfg["shap"]["background_size"], len(tr.X)), replace=False)]
        ex_idx = rng.choice(len(va.X), min(cfg["shap"]["explain_size"], len(va.X)), replace=False)
        ex = va.X[ex_idx]

        per_seed, all_sv = {}, []
        seeds = cfg["experiments"]["seeds"] if explainer == "gradient" else [cfg["seed"]]
        for seed in seeds:
            set_seed(seed)
            if explainer == "gradient":
                train_script.run_single(cfg, "lstm", "all", h, seed)
                sv = gradient_shap(load_model(cfg, run_name("lstm", "all", h, seed)), bg, ex)
            else:
                sv = tree_shap(tr.X, tr.y, ex)
            all_sv.append(sv)
            per_seed[f"seed{seed}"] = np.abs(sv).mean(axis=(0, 1))
            print(f"h={h} seed={seed}: SHAP array {sv.shape}")

        sv = np.mean(all_sv, axis=0)
        importance = pd.Series(np.abs(sv).mean(axis=(0, 1)), index=names, name="mean_abs_shap")
        table = pd.DataFrame({"mean_abs_shap": importance,
                              "share": importance / importance.sum(),
                              **{k: v for k, v in per_seed.items()}})
        table["rank"] = table["mean_abs_shap"].rank(ascending=False).astype(int)
        table.sort_values("rank").to_csv(path(cfg, "results_dir", "shap", f"importance_h{h}.csv"))

        lag_imp = pd.DataFrame(np.abs(sv).mean(axis=0), columns=names)
        lag_imp.index = [f"t-{prep.lookback - 1 - i}" for i in range(prep.lookback)]
        lag_imp.to_csv(path(cfg, "results_dir", "shap", f"lag_importance_h{h}.csv"))

        selected = select_features(importance, target, cfg)
        save_json({"horizon": h, "explainer": explainer, "rule": cfg["shap"]["selection_rule"],
                   "selected": selected, "always_included": target,
                   "ranking": table.sort_values("rank").index.tolist()},
                  path(cfg, "results_dir", "shap", f"selected_features_h{h}.json"))
        print(f"h={h}: selected {selected}\n{table.sort_values('rank').round(4).to_string()}\n")

        # Beeswarm: SHAP summed over lags vs. the feature's most recent (scaled) value
        plots(cfg, h, importance, lag_imp, sv.sum(axis=1), ex[:, -1, :], names)

    # TODO (SHAP owner):
    #   * rank stability across seeds / horizons (Kendall tau between per-seed rankings)
    #   * compare gradient vs. tree explainer rankings
    #   * SHAP over time: does importance of e.g. the dollar change in 2020-2022?


if __name__ == "__main__":
    main()
