"""
09_image_cnn.py  -  Image-based CNN direction forecasts (owner: image CNN)

Method: Ren, Jiang, Ji & Zhai (2024), "Seeing is believing: Forecasting crude oil
price trend from the perspective of images", Journal of Forecasting 43(7), 2809-2821.

  1. Turn every 20-day window of WTI futures OHLCV into a 32 x 60 black-and-white
     chart; volume is the brightness of each day's bars (oilfc/imaging.py)
  2. Label each image "up" if P_{t+h} >= P_t, using the SAME target series and
     business-day calendar as the RNNs (data/processed/levels.csv), so the CNN is
     scored on exactly the days and prices the other models are
  3. CNN of Figure 2: [conv 3x3, 32 -> ReLU -> maxpool 2 -> BN] -> [conv 3x3, 64 ->
     ReLU -> maxpool 2 -> BN] -> dropout 0.25 -> one sigmoid unit (the "FC 4993" box in
     the paper is that unit's parameter count: 6*13*64 weights + 1 bias), cross-entropy,
     Adam, early stopping with patience 3 on validation loss
  4. Average P(up) over seeds; compare its direction accuracy with the sign of every
     price forecast already produced (random walk, ARIMA, ARIMAX, LSTM/GRU per feature set)

Two training designs (image_cnn.retrain):
  fixed   the project's split: train / val / test by TARGET date from config.yaml split.
          Default, because it makes the CNN directly comparable with the RNNs.
  yearly  the paper's design: expanding window, refit once per test year. We use a
          chronological validation tail (last val_frac of the window, with an h-day purge)
          instead of the paper's random 70/30 split: with overlapping 20-day windows a
          random split puts near-identical images in train and validation, so early
          stopping would be tuned on leaked data.

Outputs (stem = image_cnn_{style}_h{h}, plus "_yearly" in yearly mode)
  results/predictions/{stem}_s{seed}.csv     per-seed P(up) on the test set
  results/predictions/{stem}_avg.csv         seed-averaged P(up)
  results/models/{stem}_s{seed}.pt           weights (fixed mode)
  results/tables/image_cnn/{stem}_s{seed}.json   history, val and test metrics
  results/tables/direction_comparison.csv / .md  all models, same test days
  results/figures/image_cnn_examples.png         the three image styles on one window
  results/figures/image_cnn_attr_h{h}.png        with --explain: where in the chart SHAP mass sits

Usage
  python 01_fetch_data.py                           # once, after adding data.ohlcv_tickers
  python 09_image_cnn.py                            # all horizons and seeds from config.yaml
  python 09_image_cnn.py --style no_volume          # image ablation (Figure 1b)
  python 09_image_cnn.py --style jiang              # Jiang, Kelly & Xiu volume panel (Figure 1c)
  python 09_image_cnn.py --retrain yearly           # the paper's expanding-window design
  python 09_image_cnn.py --horizons 1 --explain     # + SHAP attribution over the chart
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from oilfc.data import load_levels
from oilfc.direction import direction_metrics, sign_forecast
from oilfc.imaging import STYLES, clean_ohlcv, make_images
from oilfc.utils import get_device, load_config, path, save_json, set_seed

DEFAULTS = {
    "source": "WTI",            # reads data/raw/ohlcv_{source}.csv
    "window": 20,
    "height": 32,
    "style": "volume_intensity",
    "min_intensity": 0.1,
    "max_stale_days": 5,        # image may end at most this many calendar days before the origin
    "retrain": "fixed",
    "val_frac": 0.3,            # yearly mode
    "lr": 1e-3,                 # not reported in the paper
    "weight_decay": 0.0,        # not in the paper; raise if the CNN memorises noise
    "batch_size": 128,
    "max_epochs": 50,
    "patience": 3,              # paper; at ~6,000 images it can stop before any signal is learned
    "dropout": 0.25,
    "channels": [32, 64],
    "big_move": 0.05,           # paper Table 2: |return| > 5%
    "drop_ties": True,          # drop targets with zero price change: in levels.csv these are
                                # holidays forward-filled by 02_preprocess.py, not real moves,
                                # and counting them as "up" hands free hits to up-biased models
    "special_periods": {"covid_2020": ["2020-03-01", "2020-06-30"],
                        "russia_ukraine_2022": ["2022-02-01", "2022-04-30"]},
    "explain_background": 200,
    "explain_size": 300,
}


def settings(cfg: dict, args) -> dict:
    s = {**DEFAULTS, **cfg.get("image_cnn", {})}
    for k in ("style", "retrain", "source"):
        if getattr(args, k) is not None:
            s[k] = getattr(args, k)
    s["horizons"] = args.horizons or s.get("horizons") or cfg["framing"]["horizons"]
    s["seeds"] = args.seeds or s.get("seeds") or cfg["experiments"]["seeds"]
    if s["style"] not in STYLES:
        raise ValueError(f"image_cnn.style must be one of {STYLES}")
    return s


def out_file(cfg: dict, *parts) -> Path:
    f = Path(path(cfg, "results_dir", *parts))
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


# ----------------------------------------------------------------------------- model
class ImageCNN(nn.Module):
    """Figure 2 of Ren et al. (2024). Returns logits; the sigmoid lives in the loss."""

    def __init__(self, height: int = 32, width: int = 60, channels=(32, 64), dropout: float = 0.25):
        super().__init__()
        layers, c_in = [], 1
        for c_out in channels:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(c_out)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, height, width)).numel()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Flatten(), nn.Linear(n_flat, 1))

    def forward(self, x):                       # x: (batch, 1, H, W), floats in [0, 1]
        return self.head(self.features(x)).squeeze(-1)


def _as_input(xb: torch.Tensor, device) -> torch.Tensor:
    return xb.to(device).float().div_(255.0).unsqueeze(1)


def train_cnn(X_tr, y_tr, X_va, y_va, s: dict, seed: int, device, verbose: bool = True):
    set_seed(seed)
    _, H, W = X_tr.shape
    model = ImageCNN(H, W, s["channels"], s["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))),
                        batch_size=s["batch_size"], shuffle=True, generator=g)
    Xv, yv = torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32))

    best, best_state, bad = np.inf, None, 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(s["max_epochs"]):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(_as_input(xb, device)), yb.to(device))
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb); n += len(yb)
        logits = _logits(model, Xv, device)
        vl = float(loss_fn(logits, yv).item())
        va = float(((logits >= 0).float() == yv).float().mean())
        hist["train_loss"].append(tot / n); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        if verbose:
            print(f"      epoch {epoch + 1:>3}  train {tot / n:.4f}  val {vl:.4f}  val acc {va:.3f}")
        if vl < best - 1e-5:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= s["patience"]:
                break
    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["val_loss"])
    hist["best_epoch"] = int(np.argmin(hist["val_loss"])) + 1
    return model.eval(), hist


@torch.no_grad()
def _logits(model, X: torch.Tensor, device, bs: int = 2048) -> torch.Tensor:
    model.eval()
    return torch.cat([model(_as_input(X[i:i + bs], device)).cpu() for i in range(0, len(X), bs)])


def predict_proba(model, X: np.ndarray, device) -> np.ndarray:
    return torch.sigmoid(_logits(model, torch.from_numpy(X), device)).numpy()


# ----------------------------------------------------------------------------- data
def load_images(cfg: dict, s: dict) -> tuple[np.ndarray, pd.DatetimeIndex]:
    f = path(cfg, "raw_dir", f"ohlcv_{s['source']}.csv")
    if not Path(f).exists():
        raise FileNotFoundError(f"{f} not found. Add data.ohlcv_tickers to config.yaml and rerun 01_fetch_data.py.")
    ohlcv = clean_ohlcv(pd.read_csv(f, index_col=0, parse_dates=True))
    imgs, dates = make_images(ohlcv, s["window"], s["height"], s["style"], s["min_intensity"])
    print(f"{len(imgs)} images {imgs.shape[1:]} ({s['style']}) from {dates[0].date()} to {dates[-1].date()}")
    return imgs, dates


def build_samples(cfg: dict, levels: pd.DataFrame, img_dates: pd.DatetimeIndex, h: int, s: dict) -> pd.DataFrame:
    """One row per forecast origin t on the levels calendar, with the image ending at or before t."""
    P = levels[cfg["data"]["target"]]
    df = pd.DataFrame({"origin_date": P.index[:-h], "target_date": P.index[h:],
                       "base_price": P.values[:-h], "true_price": P.values[h:]})
    imgs = pd.DataFrame({"img_date": img_dates, "img_idx": np.arange(len(img_dates))})
    df = pd.merge_asof(df, imgs, left_on="origin_date", right_on="img_date", direction="backward",
                       tolerance=pd.Timedelta(days=s["max_stale_days"])).dropna(subset=["img_idx"])
    df["img_idx"] = df["img_idx"].astype(int)
    if s["drop_ties"]:
        df = df[df["true_price"] != df["base_price"]]
    df["true_up"] = (df["true_price"] >= df["base_price"]).astype(int)
    df["ret"] = np.where(df["base_price"] > 0, df["true_price"] / df["base_price"] - 1, np.nan)
    tr_end, va_end = pd.Timestamp(cfg["split"]["train_end"]), pd.Timestamp(cfg["split"]["val_end"])
    df["split"] = np.select([df.target_date <= tr_end, df.target_date <= va_end], ["train", "val"], "test")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------- one seed
def run_seed(cfg, s, imgs, samples, h, seed, device, stem, force=False) -> pd.DataFrame:
    pred_file = out_file(cfg, "predictions", f"{stem}_s{seed}.csv")
    summ_file = out_file(cfg, "tables", "image_cnn", f"{stem}_s{seed}.json")
    if pred_file.exists() and summ_file.exists() and not force:
        print(f"   [skip] {stem}_s{seed} already done")
        return pd.read_csv(pred_file, parse_dates=["origin_date", "target_date"])

    test = samples[samples.split == "test"].copy()
    if s["retrain"] == "fixed":
        tr, va = samples[samples.split == "train"], samples[samples.split == "val"]
        print(f"   seed {seed}: train {len(tr)}  val {len(va)}  test {len(test)}")
        model, hist = train_cnn(imgs[tr.img_idx], tr.true_up.values, imgs[va.img_idx], va.true_up.values,
                                s, seed, device)
        test["prob_up"] = predict_proba(model, imgs[test.img_idx], device)
        val_m = direction_metrics(va.true_up, predict_proba(model, imgs[va.img_idx], device) >= 0.5)
        torch.save({"state_dict": model.state_dict(), "settings": s, "horizon": h},
                   out_file(cfg, "models", f"{stem}_s{seed}.pt"))
        histories = {"fixed": hist}
    else:  # yearly expanding window, as in the paper
        histories, val_m = {}, {}
        for year in sorted(test.target_date.dt.year.unique()):
            pool = samples[samples.target_date < pd.Timestamp(f"{year}-01-01")]
            n_va = int(len(pool) * s["val_frac"])
            tr, va = pool.iloc[: len(pool) - n_va - h], pool.iloc[len(pool) - n_va:]   # h-step purge
            print(f"   seed {seed}, test year {year}: train {len(tr)}  val {len(va)}")
            model, hist = train_cnn(imgs[tr.img_idx], tr.true_up.values, imgs[va.img_idx], va.true_up.values,
                                    s, seed, device, verbose=False)
            m = test.target_date.dt.year == year
            test.loc[m, "prob_up"] = predict_proba(model, imgs[test.loc[m, "img_idx"]], device)
            histories[str(year)] = hist
            val_m[str(year)] = direction_metrics(va.true_up, predict_proba(model, imgs[va.img_idx], device) >= 0.5)

    test["pred_up"] = (test.prob_up >= 0.5).astype(int)
    cols = ["origin_date", "target_date", "base_price", "true_price", "true_up", "prob_up", "pred_up"]
    test[cols].to_csv(pred_file, index=False)
    save_json(_jsonable({"run": f"{stem}_s{seed}", "horizon": h, "seed": seed, "settings": s,
                         "val": val_m, "test": direction_metrics(test.true_up, test.pred_up, test.prob_up),
                         "history": histories}), summ_file)
    return test[cols]


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))


# ----------------------------------------------------------------------------- comparison
def price_forecasts(cfg: dict, h: int) -> dict[str, pd.DataFrame]:
    """Every existing price forecast for horizon h, turned into a direction forecast."""
    pred_dir = Path(path(cfg, "results_dir", "predictions"))
    e = cfg["experiments"]
    out = {}
    for name, label in (("rw", "Random walk (= always up)"), ("arima", "ARIMA"), ("arimax", "ARIMAX")):
        f = pred_dir / f"baseline_{name}_h{h}.csv"
        if f.exists():
            out[label] = pd.read_csv(f, parse_dates=["origin_date", "target_date"])
    for m in e["models"]:
        for fs in e["feature_sets"]:
            frames = [pd.read_csv(f, parse_dates=["origin_date", "target_date"])
                      for sd in e["seeds"] if (f := pred_dir / f"{m}_{fs}_h{h}_s{sd}.csv").exists()]
            if frames:
                avg = frames[0].copy()
                avg["pred_price"] = sum(fr.pred_price for fr in frames) / len(frames)
                out[f"{m.upper()} / {fs}"] = avg
    for df in out.values():
        df["pred_up"] = sign_forecast(df.pred_price, df.base_price)
        df["true_up"] = (df.true_price >= df.base_price).astype(int)
    return out


def comparison_rows(models: dict[str, pd.DataFrame], h: int, s: dict) -> list[dict]:
    common = set.intersection(*(set(df.origin_date) for df in models.values()))
    if s["drop_ties"]:   # also covers CNN prediction files saved before this option existed
        ref = next(iter(models.values()))
        common -= set(ref.origin_date[ref.true_price == ref.base_price])
    rows = []
    for label, df in models.items():
        df = df[df.origin_date.isin(common)].sort_values("origin_date")
        ret = df.true_price / df.base_price.where(df.base_price > 0) - 1
        subsets = {"all": np.ones(len(df), bool), f"|ret|>{s['big_move']:.0%}": (ret.abs() > s["big_move"]).values}
        for name, (a, b) in s["special_periods"].items():
            subsets[name] = ((df.origin_date >= a) & (df.origin_date <= b)).values
        for sub, m in subsets.items():
            if m.sum() == 0:
                continue
            prob = df.prob_up[m] if "prob_up" in df else None
            rows.append({"horizon": h, "subset": sub, "model": label,
                         **direction_metrics(df.true_up[m], df.pred_up[m], prob)})
    return rows


# ----------------------------------------------------------------------------- figures
def example_figure(cfg, s):
    f = path(cfg, "raw_dir", f"ohlcv_{s['source']}.csv")
    ohlcv = clean_ohlcv(pd.read_csv(f, index_col=0, parse_dates=True), verbose=False)
    k = len(ohlcv) // 2
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.4))
    for ax, st in zip(axes, STYLES):
        im, d = make_images(ohlcv.iloc[k - s["window"] + 1: k + 1], s["window"], s["height"], st, s["min_intensity"])
        ax.imshow(im[0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(st.replace("_", " "), fontsize=9); ax.axis("off")
    fig.suptitle(f"{s['window']}-day images ending {d[0].date()}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_file(cfg, "figures", "image_cnn_examples.png"), dpi=150); plt.close(fig)


class _TwoD(nn.Module):
    """SHAP expects outputs of shape (batch, n_outputs)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).unsqueeze(-1)


def explain(cfg, s, imgs, samples, h, stem, rng):
    """SHAP (expected gradients) on the seed-0 fixed-mode CNN: which parts of the chart matter."""
    import shap

    ckpt = torch.load(out_file(cfg, "models", f"{stem}_s{s['seeds'][0]}.pt"), map_location="cpu", weights_only=False)
    _, H, W = imgs.shape
    model = ImageCNN(H, W, s["channels"], s["dropout"])
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    wrap = _TwoD(model)                                              # (batch,) -> (batch, 1)
    tr, va = samples[samples.split == "train"], samples[samples.split == "val"]
    to_t = lambda idx: torch.from_numpy(imgs[idx]).float().div(255).unsqueeze(1)
    bg = to_t(rng.choice(tr.img_idx.values, min(s["explain_background"], len(tr)), replace=False))
    ex = to_t(rng.choice(va.img_idx.values, min(s["explain_size"], len(va)), replace=False))
    sv = np.asarray(shap.GradientExplainer(wrap, bg).shap_values(ex))
    sv = np.abs(sv.reshape(len(ex), H, W))                         # drop channel / output axes
    heat = sv.mean(axis=0)
    per_day = heat.reshape(H, s["window"], 3).sum(axis=(0, 2))
    per_part = heat.reshape(H, s["window"], 3).sum(axis=(0, 1))
    pd.DataFrame({"day": [f"t-{s['window'] - 1 - i}" for i in range(s["window"])], "mean_abs_shap": per_day}) \
        .to_csv(out_file(cfg, "shap", f"{stem}_attr_by_day.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 2.8), gridspec_kw={"width_ratios": [2.2, 2.2, 1]})
    axes[0].imshow(heat, cmap="magma", interpolation="nearest", aspect="auto")
    axes[0].set_title("mean |SHAP| per pixel"); axes[0].set_yticks([])
    axes[0].set_xticks(np.arange(1, W, 15), [f"t-{s['window'] - 1 - i}" for i in range(0, s["window"], 5)])
    axes[1].bar(range(s["window"]), per_day, color="#4a6fa5")
    axes[1].set_xticks(range(0, s["window"], 5), [f"t-{s['window'] - 1 - i}" for i in range(0, s["window"], 5)])
    axes[1].set_title("by day in window (right = origin)")
    axes[2].bar(["open", "high-low", "close"], per_part / per_part.sum(), color="#4a6fa5")
    axes[2].set_title("share by bar part")
    fig.suptitle(f"Image CNN attribution, h={h} ({s['style']})", fontsize=9)
    fig.tight_layout(); fig.savefig(out_file(cfg, "figures", f"image_cnn_attr_h{h}.png"), dpi=150); plt.close(fig)
    print(f"   SHAP: share of attribution in the last 5 days = {per_day[-5:].sum() / per_day.sum():.2f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=STYLES, default=None)
    ap.add_argument("--retrain", choices=["fixed", "yearly"], default=None)
    ap.add_argument("--source", default=None, help="OHLCV file suffix, e.g. WTI or BRENT")
    ap.add_argument("--horizons", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--explain", action="store_true", help="SHAP over the chart (fixed mode)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    s = settings(cfg, args)
    device = get_device()
    imgs, img_dates = load_images(cfg, s)
    levels = load_levels(cfg)
    example_figure(cfg, s)
    rng = np.random.default_rng(cfg.get("seed", 0))

    rows = []
    for h in s["horizons"]:
        stem = f"image_cnn_{s['style']}_h{h}" + ("_yearly" if s["retrain"] == "yearly" else "")
        samples = build_samples(cfg, levels, img_dates, h, s)
        P = levels[cfg["data"]["target"]]
        ties = (P.shift(-h) == P).iloc[:-h].mean()
        print(f"\n=== {stem} | device {device} | share up: "
              + ", ".join(f"{k} {v:.3f}" for k, v in samples.groupby("split").true_up.mean().items())
              + f" | zero-change targets {ties:.1%} " + ("(dropped)" if s["drop_ties"] else "(counted as up)"))

        per_seed = [run_seed(cfg, s, imgs, samples, h, sd, device, stem, args.force) for sd in s["seeds"]]
        avg = per_seed[0].copy()
        avg["prob_up"] = np.mean([p.prob_up.values for p in per_seed], axis=0)
        avg["pred_up"] = (avg.prob_up >= 0.5).astype(int)
        avg.to_csv(out_file(cfg, "predictions", f"{stem}_avg.csv"), index=False)

        models = {f"Image CNN / {s['style']}" + (" (yearly)" if s["retrain"] == "yearly" else ""): avg,
                  **price_forecasts(cfg, h)}
        rows += comparison_rows(models, h, s)

        if args.explain:
            if s["retrain"] != "fixed":
                print("   --explain needs --retrain fixed; skipped")
            else:
                explain(cfg, s, imgs, samples, h, stem, rng)

    table = pd.DataFrame(rows)
    suffix = "" if (s["style"], s["retrain"]) == ("volume_intensity", "fixed") else f"_{s['style']}_{s['retrain']}"
    table.to_csv(out_file(cfg, "tables", f"direction_comparison{suffix}.csv"), index=False)
    show = ["horizon", "subset", "model", "n", "Acc", "F1", "share_up_pred", "PT_stat", "PT_p", "AUC"]
    with open(out_file(cfg, "tables", f"direction_comparison{suffix}.md"), "w") as f:
        f.write(table[[c for c in show if c in table]].round(3).to_markdown(index=False))
    print("\n" + table[table.subset == "all"][[c for c in show if c in table]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
