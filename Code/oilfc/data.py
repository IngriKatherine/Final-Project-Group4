"""Loading, transforming, splitting, scaling and windowing the data.

Conventions
-----------
* Row t of the feature frame holds information known at the close of day t.
* For horizon h, the target at origin t is the h-step-ahead log return
  y_t = log(P_{t+h}) - log(P_t)          (target_transform = "log_return")
  or the price level P_{t+h}            (target_transform = "level").
* A sample is assigned to train / val / test by its TARGET date, so no target
  used for training lies inside the validation or test period.
* Scalers are fit on training rows only (no look-ahead leakage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import load_json, path

# Columns that are rates / spreads (can be <= 0): use first differences, not log returns.
RATE_COLUMNS = {"UST10Y", "UST2Y", "FEDFUNDS", "BREAKEVEN10Y"}


# -----------------------------------------------------------------------------
# Loading and transforms
# -----------------------------------------------------------------------------
def load_levels(cfg: dict) -> pd.DataFrame:
    """Clean, aligned price/rate LEVELS written by 02_preprocess.py."""
    f = path(cfg, "processed_dir", "levels.csv")
    return pd.read_csv(f, index_col=0, parse_dates=True).sort_index()


def make_features(levels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stationary transforms: log returns for prices, first differences for rates."""
    out = {}
    for col in levels.columns:
        s = levels[col]
        if col in RATE_COLUMNS or cfg["framing"]["feature_transform"] == "diff":
            out[col] = s.diff()
        else:
            out[col] = np.log(s).diff()
    return pd.DataFrame(out, index=levels.index).iloc[1:]


def make_target(levels: pd.DataFrame, target: str, horizon: int, transform: str) -> pd.Series:
    p = levels[target]
    if transform == "log_return":
        return np.log(p.shift(-horizon)) - np.log(p)
    if transform == "level":
        return p.shift(-horizon)
    raise ValueError(f"unknown target_transform {transform!r}")


def to_price(y_pred: np.ndarray, base_price: np.ndarray, transform: str) -> np.ndarray:
    """Convert model outputs (unscaled) back to price levels."""
    if transform == "log_return":
        return base_price * np.exp(y_pred)
    return y_pred


# -----------------------------------------------------------------------------
# Feature sets
# -----------------------------------------------------------------------------
def feature_columns(cfg: dict, feature_set: str | list, horizon: int, all_cols: list[str]) -> list[str]:
    """Resolve a feature-set name to a list of columns (target is always first).

    price_only    : only the target's own history
    all           : every available series
    shap_selected : target + exogenous drivers chosen by 05_shap_analysis.py
    list          : explicit list of columns
    """
    target = cfg["data"]["target"]
    others = [c for c in all_cols if c != target]
    if isinstance(feature_set, (list, tuple)):
        chosen = list(feature_set)
    elif feature_set == "price_only":
        chosen = []
    elif feature_set == "all":
        chosen = others
    elif feature_set == "shap_selected":
        sel_file = path(cfg, "results_dir", "shap", f"selected_features_h{horizon}.json")
        if not sel_file.exists():
            raise FileNotFoundError(f"{sel_file} not found - run 05_shap_analysis.py first.")
        chosen = load_json(sel_file)["selected"]
    else:
        raise ValueError(f"unknown feature_set {feature_set!r}")
    return [target] + [c for c in chosen if c != target]


# -----------------------------------------------------------------------------
# Windowed datasets
# -----------------------------------------------------------------------------
@dataclass
class SplitData:
    X: np.ndarray            # (N, lookback, n_features), scaled
    y: np.ndarray            # (N,), scaled target
    y_raw: np.ndarray        # (N,), unscaled target
    base_price: np.ndarray   # (N,), P_t at the forecast origin
    true_price: np.ndarray   # (N,), P_{t+h}
    origin_dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex


@dataclass
class Prepared:
    splits: dict[str, SplitData]
    feature_names: list[str]
    y_scaler: StandardScaler
    x_scaler: StandardScaler
    horizon: int
    lookback: int
    meta: dict = field(default_factory=dict)

    def inverse_y(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def build_datasets(cfg: dict, feature_set, horizon: int, lookback: int | None = None,
                   levels: pd.DataFrame | None = None) -> Prepared:
    """Create scaled (X, y) windows for train / val / test for one horizon and feature set."""
    lookback = lookback or cfg["framing"]["lookback"]
    transform = cfg["framing"]["target_transform"]
    target = cfg["data"]["target"]
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    val_end = pd.Timestamp(cfg["split"]["val_end"])

    levels = load_levels(cfg) if levels is None else levels
    feats = make_features(levels, cfg)
    cols = feature_columns(cfg, feature_set, horizon, list(feats.columns))
    feats = feats[cols]

    y = make_target(levels, target, horizon, transform).reindex(feats.index)
    dates = feats.index
    target_dates = pd.Series(levels.index, index=levels.index).shift(-horizon).reindex(dates)
    price = levels[target].reindex(dates)
    future_price = levels[target].shift(-horizon).reindex(dates)

    # Fit scalers on TRAIN rows only.
    x_scaler = StandardScaler().fit(feats.loc[dates <= train_end].values)
    X_all = x_scaler.transform(feats.values).astype(np.float32)

    valid = np.arange(lookback - 1, len(dates))
    valid = valid[~np.isnan(y.values[valid])]
    td = pd.DatetimeIndex(target_dates.values[valid])
    masks = {
        "train": td <= train_end,
        "val": (td > train_end) & (td <= val_end),
        "test": td > val_end,
    }
    y_scaler = StandardScaler().fit(y.values[valid][masks["train"]].reshape(-1, 1))

    splits = {}
    for name, m in masks.items():
        idx = valid[m]
        X = np.stack([X_all[i - lookback + 1: i + 1] for i in idx]) if len(idx) else \
            np.empty((0, lookback, len(cols)), np.float32)
        y_raw = y.values[idx].astype(np.float64)
        splits[name] = SplitData(
            X=X,
            y=y_scaler.transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32),
            y_raw=y_raw,
            base_price=price.values[idx],
            true_price=future_price.values[idx],
            origin_dates=pd.DatetimeIndex(dates[idx]),
            target_dates=pd.DatetimeIndex(target_dates.values[idx]),
        )
    return Prepared(splits, cols, y_scaler, x_scaler, horizon, lookback,
                    meta={"feature_set": feature_set if isinstance(feature_set, str) else "custom"})


def predictions_frame(split: SplitData, pred_price: np.ndarray) -> pd.DataFrame:
    """Standard layout for every saved prediction file (RNNs and baselines alike)."""
    return pd.DataFrame({
        "origin_date": split.origin_dates,
        "target_date": split.target_dates,
        "base_price": split.base_price,
        "true_price": split.true_price,
        "pred_price": pred_price,
    })


def torch_loader(split: SplitData, batch_size: int, shuffle: bool, num_workers: int = 0):
    """Wrap a SplitData as a PyTorch DataLoader.

    Shuffling the *windows* of the training set is fine (each window is
    self-contained); never shuffle before splitting by date.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.from_numpy(split.X), torch.from_numpy(split.y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
