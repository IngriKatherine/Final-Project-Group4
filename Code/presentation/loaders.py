"""
Data access for the presentation: every number on every slide comes from here.

Nothing is hard-coded. The app reads what the pipeline scripts wrote into
data/ and results/, so after you run 06_run_experiments.py or 07_ablations.py
the slides pick up the new runs on the next reload (press R in the browser,
or "Reload data" in the sidebar).

Reading is cached with @st.cache_data (tables) and @st.cache_resource
(PyTorch checkpoints), as in the course's Streamlit examples.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

# presentation/ -> Code/. Put Code/ on the path so we reuse the group's own
# metric and data code (oilfc) instead of re-implementing it.
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from oilfc.metrics import all_metrics, diebold_mariano  # noqa: E402

RESULTS = CODE_DIR / "results"
PRED_DIR = RESULTS / "predictions"
RUNS_DIR = RESULTS / "tables" / "runs"
SHAP_DIR = RESULTS / "shap"
FIG_DIR = RESULTS / "figures"
MODEL_DIR = RESULTS / "models"

RATE_COLUMNS = {"UST10Y", "UST2Y", "FEDFUNDS", "BREAKEVEN10Y"}

FEATURE_LABELS = {
    "WTI": "WTI crude",
    "BRENT": "Brent crude",
    "USD_BROAD": "Broad US dollar index",
    "UST10Y": "10-year Treasury yield",
    "UST2Y": "2-year Treasury yield",
    "FEDFUNDS": "Fed funds rate",
    "BREAKEVEN10Y": "10-year breakeven inflation",
    "DXY": "US dollar index (DXY)",
    "GOLD": "Gold futures",
    "SP500": "S&P 500",
    "VIX": "VIX volatility index",
    "NATGAS": "Natural gas futures",
    "XLE": "Energy ETF (XLE)",
}
FEATURE_UNITS = {
    "WTI": "USD/bbl", "BRENT": "USD/bbl", "USD_BROAD": "index", "UST10Y": "%", "UST2Y": "%",
    "FEDFUNDS": "%", "BREAKEVEN10Y": "%", "DXY": "index", "GOLD": "USD/oz", "SP500": "index",
    "VIX": "index", "NATGAS": "USD/MMBtu", "XLE": "USD",
}
FEATURE_SET_LABELS = {"price_only": "Price only", "all": "All features", "shap_selected": "SHAP-selected"}
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "rw": "Random walk", "arima": "ARIMA"}
METRIC_HELP = {
    "RMSE": "Root mean squared error in USD/bbl. Penalises big misses.",
    "MAE": "Mean absolute error in USD/bbl.",
    "MAPE": "Mean absolute percentage error, %.",
    "DirAcc": "Share of forecasts that get the direction of the move right.",
    "Skill": "1 - RMSE / RMSE of the random walk, in %. Above 0 means better than 'no change'.",
}

RUN_RE = re.compile(r"^(lstm|gru)_(price_only|all|shap_selected)_h(\d+)_s(\d+)$")
BASE_RE = re.compile(r"^baseline_(rw|arima)_h(\d+)$")


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def config_label(model: str, features: str) -> str:
    if model in ("rw", "arima"):
        return MODEL_LABELS[model]
    return f"{MODEL_LABELS.get(model, model.upper())} / {FEATURE_SET_LABELS.get(features, features)}"


def clear_all_caches():
    st.cache_data.clear()
    st.cache_resource.clear()


# -----------------------------------------------------------------------------
# Config and raw data
# -----------------------------------------------------------------------------
@st.cache_data
def config() -> dict:
    with open(CODE_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data
def levels() -> pd.DataFrame:
    """Aligned, cleaned price/rate levels written by 02_preprocess.py."""
    return pd.read_csv(CODE_DIR / "data" / "processed" / "levels.csv",
                       index_col=0, parse_dates=True).sort_index()


@st.cache_data
def raw_series(source: str) -> pd.DataFrame | None:
    """data/raw/fred.csv or yahoo.csv, before cleaning (shows the negative WTI print)."""
    f = CODE_DIR / "data" / "raw" / f"{source}.csv"
    return pd.read_csv(f, index_col=0, parse_dates=True).sort_index() if f.exists() else None


@st.cache_data
def download_log() -> dict:
    f = CODE_DIR / "data" / "raw" / "download_log.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data
def returns() -> pd.DataFrame:
    """Same transform as oilfc.data.make_features: log returns for prices, differences for rates."""
    lv = levels()
    out = {c: (lv[c].diff() if c in RATE_COLUMNS else np.log(lv[c]).diff()) for c in lv.columns}
    return pd.DataFrame(out, index=lv.index).iloc[1:]


@st.cache_data
def table(name: str) -> pd.DataFrame | None:
    """Any CSV in results/tables, or None if that script has not been run yet."""
    f = RESULTS / "tables" / f"{name}.csv"
    return pd.read_csv(f) if f.exists() else None


def series_sources() -> pd.DataFrame:
    """One row per series: where it came from and how much of it there is."""
    cfg, log, lv = config(), download_log(), levels()
    rows = []
    for source, key in (("FRED", "fred_series"), ("Yahoo Finance", "yahoo_tickers")):
        for name, sid in cfg["data"][key].items():
            cov = log.get("fred" if source == "FRED" else "yahoo", {}).get(name, {})
            rows.append({
                "Series": label(name), "Code": name, "Source": source, "ID / ticker": sid,
                "Unit": FEATURE_UNITS.get(name, ""), "First date": cov.get("first", ""),
                "Raw obs.": cov.get("n_obs"), "Used obs.": int(lv[name].notna().sum()) if name in lv else None,
                "Role": "Target" if name == cfg["data"]["target"] else "Candidate driver",
            })
    return pd.DataFrame(rows)


def split_dates() -> tuple[pd.Timestamp, pd.Timestamp]:
    s = config()["split"]
    return pd.Timestamp(s["train_end"]), pd.Timestamp(s["val_end"])


# -----------------------------------------------------------------------------
# Runs and predictions
# -----------------------------------------------------------------------------
@st.cache_data
def runs() -> pd.DataFrame:
    """Every training run summary in results/tables/runs/*.json, flattened to one row each."""
    rows = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        h = d.get("history", {})
        val_loss = h.get("val_loss", [])
        row = {
            "run": d["run"], "model": d["model"], "features": d["features"], "horizon": d["horizon"],
            "seed": d["seed"], "tag": d.get("tag", "") or "", "n_params": d.get("n_params"),
            "n_features": len(d.get("feature_names", [])), "lookback": d.get("lookback"),
            "hidden_size": d.get("model_cfg", {}).get("hidden_size"),
            "num_layers": d.get("model_cfg", {}).get("num_layers"),
            "dropout": d.get("model_cfg", {}).get("dropout"),
            "epochs": h.get("epochs_run"), "best_epoch": int(np.argmin(val_loss)) + 1 if val_loss else None,
            "train_seconds": h.get("train_seconds"),
        }
        row.update({f"val_{k}": v for k, v in d.get("val", {}).items()})
        row.update({f"test_{k}": v for k, v in d.get("test", {}).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main_runs() -> pd.DataFrame:
    r = runs()
    return r[r["tag"] == ""] if not r.empty else r


def ablation_runs() -> pd.DataFrame:
    """Runs written by 07_ablations.py carry a tag like 'abl-hidden_size128'."""
    r = runs()
    if r.empty:
        return r
    r = r[r["tag"].str.startswith("abl-")].copy()
    params = list(config().get("ablations", {}).keys())

    def split_tag(tag):
        body = tag[4:]
        for p in sorted(params, key=len, reverse=True):
            if body.startswith(p):
                return p, float(body[len(p):])
        return None, np.nan

    parsed = r["tag"].map(split_tag)
    r["param"] = [p for p, _ in parsed]
    r["value"] = [v for _, v in parsed]
    return r.dropna(subset=["param"])


@st.cache_data
def history(run: str) -> dict:
    f = RUNS_DIR / f"{run}.json"
    return json.loads(f.read_text()).get("history", {}) if f.exists() else {}


@st.cache_data
def prediction_file(stem: str) -> pd.DataFrame:
    return pd.read_csv(PRED_DIR / f"{stem}.csv", parse_dates=["origin_date", "target_date"])


@st.cache_data
def catalog() -> pd.DataFrame:
    """What prediction files exist: one row per (kind, model, features, horizon, seed)."""
    rows = []
    for f in sorted(PRED_DIR.glob("*.csv")):
        m = RUN_RE.match(f.stem)
        if m:
            rows.append({"kind": "rnn", "model": m[1], "features": m[2], "horizon": int(m[3]),
                         "seed": int(m[4]), "stem": f.stem})
            continue
        m = BASE_RE.match(f.stem)
        if m:
            rows.append({"kind": "baseline", "model": m[1], "features": "price_only",
                         "horizon": int(m[2]), "seed": -1, "stem": f.stem})
    return pd.DataFrame(rows, columns=["kind", "model", "features", "horizon", "seed", "stem"])


def horizons() -> list[int]:
    c = catalog()
    hs = sorted(c["horizon"].unique().tolist()) if not c.empty else []
    return hs or list(config()["framing"]["horizons"])


def configs(horizon: int | None = None) -> list[tuple[str, str]]:
    """Available (model, features) pairs, baselines first, in a stable order."""
    c = catalog()
    if horizon is not None:
        c = c[c["horizon"] == horizon]
    order_m = {"rw": 0, "arima": 1, "lstm": 2, "gru": 3}
    order_f = {"price_only": 0, "all": 1, "shap_selected": 2}
    pairs = {(r.model, r.features) for r in c.itertuples()}
    return sorted(pairs, key=lambda p: (order_m.get(p[0], 9), order_f.get(p[1], 9)))


@st.cache_data
def forecasts(model: str, features: str, horizon: int) -> pd.DataFrame | None:
    """Test-period forecasts. RNNs are averaged over seeds (as in 08_make_figures_tables.py)."""
    c = catalog()
    c = c[(c.model == model) & (c.features == features) & (c.horizon == horizon)]
    if c.empty:
        return None
    frames = [prediction_file(s) for s in c["stem"]]
    out = frames[0][["origin_date", "target_date", "base_price", "true_price"]].copy()
    preds = np.column_stack([f["pred_price"].to_numpy() for f in frames])
    out["pred_price"] = preds.mean(axis=1)
    out["pred_std"] = preds.std(axis=1) if preds.shape[1] > 1 else 0.0
    out["n_seeds"] = preds.shape[1]
    out["error"] = out["pred_price"] - out["true_price"]
    return out


@st.cache_data
def comparison() -> pd.DataFrame:
    """One row per (horizon, model, features): metrics of the seed-averaged forecast,
    plus the spread of RMSE across seeds, and skill relative to the random walk."""
    rows = []
    mr = main_runs()
    for h in horizons():
        rw_rmse = None
        for model, feats in configs(h):
            p = forecasts(model, feats, h)
            if p is None:
                continue
            m = all_metrics(p.true_price, p.pred_price, p.base_price)
            if model == "rw":
                m["DirAcc"] = np.nan  # predicts "no change": direction is undefined
                rw_rmse = m["RMSE"]
            row = {"horizon": h, "model": model, "features": feats, "config": config_label(model, feats),
                   "n_seeds": int(p["n_seeds"].iloc[0]), **m}
            if model in ("lstm", "gru") and not mr.empty:
                s = mr[(mr.model == model) & (mr.features == feats) & (mr.horizon == h)]["test_RMSE"]
                row["RMSE_seed_mean"] = s.mean()
                row["RMSE_seed_std"] = s.std() if len(s) > 1 else np.nan
            rows.append(row)
        if rw_rmse:
            for row in rows:
                if row["horizon"] == h:
                    row["Skill"] = 100 * (1 - row["RMSE"] / rw_rmse)
    return pd.DataFrame(rows)


@st.cache_data
def dm_tests() -> pd.DataFrame:
    """Diebold-Mariano tests (HLN-corrected, from oilfc.metrics) on matched test origins.

    Each RNN configuration is tested against both baselines, and the feature sets are
    tested against each other - the comparisons the research question is about.
    A negative statistic means the first model is more accurate."""
    rows = []

    def test(h, a, b, kind):
        pa, pb = forecasts(*a, h), forecasts(*b, h)
        if pa is None or pb is None:
            return
        j = pa.merge(pb, on="origin_date", suffixes=("", "_b"))
        r = diebold_mariano(j.true_price, j.pred_price, j.pred_price_b, h=h)
        rows.append({"horizon": h, "comparison": kind, "model": config_label(*a), "vs": config_label(*b),
                     "DM": r["DM"], "p_value": r["p_value"], "n": len(j)})

    for h in horizons():
        rnn = [c for c in configs(h) if c[0] in ("lstm", "gru")]
        for c in rnn:
            test(h, c, ("rw", "price_only"), "vs. baseline")
            test(h, c, ("arima", "price_only"), "vs. baseline")
        for model in ("lstm", "gru"):
            for a, b in (("all", "price_only"), ("shap_selected", "all"), ("shap_selected", "price_only")):
                if (model, a) in rnn and (model, b) in rnn:
                    test(h, (model, a), (model, b), "between feature sets")
    return pd.DataFrame(rows)


def verdicts(alpha: float = 0.05) -> dict[str, tuple[str, str, str]]:
    """Judge H1-H3 from whatever results exist. Returns {key: (kind, label, explanation)}
    with kind in {"yes", "no", "mixed", "pending"} (yes = hypothesis supported)."""
    comp, dm = comparison(), dm_tests()
    out = {}

    def rmse(model, feats, h):
        r = comp[(comp.model == model) & (comp.features == feats) & (comp.horizon == h)]
        return None if r.empty else float(r.RMSE.iloc[0])

    def dm_row(model_a, feats_a, model_b, feats_b, h):
        if dm.empty:
            return None
        r = dm[(dm.horizon == h) & (dm.model == config_label(model_a, feats_a)) &
               (dm.vs == config_label(model_b, feats_b))]
        return None if r.empty else r.iloc[0]

    # H1: price-only network vs. random walk at h = 1
    beat, checked = [], []
    for model in ("lstm", "gru"):
        r = dm_row(model, "price_only", "rw", "price_only", 1)
        if r is not None:
            checked.append(MODEL_LABELS[model])
            if r.DM < 0 and r.p_value < alpha:
                beat.append(MODEL_LABELS[model])
    if not checked:
        out["H1"] = ("pending", "Pending", "Needs the price-only runs from 06_run_experiments.py.")
    elif beat:
        out["H1"] = ("no", "Not supported", f"{' and '.join(beat)} with price only beat the random walk at 1 day.")
    else:
        out["H1"] = ("yes", "Supported", f"No price-only network ({', '.join(checked)}) significantly beats "
                                         f"the random walk at 1 day.")

    # H2: all features vs. price only
    better, pairs = 0, 0
    for model in ("lstm", "gru"):
        for h in horizons():
            r = dm_row(model, "all", model, "price_only", h)
            if r is not None:
                pairs += 1
                better += int(r.DM < 0 and r.p_value < alpha)
    if pairs == 0:
        extra = ""
        rw_losses = [h for h in horizons() if (a := rmse("lstm", "all", h)) and (b := rmse("rw", "price_only", h)) and a > b]
        if rw_losses:
            extra = f" So far: the all-features LSTM has a higher RMSE than the random walk at h = {', '.join(map(str, rw_losses))}."
        out["H2"] = ("pending", "Pending", "Needs price-only runs to compare against." + extra)
    elif better == 0:
        out["H2"] = ("yes", "Supported", f"All features never significantly beat price only ({pairs} comparisons).")
    elif better == pairs:
        out["H2"] = ("no", "Not supported", f"All features beat price only in all {pairs} comparisons.")
    else:
        out["H2"] = ("mixed", "Mixed", f"All features beat price only in {better} of {pairs} comparisons.")

    # H3: SHAP-selected vs. both other feature sets
    wins, sig, pairs = 0, 0, 0
    for model in ("lstm", "gru"):
        for h in horizons():
            s = rmse(model, "shap_selected", h)
            for other in ("all", "price_only"):
                o = rmse(model, other, h)
                if s is None or o is None:
                    continue
                pairs += 1
                wins += int(s < o)
                r = dm_row(model, "shap_selected", model, other, h)
                sig += int(r is not None and r.DM < 0 and r.p_value < alpha)
    if pairs == 0:
        out["H3"] = ("pending", "Pending", "Needs the SHAP-selected and price-only runs from 06_run_experiments.py.")
    elif wins == pairs and sig > 0:
        out["H3"] = ("yes", "Supported", f"SHAP-selected has the lower RMSE in all {pairs} comparisons, "
                                         f"{sig} of them significant.")
    elif wins > 0:
        out["H3"] = ("mixed", "Mixed", f"SHAP-selected has the lower RMSE in {wins} of {pairs} comparisons "
                                       f"({sig} significant).")
    else:
        out["H3"] = ("no", "Not supported", f"SHAP-selected never has the lower RMSE ({pairs} comparisons).")
    return out


# -----------------------------------------------------------------------------
# SHAP
# -----------------------------------------------------------------------------
def shap_horizons() -> list[int]:
    return sorted(int(re.findall(r"\d+", f.stem)[0]) for f in SHAP_DIR.glob("importance_h*.csv"))


@st.cache_data
def shap_importance(h: int) -> pd.DataFrame:
    df = pd.read_csv(SHAP_DIR / f"importance_h{h}.csv", index_col=0)
    seed_cols = [c for c in df.columns if c.startswith("seed")]
    df["seed_std"] = df[seed_cols].std(axis=1) if len(seed_cols) > 1 else 0.0
    return df.sort_values("rank")


@st.cache_data
def shap_lags(h: int) -> pd.DataFrame:
    return pd.read_csv(SHAP_DIR / f"lag_importance_h{h}.csv", index_col=0)


@st.cache_data
def shap_selection(h: int) -> dict:
    f = SHAP_DIR / f"selected_features_h{h}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def figure_path(name: str) -> Path | None:
    f = FIG_DIR / name
    return f if f.exists() else None


# -----------------------------------------------------------------------------
# Live inference with saved checkpoints (Model lab)
# -----------------------------------------------------------------------------
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def checkpoints() -> pd.DataFrame:
    rows = []
    for f in sorted(MODEL_DIR.glob("*.pt")):
        m = RUN_RE.match(f.stem)
        if m:
            rows.append({"model": m[1], "features": m[2], "horizon": int(m[3]), "seed": int(m[4]), "stem": f.stem})
    return pd.DataFrame(rows, columns=["model", "features", "horizon", "seed", "stem"])


@st.cache_resource(show_spinner="Loading the trained network...")
def load_checkpoint(stem: str):
    """Rebuild the RNNForecaster from a .pt file written by 04_train_rnn.py (CPU is plenty)."""
    import torch
    from oilfc.models import RNNForecaster

    ck = torch.load(MODEL_DIR / f"{stem}.pt", map_location="cpu", weights_only=False)
    m = ck["model_cfg"]
    net = RNNForecaster(len(ck["feature_names"]), m["type"], m["hidden_size"], m["num_layers"],
                        m["dropout"], m.get("bidirectional", False))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


def scaled_window(ck: dict, origin: pd.Timestamp) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The (lookback, n_features) scaled input the network sees at a forecast origin."""
    feats = returns()[ck["feature_names"]]
    pos = feats.index.get_indexer([origin])[0]
    L = ck["lookback"]
    win = feats.iloc[pos - L + 1: pos + 1]
    X = (win.to_numpy() - np.asarray(ck["x_mean"])) / np.asarray(ck["x_scale"])
    return X.astype(np.float32), win.index


def predict_return(stems: list[str], X: np.ndarray) -> np.ndarray:
    """Unscaled h-step log-return forecasts for a batch of windows, averaged over the given seeds.
    X has shape (batch, lookback, n_features)."""
    import torch

    outs = []
    with torch.no_grad():
        for stem in stems:
            net, ck = load_checkpoint(stem)
            y = net(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))).numpy()
            outs.append(y * float(np.ravel(ck["y_scale"])[0]) + float(np.ravel(ck["y_mean"])[0]))
    return np.mean(outs, axis=0)
