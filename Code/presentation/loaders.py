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
FEATURE_SET_LABELS = {"price_only": "Price only", "all": "Price + market drivers", "shap_selected": "Selected drivers",
                      "ais": "Price + AIS supply", "satellite": "Price + satellite supply",
                      "satellite_ais": "Price + satellite + AIS"}
MODEL_LABELS = {"lstm": "LSTM", "gru": "GRU", "rw": "Random walk", "arima": "ARIMA", "arimax": "ARIMAX",
                "imgcnn": "Image CNN", "cnn": "CNN"}
BASELINES = ("rw", "arima", "arimax")              # statistical models (no seeds)
MODEL_ORDER = {"rw": 0, "arima": 1, "arimax": 2, "lstm": 3, "gru": 4, "imgcnn": 5, "cnn": 6}
FEATURE_ORDER = {"price_only": 0, "all": 1, "shap_selected": 2, "ais": 3, "satellite": 4, "satellite_ais": 5}
METRIC_HELP = {
    "RMSE": "Root mean squared error in USD/bbl. Penalises big misses.",
    "MAE": "Mean absolute error in USD/bbl.",
    "MAPE": "Mean absolute percentage error, %.",
    "DirAcc": "Share of forecasts that get the direction of the move right.",
    "Skill": "1 - RMSE / RMSE of the random walk, in %. Above 0 means better than 'no change'.",
}

# <model>_<input set>_h<h>_s<seed>, e.g. lstm_price_only_h5_s0 or imgcnn_price_only_h5_s1.
# The model name has no underscore; ablation files (..._abl-...) do not match on purpose.
RUN_RE = re.compile(r"^([a-z][a-z0-9]*)_([a-z][a-z_]*?)_h(\d+)_s(\d+)$")
BASE_RE = re.compile(r"^baseline_(rw|arima|arimax)_h(\d+)$")
PRED_2026_DIR = RESULTS / "predictions_2026"
SAT_DIR = CODE_DIR / "satellite"


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


def config_label(model: str, features: str) -> str:
    if model in BASELINES:
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
            rows.append({"kind": "baseline", "model": m[1], "features": "all" if m[1] == "arimax" else "price_only",
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
    pairs = {(r.model, r.features) for r in c.itertuples()}
    return sorted(pairs, key=lambda p: (MODEL_ORDER.get(p[0], 9), FEATURE_ORDER.get(p[1], 9), p))


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
            if model not in BASELINES and not mr.empty:
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
        cf = configs(h)
        nets = [c for c in cf if c[0] not in BASELINES]
        bases = [c for c in cf if c[0] in BASELINES]
        for c in nets:
            for b in bases:
                test(h, c, b, "vs. baseline")
        if ("arimax", "all") in cf and ("arima", "price_only") in cf:
            test(h, ("arimax", "all"), ("arima", "price_only"), "vs. baseline")
        for model in sorted({m for m, _ in nets}):
            feats = [f for m, f in nets if m == model]
            for a in feats:
                if a != "price_only" and "price_only" in feats:
                    test(h, (model, a), (model, "price_only"), "between input sets")
    return pd.DataFrame(rows)


def verdicts(alpha: float = 0.05) -> dict[str, tuple[str, str, str]]:
    """Judge H1-H4 (content.HYPOTHESES) from whatever results exist.
    Returns {key: (kind, label, explanation)} with kind in {"yes", "no", "mixed", "pending"} (yes = supported)."""
    from scipy import stats

    comp, dm = comparison(), dm_tests()
    out = {}

    # H1: nothing beats the random walk at h = 1
    d1 = dm[(dm.horizon == 1) & (dm.vs == "Random walk")] if not dm.empty else dm
    if d1 is None or d1.empty:
        out["H1"] = ("pending", "Pending", "Needs 1-day forecasts and the random walk (03_baselines.py, 06_run_experiments.py).")
    else:
        beat = d1[(d1.DM < 0) & (d1.p_value < alpha)]
        if beat.empty:
            out["H1"] = ("yes", "Supported", f"None of the {len(d1)} models significantly beats the random walk at 1 day.")
        else:
            out["H1"] = ("no", "Not supported", "Significantly better than the random walk at 1 day: "
                         + ", ".join(beat.model.tolist()) + ".")

    # H2: ARIMAX no better than ARIMA
    dd = dm[(dm.model == "ARIMAX") & (dm.vs == "ARIMA")] if not dm.empty else dm
    if dd is None or dd.empty:
        out["H2"] = ("pending", "Pending", "Needs ARIMAX forecasts: results/predictions/baseline_arimax_h<h>.csv.")
    else:
        better = dd[(dd.DM < 0) & (dd.p_value < alpha)]
        out["H2"] = (("yes", "Supported", f"ARIMAX never significantly beats ARIMA ({len(dd)} horizons).") if better.empty else
                     ("no", "Not supported", f"ARIMAX significantly beats ARIMA at h = {', '.join(map(str, better.horizon))}."))

    # H3: image CNN direction better than a coin flip (binomial test on each horizon)
    rows = []
    for h in horizons():
        for model, feats in configs(h):
            if model not in ("imgcnn", "cnn"):
                continue
            p = forecasts(model, feats, h)
            if p is None:
                continue
            step = p.iloc[::max(1, h)]                       # non-overlapping targets
            hit = np.sign(step.pred_price - step.base_price) == np.sign(step.true_price - step.base_price)
            k, n = int(hit.sum()), int(len(hit))
            rows.append((h, k / n, stats.binomtest(k, n, 0.5, alternative="greater").pvalue))
    if not rows:
        out["H3"] = ("pending", "Pending", "Needs image-CNN forecasts: results/predictions/imgcnn_price_only_h<h>_s<seed>.csv.")
    else:
        sig = [r for r in rows if r[2] < alpha]
        txt = "; ".join(f"h = {h}: {a:.1%} (p = {p:.3f})" for h, a, p in rows)
        out["H3"] = (("yes", "Supported", "Direction accuracy " + txt) if len(sig) == len(rows) else
                     ("mixed", "Mixed", "Direction accuracy " + txt) if sig else
                     ("no", "Not supported", "Direction accuracy " + txt))

    # H4: 2026, satellite supply vs price only
    c26 = comparison_2026()
    if c26.empty or not {"price_only", "satellite"} <= set(c26.features):
        out["H4"] = ("pending", "Pending", "Needs 2026 forecasts with and without the satellite series in "
                                           "results/predictions_2026/.")
    else:
        wins, sig, pairs = 0, 0, 0
        d26 = dm_tests_2026()
        for (model, h), g in c26.groupby(["model", "horizon"]):
            if {"price_only", "satellite"} <= set(g.features):
                pairs += 1
                a = float(g[g.features == "satellite"].RMSE.iloc[0])
                b = float(g[g.features == "price_only"].RMSE.iloc[0])
                wins += int(a < b)
                r = d26[(d26.model == config_label(model, "satellite")) & (d26.vs == config_label(model, "price_only"))
                        & (d26.horizon == h)] if not d26.empty else d26
                sig += int(r is not None and not r.empty and r.DM.iloc[0] < 0 and r.p_value.iloc[0] < alpha)
        kind = "yes" if wins == pairs and sig else "mixed" if wins else "no"
        out["H4"] = (kind, {"yes": "Supported", "mixed": "Mixed", "no": "Not supported"}[kind],
                     f"Satellite supply has the lower 2026 RMSE in {wins} of {pairs} comparisons ({sig} significant).")
    return out


def negative_price_step() -> str:
    """How the pipeline treats WTI's negative print, in words (follows config.yaml)."""
    pol = config()["data"].get("negative_price_policy", "clip")
    return {"clip": f"Floor WTI at ${config()['data'].get('min_price', 1):.0f} on 20 April 2020",
            "interpolate": "Interpolate WTI's negative print on 20 April 2020",
            "drop": "Drop 20 April 2020 (WTI below zero)",
            "keep": "Keep WTI's negative print on 20 April 2020"}.get(pol, f"Negative price policy: {pol}")


# -----------------------------------------------------------------------------
# 2026 forecasts (results/predictions_2026/, same file names and columns as results/predictions/)
# -----------------------------------------------------------------------------
@st.cache_data
def catalog_2026() -> pd.DataFrame:
    rows = []
    for f in sorted(PRED_2026_DIR.glob("*.csv")) if PRED_2026_DIR.exists() else []:
        m = RUN_RE.match(f.stem)
        if m:
            rows.append({"model": m[1], "features": m[2], "horizon": int(m[3]), "seed": int(m[4]), "stem": f.stem})
    return pd.DataFrame(rows, columns=["model", "features", "horizon", "seed", "stem"])


@st.cache_data
def forecasts_2026(model: str, features: str, horizon: int) -> pd.DataFrame | None:
    c = catalog_2026()
    c = c[(c.model == model) & (c.features == features) & (c.horizon == horizon)]
    if c.empty:
        return None
    frames = [pd.read_csv(PRED_2026_DIR / f"{s}.csv", parse_dates=["origin_date", "target_date"]) for s in c["stem"]]
    out = frames[0][["origin_date", "target_date", "base_price", "true_price"]].copy()
    preds = np.column_stack([f["pred_price"].to_numpy() for f in frames])
    out["pred_price"] = preds.mean(axis=1)
    out["n_seeds"] = preds.shape[1]
    return out


@st.cache_data
def comparison_2026() -> pd.DataFrame:
    """Metrics on the 2026 days that EVERY 2026 forecast covers (per horizon), plus the random walk
    (forecast = price at the origin), so all rows are compared like for like."""
    c = catalog_2026()
    rows = []
    for h in sorted(c.horizon.unique()) if not c.empty else []:
        pairs = sorted({(r.model, r.features) for r in c[c.horizon == h].itertuples()},
                       key=lambda p: (MODEL_ORDER.get(p[0], 9), FEATURE_ORDER.get(p[1], 9)))
        data = {p: forecasts_2026(*p, h) for p in pairs}
        common = set.intersection(*[set(d.origin_date) for d in data.values()])
        ref = None
        for (model, feats), d in data.items():
            d = d[d.origin_date.isin(common)].sort_values("origin_date")
            ref = d if ref is None else ref
            rows.append({"horizon": h, "model": model, "features": feats, "config": config_label(model, feats),
                         "n_days": len(d), **all_metrics(d.true_price, d.pred_price, d.base_price)})
        rw = all_metrics(ref.true_price, ref.base_price, ref.base_price)
        rw["DirAcc"] = np.nan
        rows.append({"horizon": h, "model": "rw", "features": "price_only", "config": "Random walk",
                     "n_days": len(ref), **rw})
        for r in rows:
            if r["horizon"] == h:
                r["Skill"] = 100 * (1 - r["RMSE"] / rw["RMSE"])
    return pd.DataFrame(rows)


@st.cache_data
def dm_tests_2026() -> pd.DataFrame:
    """Each input set against price only for the same network and horizon, on common 2026 days."""
    c = catalog_2026()
    rows = []
    for (model, h), g in c.groupby(["model", "horizon"]) if not c.empty else []:
        feats = sorted(set(g.features))
        if "price_only" not in feats:
            continue
        base = forecasts_2026(model, "price_only", h)
        for f in feats:
            if f == "price_only":
                continue
            j = forecasts_2026(model, f, h).merge(base, on="origin_date", suffixes=("", "_b"))
            r = diebold_mariano(j.true_price, j.pred_price, j.pred_price_b, h=h)
            rows.append({"horizon": h, "model": config_label(model, f), "vs": config_label(model, "price_only"),
                         "DM": r["DM"], "p_value": r["p_value"], "n": len(j)})
    return pd.DataFrame(rows, columns=["horizon", "model", "vs", "DM", "p_value", "n"])


# -----------------------------------------------------------------------------
# Satellite track (Code/satellite): written by s04-s07
# -----------------------------------------------------------------------------
def _csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


@st.cache_data
def sat_series() -> pd.DataFrame | None:
    return _csv(SAT_DIR / "data" / "series" / "supply_by_scene.csv", parse_dates=["date"])


@st.cache_data
def sat_vs_ais() -> pd.DataFrame | None:
    return _csv(SAT_DIR / "data" / "series" / "satellite_vs_ais_weekly.csv", index_col=0, parse_dates=True)


@st.cache_data
def sat_calibration() -> dict:
    f = SAT_DIR / "data" / "series" / "calibration.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data
def portwatch_hormuz() -> pd.DataFrame | None:
    """Daily AIS transits through Hormuz saved by s07 (or by the s00 check)."""
    df = _csv(SAT_DIR / "data" / "series" / "portwatch_hormuz_daily.csv")
    if df is None:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date")


@st.cache_data
def sat_features() -> pd.DataFrame | None:
    """Business-day features for the price model, written by s07 (Code/data/processed/)."""
    return _csv(CODE_DIR / "data" / "processed" / "satellite_features.csv", index_col=0, parse_dates=True)


@st.cache_data
def detector_runs() -> pd.DataFrame:
    """One row per trained ship detector (Code/satellite/runs/<tag>/metrics.json)."""
    rows = []
    for f in sorted((SAT_DIR / "runs").glob("*/metrics.json")) if (SAT_DIR / "runs").exists() else []:
        d = json.loads(f.read_text())
        a = d.get("args", {})
        row = {"run": f.parent.name, "backbone": a.get("backbone"), "sources": "+".join(a.get("sources", [])),
               "init": a.get("init") or "", "epochs": d.get("best_epoch"), "n_train": d.get("n_train")}
        for split in ("val", "test"):
            for k in ("ap", "precision", "recall", "f1", "count_mae"):
                row[f"{split}_{k}"] = d.get(split, {}).get(k)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def detector_history(run: str) -> pd.DataFrame | None:
    return _csv(SAT_DIR / "runs" / run / "history.csv")


def sat_figure(name: str) -> Path | None:
    f = SAT_DIR / "figures" / name
    return f if f.exists() else None


def sat_quicklooks() -> list[Path]:
    d = SAT_DIR / "figures" / "quicklooks"
    return sorted(d.glob("*.png")) if d.exists() else []


# -----------------------------------------------------------------------------
# Price charts as images (slide 9): the same drawing the image CNN is trained on
# -----------------------------------------------------------------------------
def chart_image(prices: np.ndarray, height: int = 64, width: int = 64, line: int = 1) -> np.ndarray:
    """Draw a price path as a black-and-white image (1 = ink): x is time, y is the min-max scaled price.
    Consecutive points are joined with straight segments so the chart has no gaps."""
    p = np.asarray(prices, dtype=float)
    p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    img = np.zeros((height, width), dtype=np.uint8)
    xs = np.linspace(0, width - 1, len(p))
    ys = (1 - p) * (height - 1)
    for i in range(len(p) - 1):
        n = int(max(abs(xs[i + 1] - xs[i]), abs(ys[i + 1] - ys[i]))) * 2 + 2
        xx = np.linspace(xs[i], xs[i + 1], n).round().astype(int)
        yy = np.linspace(ys[i], ys[i + 1], n).round().astype(int)
        for dy in range(-(line // 2), line // 2 + 1):
            img[np.clip(yy + dy, 0, height - 1), xx] = 1
    return img


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
        if m and m[1] in ("lstm", "gru"):                 # the lab rebuilds RNNForecaster checkpoints only
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
