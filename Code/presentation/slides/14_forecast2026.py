"""2026 forecasts with the satellite oil-supply series: the data, the input sets, and the 2026 scores
(results/predictions_2026/, compared on the days every model covers, against the random walk)."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import loaders as L
import theme

KEY = "14_forecast2026"
theme.header(KEY)

CLOSURE = pd.Timestamp("2026-02-28")
cfg = L.config()
target = cfg["data"]["target"]
feats = L.sat_features()
lv = L.levels()

# --- 1. Price and supply in 2026 -----------------------------------------------------------------
st.subheader("Price and oil supply through Hormuz")
price_src = feats if feats is not None and target in feats and feats[target].notna().any() else lv
price = price_src[target].dropna() if target in price_src else pd.Series(dtype=float)
price = price[price.index >= "2025-07-01"]
fig = make_subplots(specs=[[{"secondary_y": True}]])
if len(price):
    fig.add_trace(go.Scatter(x=price.index, y=price.values, name=f"{L.label(target)} (left)", mode="lines",
                             line=dict(color=theme.INK, width=2)), secondary_y=False)
if feats is not None:
    f = feats[feats.index >= "2025-07-01"]
    if "HZ_SAT_FLOW_MBD" in f:
        fig.add_trace(go.Scatter(x=f.index, y=f.HZ_SAT_FLOW_MBD, name="Satellite supply, mb/d (right)",
                                 line=dict(color=theme.AQUA, width=2)), secondary_y=True)
    if "HZ_AIS_FLOW_MBD" in f:
        fig.add_trace(go.Scatter(x=f.index, y=f.HZ_AIS_FLOW_MBD, name="AIS supply, mb/d (right)",
                                 line=dict(color=theme.PINK, width=1.4, dash="dot")), secondary_y=True)
fig.add_vline(x=CLOSURE, line=dict(color=theme.ORANGE, width=1, dash="dash"))
fig.update_yaxes(title_text="USD per barrel", secondary_y=False)
fig.update_yaxes(title_text="Million barrels per day", secondary_y=True, showgrid=False)
fig.update_layout(hovermode="x unified")
if len(fig.data):
    theme.show(fig, height=360)
msgs = []
if price.empty or price.index.max() < pd.Timestamp("2026-03-01"):
    msgs.append(("2026 prices are not in the data yet. In Code/config.yaml set data.end_date to the latest date, "
                 "then rerun the download and preprocessing:", "python 01_fetch_data.py\npython 02_preprocess.py"))
if feats is None:
    msgs.append(("The satellite features are not built yet:",
                 "cd satellite\npython s07_calibrate_and_export.py   # writes data/processed/satellite_features.csv"))
for what, cmd in msgs:
    theme.pending(what, cmd)

# --- 2. How the networks use it ------------------------------------------------------------------
st.subheader("Four input sets, one held-out year")
sets = [
    ("price_only", "Price only", "The oil price's own history (the reference)."),
    ("ais", "Price + AIS supply", "Adds PortWatch tanker flow: what ship tracking sees (HZ_AIS_FLOW_MBD)."),
    ("satellite", "Price + satellite supply", "Adds the CNN's supply series: flow through the strait, oil waiting "
                                              "at anchor, large-ship count (HZ_SAT_*)."),
    ("satellite_ais", "Price + satellite + AIS", "Both, so the network can use the gap between them."),
]
cols = st.columns(len(sets), gap="small")
for col, (k, name, desc) in zip(cols, sets):
    col.html(f'<div style="border-left:4px solid {theme.FEATURE_SET_COLORS[k]};padding-left:.7rem">'
             f'<strong>{html.escape(name)}</strong><br><span style="color:{theme.INK_2};font-size:.9rem">'
             f'{html.escape(desc)}</span></div>')
st.caption("Train 2023-01 to 2025-06, validate 2025-07 to 2025-12, test every day of 2026. The supply features "
           "are causal: on day t only images taken on or before t are used.")

# --- 3. The 2026 scores -------------------------------------------------------------------------
st.subheader("2026 test results")
c26 = L.comparison_2026()
if c26.empty:
    theme.pending("No 2026 forecasts yet. Save them in `results/predictions_2026/` with the same names and columns "
                  "as results/predictions, e.g. `lstm_satellite_h5_s0.csv`, `gru_price_only_h5_s1.csv` (input sets: "
                  "price_only, ais, satellite, satellite_ais). The random walk is computed here from base_price.",
                  "python 06_run_experiments.py   # with the 2026 split and the HZ_* features (see satellite/README.md)")
else:
    hs = sorted(c26.horizon.unique())
    h = st.segmented_control("Horizon", hs, default=hs[0], required=True, key="h26",
                             format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""))
    d = c26[c26.horizon == h].sort_values("RMSE")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        fig = go.Figure(go.Bar(x=d.Skill, y=d.config, orientation="h",
                               marker_color=[theme.config_color(m, f) for m, f in zip(d.model, d.features)],
                               hovertemplate="%{y}: %{x:+.1f}%<extra></extra>"))
        fig.add_vline(x=0, line=dict(color=theme.INK, width=1))
        fig.update_layout(xaxis_title="Skill vs. random walk, 2026 (%)", yaxis=dict(autorange="reversed"))
        theme.show(fig, height=60 + 38 * len(d))
    with b:
        st.dataframe(d[["config", "n_days", "RMSE", "MAE", "DirAcc", "Skill"]], hide_index=True, width="stretch",
                     column_config={"config": "Model / input set", "n_days": "Days",
                                    "RMSE": st.column_config.NumberColumn(format="%.3f"),
                                    "MAE": st.column_config.NumberColumn(format="%.3f"),
                                    "DirAcc": st.column_config.NumberColumn("Direction", format="%.3f"),
                                    "Skill": st.column_config.NumberColumn("Skill %", format="%+.1f")})
    dm = L.dm_tests_2026()
    dm = dm[dm.horizon == h]
    if not dm.empty:
        st.markdown("**Does the extra input help? (Diebold-Mariano against price only, same network)**")
        st.dataframe(dm.drop(columns="horizon").style.format({"DM": "{:+.2f}", "p_value": "{:.4f}"}),
                     hide_index=True, width="stretch",
                     column_config={"model": "Model", "vs": "Against", "p_value": "p-value", "n": "Days"})
        st.caption("A negative statistic means the first model is more accurate.")

    cat = L.catalog_2026()
    pairs = sorted({(r.model, r.features) for r in cat[cat.horizon == h].itertuples()},
                   key=lambda p: (L.MODEL_ORDER.get(p[0], 9), L.FEATURE_ORDER.get(p[1], 9)))
    labels = {L.config_label(*p): p for p in pairs}
    chosen = st.multiselect("Forecasts to plot", list(labels), default=list(labels)[:3], key="pick26")
    fig = go.Figure()
    ref = None
    for k in chosen:
        p = L.forecasts_2026(*labels[k], h)
        ref = p if ref is None else ref
        fig.add_trace(go.Scatter(x=p.target_date, y=p.pred_price, name=k, mode="lines",
                                 line=dict(color=theme.config_color(*labels[k]), width=1.5,
                                           dash=theme.config_dash(labels[k][0]))))
    if ref is not None:
        fig.add_trace(go.Scatter(x=ref.target_date, y=ref.true_price, name="Actual", mode="lines",
                                 line=dict(color=theme.INK, width=2.2)))
        fig.add_vline(x=CLOSURE, line=dict(color=theme.ORANGE, width=1, dash="dash"))
        fig.update_layout(hovermode="x unified", yaxis_title="USD per barrel")
        theme.show(fig, height=360)

theme.footer(KEY)
