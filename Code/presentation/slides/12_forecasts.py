"""Forecast explorer: actual vs. predicted in the test period, rolling error, and a single-day inspector."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "12_forecasts"
theme.header(KEY)

hs = L.horizons()
c1, c2 = st.columns([1, 3])
h = c1.segmented_control("Horizon", hs, default=hs[0], required=True,
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""))
cfgs = L.configs(h)
labels = {L.config_label(*c): c for c in cfgs}
default = [k for k in labels if labels[k][0] in ("rw", "lstm", "gru")][:4]
chosen = c2.multiselect("Models", list(labels), default=default)

data = {k: L.forecasts(*labels[k], h) for k in chosen}
data = {k: v for k, v in data.items() if v is not None}
if not data:
    st.info("Pick at least one model.")
    theme.footer(KEY)
    st.stop()

ref = next(iter(data.values()))
first, last = ref.target_date.min(), ref.target_date.max()
periods = {"Whole test period": (first, last), "2022 spike": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
           "2023": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31")),
           "2024 to 2025": (pd.Timestamp("2024-01-01"), last)}
period = st.pills("Zoom", list(periods), default="Whole test period", required=True)
a, b = periods[period]

fig = go.Figure()
fig.add_trace(go.Scatter(x=ref.target_date, y=ref.true_price, name="Actual", mode="lines",
                         line=dict(color=theme.INK, width=2.2)))
for k, d in data.items():
    m, f = labels[k]
    if d["n_seeds"].iloc[0] > 1:
        fig.add_trace(go.Scatter(x=pd.concat([d.target_date, d.target_date[::-1]]),
                                 y=pd.concat([d.pred_price + d.pred_std, (d.pred_price - d.pred_std)[::-1]]),
                                 fill="toself", fillcolor="rgba(235,104,52,0.12)" if f == "all" else
                                 "rgba(42,120,214,0.12)" if f == "price_only" else "rgba(27,175,122,0.12)",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=k))
    fig.add_trace(go.Scatter(x=d.target_date, y=d.pred_price, name=k, mode="lines", legendgroup=k,
                             line=dict(color=theme.config_color(m, f), width=1.4, dash=theme.config_dash(m))))
fig.update_layout(hovermode="x unified", yaxis_title="WTI, USD per barrel",
                  xaxis=dict(range=[a, b], rangeslider=dict(visible=True, thickness=0.06, bgcolor="#EEF1F3")))
theme.show(fig, height=450)
st.caption(f"{h}-day-ahead forecasts plotted on the day they target. Bands show ±1 std. across seeds. "
           "Drag the slider under the chart to pan.")

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.subheader("When do the models struggle?")
    win = st.slider("Rolling window (business days)", 20, 120, 60, 10)
    fig = go.Figure()
    for k, d in data.items():
        m, f = labels[k]
        r = np.sqrt((d.set_index("target_date")["error"] ** 2).rolling(win).mean())
        fig.add_trace(go.Scatter(x=r.index, y=r, name=k, mode="lines",
                                 line=dict(color=theme.config_color(m, f), width=1.6, dash=theme.config_dash(m))))
    fig.update_layout(hovermode="x unified", yaxis_title=f"{win}-day rolling RMSE (USD/bbl)")
    theme.show(fig, height=330)
with right:
    st.subheader("Error distribution")
    fig = go.Figure()
    for k, d in data.items():
        m, f = labels[k]
        fig.add_trace(go.Box(x=d.error, name=k, marker_color=theme.config_color(m, f), boxpoints=False,
                             line=dict(width=1.4), orientation="h", hoverinfo="x"))
    fig.add_vline(x=0, line=dict(color=theme.MUTED, width=1))
    fig.update_layout(showlegend=False, xaxis_title="Forecast minus actual (USD/bbl)", yaxis=dict(autorange="reversed"))
    theme.show(fig, height=330)

st.subheader("Inspect one day")
day = st.select_slider("Target date", options=list(ref.target_date), value=ref.target_date.iloc[len(ref) // 3],
                       format_func=lambda d: f"{d:%a %d %b %Y}")
rows = []
for k, d in data.items():
    r = d[d.target_date == day]
    if not r.empty:
        r = r.iloc[0]
        rows.append({"Model": k, "Origin": r.origin_date.date(), "Price at origin": r.base_price,
                     "Forecast": r.pred_price, "Actual": r.true_price, "Error": r.pred_price - r.true_price,
                     "Called direction": "n/a" if labels[k][0] == "rw" else
                     ("yes" if np.sign(r.pred_price - r.base_price) == np.sign(r.true_price - r.base_price) else "no")})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
    c: st.column_config.NumberColumn(format="$%.2f") for c in ("Price at origin", "Forecast", "Actual", "Error")})

theme.footer(KEY)
