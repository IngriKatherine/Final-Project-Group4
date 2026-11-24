"""Benchmarks: random walk and rolling ARIMA on the same test origins."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "06_baselines"
theme.header(KEY)

cfg = L.config()
comp = L.comparison()
base = comp[comp.model.isin(["rw", "arima"])]

left, right = st.columns([1, 1.3], gap="large")
with left:
    st.subheader("Two baselines")
    st.markdown("**Random walk.** Tomorrow's price is today's price.")
    st.latex(r"\hat P_{t+h} = P_t")
    order = tuple(cfg["baselines"]["arima_order"])
    st.markdown(f"**ARIMA{order}** on log prices, rolling origin: refit every "
                f"{cfg['baselines']['refit_every']} test days and updated with each new observation, so it only "
                "ever uses data up to the forecast origin.")
    st.latex(r"\Delta \log P_t = c + \phi\, \Delta \log P_{t-1} + \varepsilon_t + \theta\, \varepsilon_{t-1}")

    if base.empty:
        theme.pending("No baseline forecasts yet.", "python 03_baselines.py")
    else:
        show = base.pivot_table(index="config", columns="horizon", values="RMSE").reindex(["Random walk", "ARIMA"])
        show.columns = [f"h = {c}" for c in show.columns]
        show.index.name = "Model"
        st.markdown("**Test RMSE, USD per barrel**")
        st.dataframe(show.round(3), width="stretch")

with right:
    if not base.empty:
        metric = st.segmented_control("Metric", ["RMSE", "MAE", "MAPE"], default="RMSE", required=True,
                                      help=L.METRIC_HELP["RMSE"])
        fig = go.Figure()
        for model in ("rw", "arima"):
            d = base[base.model == model].sort_values("horizon")
            fig.add_trace(go.Scatter(x=d.horizon, y=d[metric], name=L.MODEL_LABELS[model], mode="lines+markers",
                                     line=dict(color=theme.BASELINE_COLORS[model], width=2, dash=theme.config_dash(model)),
                                     marker=dict(size=9), hovertemplate="h = %{x}: %{y:.3f}<extra>%{fullData.name}</extra>"))
        h_max = int(base.horizon.max())
        rw1 = base[(base.model == "rw") & (base.horizon == base.horizon.min())][metric].iloc[0]
        hs = np.linspace(base.horizon.min(), h_max, 50)
        fig.add_trace(go.Scatter(x=hs, y=rw1 * np.sqrt(hs / base.horizon.min()), name="RW at 1 day x sqrt(h)",
                                 mode="lines", line=dict(color=theme.MUTED, width=1, dash="dot"), hoverinfo="skip"))
        fig.update_layout(xaxis=dict(title="Forecast horizon h (business days)", tickvals=sorted(base.horizon.unique())),
                          yaxis_title=f"Test {metric}" + (" (%)" if metric == "MAPE" else " (USD/bbl)"))
        theme.show(fig, height=330)
        st.caption("For a pure random walk the error grows like the square root of the horizon (dotted line); "
                   "growth slower than that hints at mean reversion over longer horizons.")

if not base.empty:
    st.subheader("Rolling ARIMA vs. no change, up close")
    h = st.segmented_control("Horizon", sorted(base.horizon.unique()), default=int(base.horizon.min()),
                             format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""), required=True, key="bh")
    rw, ar = L.forecasts("rw", "price_only", h), L.forecasts("arima", "price_only", h)
    if rw is not None and ar is not None:
        start, end = st.select_slider("Window", options=list(rw.target_date),
                                      value=(rw.target_date.iloc[0], rw.target_date.iloc[min(120, len(rw) - 1)]),
                                      format_func=lambda d: f"{d:%b %Y}")
        m = (rw.target_date >= start) & (rw.target_date <= end)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rw.target_date[m], y=rw.true_price[m], name="Actual", mode="lines",
                                 line=dict(color=theme.INK, width=2)))
        fig.add_trace(go.Scatter(x=rw.target_date[m], y=rw.pred_price[m], name="Random walk", mode="lines",
                                 line=dict(color=theme.BASELINE_COLORS["rw"], width=1.6)))
        fig.add_trace(go.Scatter(x=ar.target_date[m], y=ar.pred_price[m], name="ARIMA", mode="lines",
                                 line=dict(color=theme.BASELINE_COLORS["arima"], width=1.6, dash="dash")))
        fig.update_layout(hovermode="x unified", yaxis_title="USD per barrel")
        theme.show(fig, height=320)
        diff = (ar.pred_price - rw.pred_price).abs()
        st.caption(f"ARIMA's forecast differs from the random walk by ${diff.mean():.2f} on average at this horizon: "
                   "with almost no autocorrelation to exploit, it stays close to 'no change'. The random walk "
                   "never predicts a direction, so directional accuracy is not reported for it.")

theme.footer(KEY)
