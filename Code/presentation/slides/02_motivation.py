"""Why oil prices: volatile levels, near-memoryless returns, and a strong naive benchmark."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "02_motivation"
theme.header(KEY)

lv = L.levels()
ret = L.returns()

left, right = st.columns([1.7, 1], gap="large")

with left:
    st.subheader(f"WTI and Brent, {lv.index[0]:%Y} to {lv.index[-1]:%Y}")
    events = st.toggle("Mark major events", value=True)
    fig = go.Figure()
    for col in ("WTI", "BRENT"):
        fig.add_trace(go.Scatter(x=lv.index, y=lv[col], name=L.label(col), mode="lines",
                                 line=dict(color=theme.SERIES_COLORS[col], width=1.5),
                                 hovertemplate="%{x|%d %b %Y}: $%{y:.2f}<extra>" + L.label(col) + "</extra>"))
    theme.add_split_bands(fig)
    if events:
        for d, text in [("2008-12-19", "Financial crisis"), ("2014-11-27", "OPEC keeps output up"),
                        ("2020-04-20", "COVID: WTI below zero"), ("2022-03-08", "Russia invades Ukraine"),
                        ("2026-02-28", "Strait of Hormuz closed")]:
            if pd.Timestamp(d) > lv.index[-1]:
                continue
            fig.add_vline(x=pd.Timestamp(d), line=dict(color=theme.MUTED, width=1))
            fig.add_annotation(x=pd.Timestamp(d), y=0.02, yref="paper", text=text, showarrow=False,
                               textangle=-90, xanchor="right", yanchor="bottom", font=dict(size=11, color=theme.INK_2))
    fig.update_layout(hovermode="x unified", yaxis_title="USD per barrel",
                      xaxis=dict(rangeselector=dict(buttons=[
                          dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(count=5, label="5y", step="year", stepmode="backward"),
                          dict(step="all", label="All")], bgcolor=theme.SURFACE, activecolor=theme.CRUDE_SOFT,
                          font=dict(size=12), x=1, xanchor="right", y=1.08)))
    theme.show(fig, height=430)
    st.caption(f"Levels after cleaning ({L.negative_price_step().lower()}; see the preprocessing slide). "
               "Shading marks the train, validation and test periods.")

with right:
    st.subheader("Do daily returns remember?")
    max_lag = st.slider("Lags to show", 5, 30, 15)
    skip_2020 = st.toggle("Leave out March to June 2020", value=True,
                          help="The COVID crash and the negative WTI print produce a handful of extreme daily "
                               "returns that dominate the autocorrelation estimate.")
    r = ret["WTI"].dropna()
    if skip_2020:
        r = r[~((r.index >= "2020-03-01") & (r.index <= "2020-06-30"))]
    acf = [r.autocorr(k) for k in range(1, max_lag + 1)]
    band = 1.96 / np.sqrt(len(r))
    fig = go.Figure(go.Bar(x=list(range(1, max_lag + 1)), y=acf, marker_color=theme.BLUE, name="Autocorrelation",
                           hovertemplate="lag %{x}: %{y:.3f}<extra></extra>"))
    fig.add_hrect(y0=-band, y1=band, fillcolor="rgba(21,33,43,0.08)", line_width=0)
    fig.add_annotation(x=max_lag + 0.4, y=band, text="95% noise band", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=11, color=theme.INK_2))
    lim = max(0.12, float(np.max(np.abs(acf))) * 1.15)
    fig.update_layout(xaxis_title="Lag (business days)", yaxis_title="Autocorrelation",
                      showlegend=False, yaxis_range=[-lim, lim])
    theme.show(fig, height=300)
    st.caption("Autocorrelation of WTI daily log returns. Inside the band means indistinguishable from noise.")

    r_all = ret["WTI"].dropna()
    vol = r_all.std() * np.sqrt(252) * 100
    big = int((r_all.abs() > 0.05).sum())
    theme.figures([(f"{vol:.0f}%", "annualised volatility"), (f"{big}", "days moving more than 5%")])

comp = L.comparison()
rw1 = comp[(comp.model == "rw") & (comp.horizon == 1)]
rw_txt = (f" On the test days it misses by <strong>${rw1.RMSE.iloc[0]:.2f}</strong> per barrel "
          "(RMSE, 1 day ahead).") if not rw1.empty else ""
theme.callout("If returns had memory, yesterday's move would help predict today's. It barely does, so the "
              "<strong>random walk</strong> (tomorrow's price equals today's) is the benchmark every model must beat." + rw_txt)

theme.footer(KEY)
