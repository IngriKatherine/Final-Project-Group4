"""The data set: sources, an explorer for every series, statistics and correlations."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "04_data"
theme.header(KEY)

lv = L.levels()
ret = L.returns()
tr_end, va_end = L.split_dates()
n_train = int((lv.index <= tr_end).sum())
n_val = int(((lv.index > tr_end) & (lv.index <= va_end)).sum())
n_test = int((lv.index > va_end).sum())

theme.figures([
    (f"{lv.shape[1]}", "series (1 target, 12 candidate drivers)"),
    (f"{len(lv):,}", f"business days, {lv.index[0]:%b %Y} to {lv.index[-1]:%b %Y}"),
    (f"{n_train:,}", f"training days (to {tr_end:%Y})"),
    (f"{n_val:,}", f"validation days ({tr_end.year + 1} to {va_end:%Y})"),
    (f"{n_test:,}", f"test days ({va_end.year + 1} to {lv.index[-1]:%Y})"),
])

t_src, t_explore, t_stats, t_corr = st.tabs(["Sources", "Explore the series", "Statistics", "Correlations"])

with t_src:
    src = L.series_sources()
    st.dataframe(src, hide_index=True, width="stretch", height=36 * (len(src) + 1) + 4,
                 column_config={"Raw obs.": st.column_config.NumberColumn(format="%d"),
                                "Used obs.": st.column_config.ProgressColumn(
                                    "Used obs.", format="%d", min_value=0, max_value=int(src["Raw obs."].max())),
                                })
    late = src.sort_values("First date").iloc[-1]
    st.caption(f"The common sample starts when the latest series begins: {late['Series']} "
               f"({late['ID / ticker']}) from {late['First date']}. Gaps of up to 5 days (holidays, "
               "different exchange calendars) are forward-filled.")

with t_explore:
    c1, c2 = st.columns([2, 1])
    with c1:
        picked = st.multiselect("Series", list(lv.columns), default=["WTI", "BRENT", "XLE", "DXY"],
                                format_func=L.label, max_selections=6)
    with c2:
        view = st.segmented_control("Show", ["Indexed to 100", "Levels", "Daily change"], default="Indexed to 100",
                                    help="Indexed: every series starts at 100, so they share one axis. "
                                         "Daily change: what the network actually sees (log returns, or "
                                         "first differences for interest rates).")
    if not picked:
        st.info("Pick at least one series.")
    else:
        palette = [theme.BLUE, theme.ORANGE, theme.AQUA, theme.YELLOW, "#e87ba4", "#008300"]
        if view == "Levels" and len({L.FEATURE_UNITS[c] for c in picked}) > 1:
            st.caption("These series have different units, so each gets its own small chart.")
            cols = st.columns(min(len(picked), 3))
            for i, c in enumerate(picked):
                fig = go.Figure(go.Scatter(x=lv.index, y=lv[c], mode="lines", line=dict(color=palette[i], width=1.4),
                                           hovertemplate="%{x|%d %b %Y}: %{y:.2f}<extra></extra>"))
                theme.add_split_bands(fig, labels=False)
                fig.update_layout(title=f"{L.label(c)} ({L.FEATURE_UNITS[c]})", showlegend=False)
                with cols[i % len(cols)]:
                    theme.show(fig, height=240)
        else:
            fig = go.Figure()
            for i, c in enumerate(picked):
                if view == "Indexed to 100":
                    y = 100 * lv[c] / lv[c].iloc[0]
                    unit = "index, first day = 100"
                elif view == "Levels":
                    y, unit = lv[c], L.FEATURE_UNITS[c]
                else:
                    y = ret[c] * (1 if c in L.RATE_COLUMNS else 100)
                    unit = "% change (rates: change in points)"
                fig.add_trace(go.Scatter(x=y.index, y=y, name=L.label(c), mode="lines",
                                         line=dict(color=palette[i], width=1.3 if view != "Daily change" else 0.8),
                                         hovertemplate="%{y:.2f}<extra>" + L.label(c) + "</extra>"))
            theme.add_split_bands(fig)
            fig.update_layout(hovermode="x unified", yaxis_title=unit,
                              yaxis_type="log" if view == "Indexed to 100" and st.session_state.get("logy") else None)
            theme.show(fig, height=430)
            if view == "Indexed to 100":
                st.checkbox("Log scale", key="logy")

with t_stats:
    stats = lv.describe().T
    stats.insert(0, "Series", [L.label(c) for c in stats.index])
    stats["Unit"] = [L.FEATURE_UNITS[c] for c in stats.index]
    r = ret
    stats["Daily volatility"] = [r[c].std() * (1 if c in L.RATE_COLUMNS else 100) for c in stats.index]
    st.dataframe(stats.drop(columns=["count"]).round(2), width="stretch", hide_index=True,
                 column_config={"Daily volatility": st.column_config.NumberColumn(
                     help="Std. of daily log returns in %, or of daily changes in points for rates", format="%.2f")})
    st.caption("Levels after cleaning (minimum WTI of 1.00 is the floored April 2020 price).")

with t_corr:
    corr = L.table("correlations")
    corr = corr.set_index(corr.columns[0]) if corr is not None else ret.corr().round(3)
    order = corr["WTI"].sort_values(ascending=False).index.tolist()
    corr = corr.loc[order, order]
    c1, c2 = st.columns([1.5, 1], gap="large")
    with c1:
        names = [L.label(c) for c in corr.columns]
        fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale=theme.DIVERGING,
                                   xgap=2, ygap=2, colorbar=dict(title="r", thickness=10),
                                   hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
        fig.update_layout(yaxis_autorange="reversed", xaxis_tickangle=-40, xaxis_showgrid=False, yaxis_showgrid=False)
        theme.show(fig, height=520)
    with c2:
        w = corr["WTI"].drop("WTI").sort_values()
        fig = go.Figure(go.Bar(x=w.values, y=[L.label(c) for c in w.index], orientation="h",
                               marker_color=[theme.BLUE if v >= 0 else theme.ORANGE for v in w.values],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Correlation with WTI daily returns", xaxis_title="r", showlegend=False)
        theme.show(fig, height=520)
    top = w.abs().sort_values(ascending=False).index[:2]
    st.caption(f"Correlations of daily log returns (differences for rates) over the full sample, sorted by "
               f"their relation to WTI. Even the strongest, {L.label(top[0])} at {corr.loc[top[0], 'WTI']:.2f} and "
               f"{L.label(top[1])} at {corr.loc[top[1], 'WTI']:.2f}, are weak: no single driver explains much of WTI's daily moves.")

theme.footer(KEY)
