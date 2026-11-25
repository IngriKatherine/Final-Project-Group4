"""Preprocessing and the supervised framing: pipeline, the negative price, and the sliding window."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "05_preprocessing"
theme.header(KEY)

cfg = L.config()
lv = L.levels()
ret = L.returns()
tr_end, va_end = L.split_dates()

left, right = st.columns([1, 1.25], gap="large")

with left:
    st.subheader("From two downloads to model inputs")
    st.graphviz_chart(f"""
    digraph {{
      rankdir=TB; nodesep=0.25; ranksep=0.22;
      node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{theme.RULE}",
            fontname="Helvetica", fontsize=11, fontcolor="{theme.INK}", margin="0.18,0.07"];
      edge [color="{theme.MUTED}", arrowsize=0.6];
      a [label="FRED (7 series)  +  Yahoo Finance (6 tickers)"];
      b [label="Align to business days, forward-fill gaps of 5 days or less"];
      c [label="{L.negative_price_step()}", fillcolor="{theme.CRUDE_SOFT}"];
      d [label="Keep the period where all 13 series exist: 2006 to 2025"];
      e [label="Log returns for prices, first differences for rates"];
      f [label="Standardise with means and std. from training rows only", fillcolor="{theme.CRUDE_SOFT}"];
      g [label="Slide a {cfg['framing']['lookback']}-day window; target = log(P[t+h] / P[t])"];
      h [label="Assign each window to train / val / test by its target date", fillcolor="{theme.CRUDE_SOFT}"];
      a -> b -> c -> d -> e -> f -> g -> h;
    }}""", width="stretch")
    st.caption("Shaded steps are the ones that guard against look-ahead leakage or broken math.")

with right:
    st.subheader("The day WTI went negative")
    fred = L.raw_series("fred")
    if fred is not None:
        raw = fred["WTI"].dropna().loc["2020-03-01":"2020-06-15"]
        clean = lv["WTI"].loc["2020-03-01":"2020-06-15"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=raw.index, y=raw, name="Raw FRED price", mode="lines+markers",
                                 line=dict(color=theme.MUTED, width=1.5), marker=dict(size=5),
                                 hovertemplate="%{x|%d %b}: $%{y:.2f}<extra>raw</extra>"))
        fig.add_trace(go.Scatter(x=clean.index, y=clean, name="After cleaning", mode="lines",
                                 line=dict(color=theme.BLUE, width=2),
                                 hovertemplate="%{x|%d %b}: $%{y:.2f}<extra>cleaned</extra>"))
        low = raw.idxmin()
        fig.add_annotation(x=low, y=raw.min(), text=f"{low:%d %b}: ${raw.min():.2f}", showarrow=True, arrowhead=0,
                           ax=70, ay=-10, font=dict(size=12, color=theme.INK))
        fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1))
        fig.update_layout(yaxis_title="USD per barrel", hovermode="x unified")
        theme.show(fig, height=300)
    r = ret["WTI"]
    pol = cfg["data"].get("negative_price_policy", "clip")
    worst = r.loc["2020-04-01":"2020-05-31"].abs().max()
    if pol == "clip":
        why = (f"A log return needs a positive price. Flooring at ${cfg['data']['min_price']:.0f} keeps the series "
               f"usable but creates extreme daily log returns around 20 April (largest: {worst:.2f}).")
    else:
        why = ("A log return needs a positive price. Replacing the one negative day with a time interpolation of "
               f"its neighbours keeps the series usable without inventing a crash: the largest daily log return in "
               f"April-May 2020 is now {worst:.2f}, where a typical day is about {r.abs().median():.2f}.")
    st.write(why + " The day falls in the validation period, so it affects early stopping but not the test score.")

st.subheader("What one training example looks like")
st.write("Pick a forecast origin. The network reads the shaded window of daily changes for every input and "
         "predicts one number: the log return from the origin to the target day.")

c1, c2, c3 = st.columns([1, 1, 2])
lookback = c1.select_slider("Lookback window (days)", [10, 20, 30, 45, 60], value=cfg["framing"]["lookback"])
h = c2.segmented_control("Horizon", cfg["framing"]["horizons"], default=cfg["framing"]["horizons"][0],
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""), required=True)
dates = lv.index
origin = c3.select_slider("Forecast origin", options=list(dates[lookback:-h]), value=dates[dates.get_indexer(
    [pd.Timestamp("2021-12-15")], method="nearest")[0]], format_func=lambda d: f"{d:%d %b %Y}")

pos = dates.get_loc(origin)
win = dates[pos - lookback + 1: pos + 1]
target_day = dates[pos + h]
view = lv["WTI"].iloc[max(0, pos - lookback - 25): pos + h + 25]
split = "train" if target_day <= tr_end else "validation" if target_day <= va_end else "test"

fig = go.Figure()
fig.add_vrect(x0=win[0], x1=win[-1], fillcolor="rgba(42,120,214,0.12)", line_width=0,
              annotation_text=f"input: {lookback} days x {lv.shape[1]} features", annotation_position="top left",
              annotation_font=dict(size=12, color=theme.INK_2))
fig.add_trace(go.Scatter(x=view.index, y=view, mode="lines", line=dict(color=theme.INK, width=1.5), name="WTI",
                         hovertemplate="%{x|%d %b %Y}: $%{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=[origin], y=[lv.loc[origin, "WTI"]], mode="markers", name="Origin P[t]",
                         marker=dict(size=11, color=theme.BLUE, line=dict(color="#fff", width=2))))
fig.add_trace(go.Scatter(x=[target_day], y=[lv.loc[target_day, "WTI"]], mode="markers", name=f"Target P[t+{h}]",
                         marker=dict(size=12, color=theme.CRUDE, symbol="diamond", line=dict(color="#fff", width=2))))
for b, name in ((tr_end, "train | validation"), (va_end, "validation | test")):
    if view.index[0] <= b <= view.index[-1]:
        fig.add_vline(x=b, line=dict(color=theme.MUTED, width=1, dash="dot"))
        fig.add_annotation(x=b, y=0, yref="paper", text=name, showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=11, color=theme.INK_2))
fig.update_layout(yaxis_title="WTI, USD per barrel", hovermode="closest")
theme.show(fig, height=330)

y = np.log(lv.loc[target_day, "WTI"] / lv.loc[origin, "WTI"])
theme.figures([
    (f"{origin:%d %b %Y}", "origin t (last day the network sees)"),
    (f"{target_day:%d %b %Y}", f"target day t+{h}"),
    (f"{y:+.2%}", "target: log return to predict"),
    (split.capitalize(), "split, decided by the target date"),
])
st.caption("Assigning by target date means no training target lies inside the validation or test period, even when "
           "the input window does. Scalers are fit on training rows only, and the prediction is turned back into "
           "a price with P[t] x exp(predicted return), so every error is in USD per barrel.")

theme.footer(KEY)
