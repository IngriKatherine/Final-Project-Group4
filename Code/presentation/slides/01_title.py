"""Title slide: the project, the team, the question, and twenty years of WTI."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import content
import loaders as L
import theme

KEY = "01_title"
st.html(theme.rail(KEY))

names = ", ".join(f"<strong>{html.escape(n)}</strong>" for n in content.TEAM)
st.html(f"""
<div class="title-hero">
  <h1>{html.escape(content.PROJECT_TITLE)}</h1>
  <div class="by">{html.escape(content.GROUP)}: {names}<br>{html.escape(content.COURSE)}</div>
</div>
""")

# --- Hero chart: the raw WTI spot price, negative print included -------------------------
fred = L.raw_series("fred")
lv = L.levels()
wti = (fred["WTI"].dropna() if fred is not None else lv["WTI"]).loc[lv.index[0]:]

fig = go.Figure()
fig.add_trace(go.Scatter(x=wti.index, y=wti.values, mode="lines", line=dict(color=theme.INK, width=1.4),
                         name="WTI", hovertemplate="%{x|%d %b %Y}<br>$%{y:.2f} per barrel<extra></extra>"))
theme.add_split_bands(fig, start=wti.index[0], end=wti.index[-1])


def mark(date, text, ay):
    d = wti.index[wti.index.get_indexer([pd.Timestamp(date)], method="nearest")[0]]
    fig.add_annotation(x=d, y=wti[d], text=text, showarrow=True, arrowhead=0, arrowwidth=1,
                       arrowcolor=theme.MUTED, ax=0, ay=ay, font=dict(size=12, color=theme.INK_2),
                       bgcolor="rgba(255,255,255,0.85)")


mark(wti.loc[:"2009"].idxmax(), f"July 2008: ${wti.loc[:'2009'].max():.0f}", -30)
mark(wti.loc["2015":"2016"].idxmin(), "2016 supply glut", 40)
mark(wti.idxmin(), f"20 April 2020: ${wti.min():.2f}", -10 if wti.min() > 0 else 26)
mark(wti.loc["2022"].idxmax(), "2022 invasion spike", -30)
fig.update_layout(height=360, showlegend=False, margin=dict(l=12, r=12, t=24, b=12),
                  yaxis=dict(title="WTI spot, USD per barrel", zeroline=True, zerolinecolor=theme.MUTED),
                  xaxis=dict(showgrid=False))
theme.show(fig)

# --- The question and the setup at a glance -------------------------------------------------
left, right = st.columns([1.35, 1], gap="large")
with left:
    st.html(f'<div class="callout"><p class="question">{html.escape(content.RESEARCH_QUESTION)}</p></div>')
    st.write("We compare three input sets for an LSTM and a GRU: the oil price alone, the price plus every "
             "candidate driver, and the price plus the few drivers that SHAP ranks highest. Each is judged "
             "against the random walk and ARIMA on the same 2022 to 2025 test days.")
with right:
    cfg = L.config()
    tr, va = L.split_dates()
    rows = [
        ("Target", "WTI spot price, USD per barrel"),
        ("Inputs", f"{lv.shape[1]} daily series from FRED and Yahoo Finance"),
        ("Sample", f"{lv.index[0]:%b %Y} to {lv.index[-1]:%b %Y}, {len(lv):,} business days"),
        ("Split", f"train to {tr.year}, validation to {va.year}, test after"),
        ("Horizons", ", ".join(f"{h}" for h in cfg["framing"]["horizons"]) + " business days ahead"),
        ("Networks", "stacked LSTM and GRU in PyTorch"),
        ("Explainer", "SHAP expected gradients on the trained LSTM"),
    ]
    st.html('<dl class="spec">' + "".join(f"<dt>{k}</dt><dd>{html.escape(v)}</dd>" for k, v in rows) + "</dl>")

theme.footer(KEY)
