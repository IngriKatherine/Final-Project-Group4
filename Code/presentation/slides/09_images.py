"""Price charts as images: build the picture the CNN sees, examples of up and down charts,
the network, and its directional accuracy (Ren et al., 2024)."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

import content
import loaders as L
import theme

KEY = "09_images"
theme.header(KEY)

cfg = L.config()
lv = L.levels()
target = cfg["data"]["target"]
price = lv[target].dropna()
tr_end, va_end = L.split_dates()
IC = content.IMAGE_CNN
H_IMG, W_IMG = IC["size"]


def image_trace(img, name=""):
    return go.Heatmap(z=img, colorscale=[[0, "#FFFFFF"], [1, theme.INK]], showscale=False, zmin=0, zmax=1,
                      hoverinfo="skip", name=name)


# --- 1. Build one image ------------------------------------------------------------------------
st.subheader("From a window of prices to a picture")
c1, c2, c3 = st.columns([1, 1, 2])
window = c1.select_slider("Days in the image", [10, 20, 40, 60], value=IC["window"])
hs = L.horizons()
h = c2.segmented_control("Predict the move over", hs, default=hs[min(1, len(hs) - 1)], required=True,
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""), key="img_h")
dates = price.index[window: len(price) - h]
origin = c3.select_slider("Last day in the image", options=list(dates), value=dates[int(len(dates) * 0.55)],
                          format_func=lambda d: f"{d:%d %b %Y}")
pos = price.index.get_loc(origin)
win = price.iloc[pos - window + 1: pos + 1]
future = price.iloc[pos: pos + h + 1]
move = future.iloc[-1] / future.iloc[0] - 1
img = L.chart_image(win.values, H_IMG, W_IMG)

left, right = st.columns([1.5, 1], gap="large")
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win.index, y=win.values, mode="lines+markers", name="In the image",
                             line=dict(color=theme.INK, width=2), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=future.index, y=future.values, mode="lines+markers", name=f"Next {h} days (label)",
                             line=dict(color=theme.AQUA if move > 0 else theme.ORANGE, width=2, dash="dot"),
                             marker=dict(size=4)))
    fig.update_layout(yaxis_title=f"{L.label(target)}, USD per barrel", hovermode="x unified")
    theme.show(fig, height=300)
with right:
    fig = go.Figure(image_trace(img))
    fig.update_layout(height=300, margin=dict(l=8, r=8, t=30, b=8),
                      title=f"What the CNN sees: {H_IMG} x {W_IMG} pixels",
                      xaxis=dict(visible=False, constrain="domain"),
                      yaxis=dict(visible=False, autorange="reversed", scaleanchor="x"))
    theme.show(fig)
label = "UP" if move > 0 else "DOWN"
theme.callout(f"Label for this image: <strong>{label}</strong> ({move:+.1%} over the next {h} "
              f"business day{'s' if h > 1 else ''}). The network never sees a number: prices are min-max scaled "
              "inside each window, so it can only learn from the <em>shape</em> of the path.")

# --- 2. Examples -------------------------------------------------------------------------------
st.subheader("Can you tell which charts went up?")
rng = np.random.default_rng(7)
train_idx = [i for i in range(window, len(price) - h) if price.index[i] <= tr_end]
ups, downs = [], []
for i in rng.permutation(train_idx):
    r = price.iloc[i + h] / price.iloc[i] - 1
    (ups if r > 0 else downs).append((i, r))
    if len(ups) >= 4 and len(downs) >= 4:
        break
reveal = st.toggle("Reveal the labels", value=False)
ex = ups[:4] + downs[:4]
order = rng.permutation(len(ex))
fig = make_subplots(rows=2, cols=4, horizontal_spacing=0.02, vertical_spacing=0.1,
                    subplot_titles=[(f"{'UP' if ex[k][1] > 0 else 'DOWN'} {ex[k][1]:+.1%}" if reveal else f"Chart {n + 1}")
                                    for n, k in enumerate(order)])
for n, k in enumerate(order):
    i = ex[k][0]
    fig.add_trace(image_trace(L.chart_image(price.iloc[i - window + 1: i + 1].values, H_IMG, W_IMG)),
                  row=n // 4 + 1, col=n % 4 + 1)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, autorange="reversed")
for a in fig.layout.annotations:
    good = "UP" in a.text
    a.update(font=dict(size=12, color=(theme.AQUA if good else theme.ORANGE) if reveal else theme.INK_2))
fig.update_layout(height=330, margin=dict(l=8, r=8, t=30, b=8))
theme.show(fig)
st.caption(f"Eight {window}-day windows from the training period. Without the labels, up and down charts look alike, "
           "which is exactly why this is hard.")

# --- 3. The network and the paper --------------------------------------------------------------
a, b = st.columns([1, 1.2], gap="large")
with a:
    st.subheader("The CNN")
    nodes = [f'x [label="{H_IMG} x {W_IMG} chart image (1 channel)"]']
    for n, layer in enumerate(IC["layers"]):
        nodes.append(f'l{n} [label="{layer}", fillcolor="#FDF1D6"]')
    nodes.append('y [label="P(up), P(down)", fillcolor="' + theme.CRUDE_SOFT + '"]')
    chain = " -> ".join(["x"] + [f"l{n}" for n in range(len(IC["layers"]))] + ["y"])
    st.graphviz_chart(f"""
    digraph {{
      rankdir=TB; nodesep=0.25; ranksep=0.18;
      node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{theme.RULE}", fontname="Helvetica",
            fontsize=11, fontcolor="{theme.INK}", margin="0.18,0.06"];
      edge [color="{theme.MUTED}", arrowsize=0.6];
      {"; ".join(nodes)};
      {chain};
    }}""", width="stretch")
    st.caption(f"Loss: {IC['loss']}. Same train / validation / test dates as every other model.")
with b:
    st.subheader("Why images? Ren et al. (2024)")
    st.write("Ren, Jiang, Ji and Zhai turn daily WTI crude oil futures prices into images and train a CNN to "
             "predict the price trend. In their tests the image-based CNN was more accurate than the benchmark "
             "forecasting methods and the most stable approach in robustness checks; they argue images carry "
             "nonlinear information that matters most when prices swing sharply.")
    st.caption("Journal of Forecasting, 43(7), 2809-2821. doi:10.1002/for.3149")

# --- 4. Results --------------------------------------------------------------------------------
st.subheader("Does the image CNN call the direction?")
cat = L.catalog()
img_cfgs = sorted({(r.model, r.features) for r in cat.itertuples() if r.model in ("imgcnn", "cnn")})
if not img_cfgs:
    theme.pending("No image-CNN forecasts yet. Save the test forecasts as "
                  "`results/predictions/imgcnn_price_only_h<h>_s<seed>.csv` with the usual columns "
                  "(origin_date, target_date, base_price, true_price, pred_price) and, optionally, `prob_up`. "
                  "For a pure up/down classifier, set pred_price = base_price x (1 + 0.001) for up and "
                  "x (1 - 0.001) for down.",
                  "python <image_cnn_script>.py   # writes results/predictions/imgcnn_*.csv")
else:
    rows = []
    comp = L.comparison()
    for h_ in L.horizons():
        for m, f in img_cfgs:
            p = L.forecasts(m, f, h_)
            if p is None:
                continue
            step = p.iloc[::max(1, h_)]
            hit = np.sign(step.pred_price - step.base_price) == np.sign(step.true_price - step.base_price)
            k, n = int(hit.sum()), len(hit)
            rnn = comp[(comp.horizon == h_) & comp.model.isin(["lstm", "gru"])]
            best = rnn.loc[rnn.DirAcc.idxmax()] if not rnn.empty and rnn.DirAcc.notna().any() else None
            rows.append({"Horizon": h_, "Model": L.config_label(m, f), "Direction accuracy": k / n,
                         "p-value vs. coin flip": stats.binomtest(k, n, 0.5, alternative="greater").pvalue,
                         "Best LSTM/GRU": None if best is None else best.DirAcc,
                         "Best LSTM/GRU model": "" if best is None else best.config, "Test days (non-overlapping)": n})
    res = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"{r.Horizon} d" for r in res.itertuples()], y=res["Direction accuracy"],
                         name="Image CNN", marker_color=theme.YELLOW))
    if res["Best LSTM/GRU"].notna().any():
        fig.add_trace(go.Bar(x=[f"{r.Horizon} d" for r in res.itertuples()], y=res["Best LSTM/GRU"],
                             name="Best LSTM/GRU", marker_color=theme.BLUE))
    fig.add_hline(y=0.5, line=dict(color=theme.MUTED, dash="dot"), annotation_text="coin flip",
                  annotation_position="top left")
    fig.update_layout(yaxis=dict(title="Share of correct directions", range=[0.35, 0.7]), barmode="group")
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        theme.show(fig, height=300)
    with c2:
        st.dataframe(res, hide_index=True, width="stretch", column_config={
            "Direction accuracy": st.column_config.NumberColumn(format="%.3f"),
            "p-value vs. coin flip": st.column_config.NumberColumn(format="%.3f"),
            "Best LSTM/GRU": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Scored on every h-th test day so the targets do not overlap; the p-value is a one-sided binomial "
               "test against 50%.")

theme.footer(KEY)
