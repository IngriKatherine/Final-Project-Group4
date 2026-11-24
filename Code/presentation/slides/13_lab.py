"""Model lab: live inference with a saved checkpoint, the input window, and what-if shocks."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kendalltau

import loaders as L
import theme

KEY = "13_lab"
theme.header(KEY)

if not L.torch_available():
    st.warning("This slide runs the trained network live and needs PyTorch.")
    st.code("pip install torch", language="bash")
    theme.footer(KEY)
    st.stop()

ck_df = L.checkpoints()
if ck_df.empty:
    theme.pending("No checkpoints in results/models (they are git-ignored, so copy them from the AWS machine "
                  "or retrain).", "python 06_run_experiments.py")
    theme.footer(KEY)
    st.stop()

lv = L.levels()
ret = L.returns()
_, va_end = L.split_dates()

# --- Choose a network and a day ----------------------------------------------------------------
c1, c2, c3 = st.columns([1, 1.2, 1])
h = c1.segmented_control("Horizon", sorted(ck_df.horizon.unique()), default=int(ck_df.horizon.min()), required=True,
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""))
sub = ck_df[ck_df.horizon == h]
pairs = sorted({(r.model, r.features) for r in sub.itertuples()})
pair = c2.selectbox("Network", pairs, format_func=lambda p: L.config_label(*p))
seeds = sorted(sub[(sub.model == pair[0]) & (sub.features == pair[1])].seed)
seed_pick = c3.segmented_control("Seeds", ["Average"] + [f"seed {s}" for s in seeds], default="Average", required=True)
stems = [f"{pair[0]}_{pair[1]}_h{h}_s{s}" for s in
         (seeds if seed_pick == "Average" else [int(seed_pick.split()[1])])]

net, ck = L.load_checkpoint(stems[0])
names = ck["feature_names"]
lookback = ck["lookback"]
valid_dates = ret.index[lookback - 1:]
default_day = valid_dates[valid_dates.get_indexer([pd.Timestamp("2022-02-23")], method="nearest")[0]]
origin = st.select_slider("Forecast origin (the network sees data up to and including this day)",
                          options=list(valid_dates), value=default_day, format_func=lambda d: f"{d:%a %d %b %Y}")

X, win_dates = L.scaled_window(ck, origin)
base_ret = float(L.predict_return(stems, X[None])[0])
p0 = float(lv.loc[origin, "WTI"])
pos = lv.index.get_loc(origin)
target_day = lv.index[pos + h] if pos + h < len(lv) else None
actual = float(lv["WTI"].iloc[pos + h]) if target_day is not None else None
forecast = p0 * np.exp(base_ret)

items = [(f"${p0:.2f}", f"WTI on {origin:%d %b %Y}"),
         (f"${forecast:.2f}", f"network forecast for {h} day{'s' if h > 1 else ''} later ({base_ret:+.2%})")]
if actual is not None:
    items += [(f"${actual:.2f}", f"actual on {target_day:%d %b %Y}"),
              (f"{abs(forecast - actual) - abs(p0 - actual):+.2f}",
               "network error minus random-walk error (negative is better)")]
theme.figures(items)
if target_day is not None and target_day <= va_end:
    st.caption("This target lies in the training or validation period, so the network has seen it during fitting "
               "or model selection. Pick a day after the validation period for an honest out-of-sample check.")

# --- What the network sees ---------------------------------------------------------------------
left, right = st.columns([1.45, 1], gap="large")
with left:
    st.subheader("What the network sees")
    z = np.clip(X.T, -4, 4)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{d:%d %b}" for d in win_dates], y=[L.label(c) for c in names], zmid=0, zmin=-4, zmax=4,
        colorscale=theme.DIVERGING, xgap=1, ygap=1, colorbar=dict(title="std.", thickness=10),
        customdata=ret.loc[win_dates, names].T.values * 100,
        hovertemplate="%{y}, %{x}<br>%{z:.2f} std. (raw change %{customdata:.2f})<extra></extra>"))
    fig.update_layout(xaxis=dict(showgrid=False, tickangle=-45, nticks=12), yaxis=dict(autorange="reversed", showgrid=False))
    theme.show(fig, height=430)
    st.caption(f"The {lookback} x {len(names)} input: each input's daily change, standardised with training-period "
               "statistics. Red is an unusually large rise, blue an unusually large fall.")

# --- What-if -----------------------------------------------------------------------------------
with right:
    st.subheader("What if one input had moved?")
    feat = st.selectbox("Input to shock", names, index=names.index("BRENT") if "BRENT" in names else 0,
                        format_func=L.label)
    size = st.slider("Shock per day (standard deviations)", -3.0, 3.0, 2.0, 0.5)
    days = st.slider("Applied to the last N days of the window", 1, lookback, 5)
    Xs = X.copy()
    Xs[-days:, names.index(feat)] += size
    shocked = float(L.predict_return(stems, Xs[None])[0])
    fig = go.Figure(go.Bar(x=["As observed", "With shock"], y=[p0 * np.exp(base_ret), p0 * np.exp(shocked)],
                           marker_color=["#C9D1D6", theme.CRUDE], text=[f"${p0 * np.exp(base_ret):.2f}",
                                                                         f"${p0 * np.exp(shocked):.2f}"],
                           textposition="outside", hovertemplate="%{x}: $%{y:.2f}<extra></extra>"))
    lo = min(p0 * np.exp(base_ret), p0 * np.exp(shocked))
    hi = max(p0 * np.exp(base_ret), p0 * np.exp(shocked))
    pad = max(0.5, (hi - lo) * 0.8)
    fig.update_layout(yaxis=dict(title="Forecast, USD per barrel", range=[lo - pad, hi + pad]), showlegend=False)
    theme.show(fig, height=280)
    st.write(f"A {size:+g} std. move in **{L.label(feat)}** on each of the last {days} days changes the "
             f"forecast by **{'+' if shocked >= base_ret else '-'}${abs(p0 * (np.exp(shocked) - np.exp(base_ret))):.2f}** per barrel.")

# --- Sensitivity of every input ----------------------------------------------------------------
st.subheader("Which inputs move this forecast most?")
k = st.slider("Shock size (std.) applied to the last 5 days, up and down", 0.5, 3.0, 2.0, 0.5, key="sens_k")
batch = []
for j in range(len(names)):
    for sgn in (1, -1):
        Xj = X.copy()
        Xj[-5:, j] += sgn * k
        batch.append(Xj)
out = L.predict_return(stems, np.stack(batch))
up, down = p0 * (np.exp(out[0::2]) - np.exp(base_ret)), p0 * (np.exp(out[1::2]) - np.exp(base_ret))
sens = pd.DataFrame({"up": up, "down": down, "range": np.abs(up - down)}, index=names).sort_values("range")

c1, c2 = st.columns([1.6, 1], gap="large")
with c1:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[L.label(c) for c in sens.index], x=sens.up, orientation="h", name=f"+{k:g} std.",
                         marker_color=theme.BLUE, hovertemplate="%{y}: %{x:+.2f} USD<extra>up</extra>"))
    fig.add_trace(go.Bar(y=[L.label(c) for c in sens.index], x=sens.down, orientation="h", name=f"-{k:g} std.",
                         marker_color=theme.ORANGE, hovertemplate="%{y}: %{x:+.2f} USD<extra>down</extra>"))
    fig.add_vline(x=0, line=dict(color=theme.INK, width=1))
    fig.update_layout(barmode="overlay", xaxis_title="Change in forecast (USD/bbl)", bargap=0.35)
    theme.show(fig, height=420)
with c2:
    hs_shap = L.shap_horizons()
    if h in hs_shap:
        imp = L.shap_importance(h)["mean_abs_shap"]
        common = [c for c in sens.index if c in imp.index]
        if len(common) > 2:
            tau, p = kendalltau(sens.loc[common, "range"], imp.loc[common])
            theme.figures([(f"{tau:.2f}", "rank agreement with SHAP (Kendall tau)")])
            st.write("SHAP averages over 500 validation windows; this chart perturbs one window. Agreement "
                     "between the two is a sanity check that SHAP reflects how the network behaves.")
    top = sens.index[-1]
    st.write(f"On this day the forecast reacts most to **{L.label(top)}**.")

theme.footer(KEY)
