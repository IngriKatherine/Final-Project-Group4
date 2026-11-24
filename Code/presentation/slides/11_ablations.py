"""One-factor-at-a-time ablations from 07_ablations.py (read from the run summaries)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "11_ablations"
theme.header(KEY)

cfg = L.config()
grid = cfg.get("ablations", {})
abl = L.ablation_runs()
defaults = {"hidden_size": cfg["model"]["hidden_size"], "num_layers": cfg["model"]["num_layers"],
            "dropout": cfg["model"]["dropout"], "lookback": cfg["framing"]["lookback"]}
names = {"hidden_size": "Hidden units", "num_layers": "Layers", "lookback": "Lookback window (days)",
         "dropout": "Dropout"}

if abl.empty:
    st.write("Starting from the default network, each row changes one setting and keeps everything else fixed. "
             "Every cell is trained with each model and seed.")
    plan = pd.DataFrame([{"Setting": names.get(p, p), "Values tried": ", ".join(map(str, v)),
                          "Default": defaults.get(p)} for p, v in grid.items()])
    st.dataframe(plan, hide_index=True, width="stretch")
    n = sum(len(v) for v in grid.values()) * len(cfg["experiments"]["models"]) * len(cfg["experiments"]["seeds"])
    theme.pending(f"No ablation runs yet ({n} runs planned).",
                  "python 07_ablations.py --features shap_selected --horizon 5")
    theme.footer(KEY)
    st.stop()

feats = sorted(abl.features.unique())
hs = sorted(abl.horizon.unique())
c1, c2, c3 = st.columns([1.3, 1, 1])
param = c1.segmented_control("Setting", [p for p in grid if p in set(abl.param)], format_func=lambda p: names.get(p, p),
                             default=[p for p in grid if p in set(abl.param)][0], required=True)
fs = c2.selectbox("Input set", feats, format_func=lambda f: L.FEATURE_SET_LABELS.get(f, f))
h = c3.selectbox("Horizon", hs, format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""))

d = abl[(abl.param == param) & (abl.features == fs) & (abl.horizon == h)]
agg = d.groupby(["model", "value"]).agg(val=("val_RMSE", "mean"), val_sd=("val_RMSE", "std"),
                                        test=("test_RMSE", "mean"), params=("n_params", "first"),
                                        n=("seed", "count")).reset_index()

left, right = st.columns([1.6, 1], gap="large")
with left:
    fig = go.Figure()
    for model, g in agg.groupby("model"):
        color = theme.BLUE if model == "lstm" else theme.ORANGE
        fig.add_trace(go.Scatter(x=g.value, y=g.val, name=f"{model.upper()}, validation", mode="lines+markers",
                                 line=dict(color=color, width=2.2), marker=dict(size=9),
                                 error_y=dict(type="data", array=g.val_sd.fillna(0), color=color, thickness=1, width=4),
                                 customdata=np.stack([g.params, g.n], axis=1),
                                 hovertemplate="%{x}: %{y:.3f} (%{customdata[0]:,} params, %{customdata[1]} seeds)<extra></extra>"))
        fig.add_trace(go.Scatter(x=g.value, y=g.test, name=f"{model.upper()}, test", mode="lines+markers",
                                 line=dict(color=color, width=1.2, dash="dot"), marker=dict(size=6, symbol="circle-open"),
                                 hovertemplate="%{x}: test %{y:.3f}<extra></extra>"))
    rw = L.comparison()
    rw = rw[(rw.model == "rw") & (rw.horizon == h)]
    if not rw.empty:
        fig.add_hline(y=float(rw.RMSE.iloc[0]), line=dict(color=theme.MUTED, width=1, dash="dot"),
                      annotation_text="random walk, test", annotation_position="top left",
                      annotation_font=dict(size=10, color=theme.INK_2))
    if param in defaults:
        fig.add_vline(x=defaults[param], line=dict(color=theme.MUTED, width=1),
                      annotation_text="default", annotation_position="top right",
                      annotation_font=dict(size=10, color=theme.INK_2))
    fig.update_layout(xaxis=dict(title=names.get(param, param), tickvals=sorted(agg.value.unique())),
                      yaxis_title="RMSE (USD/bbl)", hovermode="closest")
    theme.show(fig, height=400)
with right:
    best = agg.loc[agg.groupby("model")["val"].idxmin()]
    theme.figures([(f"{b.value:g}", f"best {names.get(param, param).lower()} for {b.model.upper()} "
                                    f"(validation RMSE {b.val:.3f})") for b in best.itertuples()])
    st.dataframe(agg.assign(model=agg.model.str.upper()).rename(columns={"model": "Model", "value": names.get(param, param), "val": "Val. RMSE", "val_sd": "Std",
                                     "test": "Test RMSE", "params": "Parameters", "n": "Seeds"}).round(3),
                 hide_index=True, width="stretch")
st.caption("Solid lines and the best-setting figures use validation error only, which is how a setting would be "
           "chosen. Dotted test lines are for information and were not used to choose anything.")

png = L.figure_path(f"ablation_{param}.png")
if png:
    with st.expander("Figure from 07_ablations.py"):
        st.image(str(png), width="stretch")

theme.footer(KEY)
