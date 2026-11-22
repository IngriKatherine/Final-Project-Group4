"""SHAP: the method in one panel, then what drives the forecasts at each horizon."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import kendalltau

import loaders as L
import theme

KEY = "09_shap"
theme.header(KEY)

cfg = L.config()
s = cfg["shap"]
hs = L.shap_horizons()
target = cfg["data"]["target"]

with st.expander("SHAP in one slide", expanded=False, icon=":material/school:"):
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.write("A Shapley value splits one prediction fairly among the inputs: each input gets its average "
                 "marginal contribution over every order in which inputs could be added.")
        st.latex(r"\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}"
                 r"\left[f(S \cup \{j\}) - f(S)\right]")
        st.latex(r"f(x) = \mathbb{E}[f(X)] + \sum_j \phi_j \quad \text{(additivity)}")
    with b:
        st.markdown(f"""
- **Explainer:** `shap.GradientExplainer` (expected gradients) on the trained all-features LSTM.
- **Background:** {s['background_size']} training windows. **Explained:** {s['explain_size']} validation windows.
  The test period is never touched.
- **Aggregation:** mean |SHAP| over windows and over the {cfg['framing']['lookback']} lags gives one score per
  input, averaged over {len(cfg['experiments']['seeds'])} seeds.
- **Selection:** WTI's own history is always kept, plus the top {s['top_k']} other drivers.
""")

if not hs:
    theme.pending("No SHAP results yet.", "python 05_shap_analysis.py")
    theme.footer(KEY)
    st.stop()

h = st.segmented_control("Forecast horizon", hs, default=hs[0], required=True,
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""))
imp = L.shap_importance(h)
sel = L.shap_selection(h).get("selected", [])

t_rank, t_lag, t_hor, t_stab, t_png = st.tabs(
    ["Ranking", "Which days in the window", "Across horizons", "Agreement between seeds", "Report figures"])

with t_rank:
    c1, c2 = st.columns([1.6, 1], gap="large")
    with c1:
        d = imp.sort_values("mean_abs_shap")
        colors = [theme.INK if c == target else theme.AQUA if c in sel else "#C9D1D6" for c in d.index]
        fig = go.Figure(go.Bar(
            x=d.mean_abs_shap, y=[L.label(c) for c in d.index], orientation="h", marker_color=colors,
            error_x=dict(type="data", array=d.seed_std, color=theme.MUTED, thickness=1, width=3),
            customdata=np.stack([d.share * 100, d["rank"]], axis=1),
            hovertemplate="%{y}<br>mean |SHAP| %{x:.4f}<br>%{customdata[0]:.1f}% of total, rank %{customdata[1]}"
                          "<extra></extra>"))
        fig.update_layout(xaxis_title="mean |SHAP| (standardised target units)", showlegend=False,
                          title=f"Feature importance, {h}-day horizon")
        theme.show(fig, height=460)
        st.caption("Dark: WTI itself (always kept). Green: the drivers selected for the SHAP input set. "
                   "Whiskers: spread across seeds.")
    with c2:
        share_sel = imp.loc[sel, "share"].sum() * 100 if sel else 0
        theme.figures([(f"{imp.loc[target, 'share'] * 100:.0f}%", "of importance on WTI's own history"),
                       (f"{share_sel:.0f}%", f"on the {len(sel)} selected drivers")])
        st.markdown("**Selected for this horizon**")
        for i, c in enumerate(sel, 1):
            st.markdown(f"{i}. {L.label(c)}  \n<span style='color:{theme.INK_2};font-size:.9rem'>"
                        f"{imp.loc[c, 'share'] * 100:.1f}% of total importance</span>", unsafe_allow_html=True)
        least = imp.sort_values("mean_abs_shap").index[0]
        st.caption(f"Least used: {L.label(least)}. A rate that changes a handful of times a year "
                   "carries almost no daily signal." if least == "FEDFUNDS" else f"Least used: {L.label(least)}.")

with t_lag:
    lag = L.shap_lags(h)
    order = imp.index.tolist()
    z = lag[order].T.values
    fig = go.Figure(go.Heatmap(z=z, x=list(lag.index), y=[L.label(c) for c in order], colorscale=theme.SEQ_BLUE,
                               xgap=1, ygap=1, colorbar=dict(title="mean |SHAP|", thickness=10),
                               hovertemplate="%{y}, %{x}: %{z:.5f}<extra></extra>"))
    fig.update_layout(xaxis=dict(title="Day in the input window (t-0 is the forecast origin)", showgrid=False),
                      yaxis=dict(autorange="reversed", showgrid=False))
    theme.show(fig, height=470)
    tot = lag.sum(axis=1)
    last5 = tot.iloc[-5:].sum() / tot.sum() * 100
    st.caption(f"The last 5 of {len(lag)} days carry {last5:.0f}% of all attribution at this horizon. "
               "The network mostly reacts to the latest moves; older days matter little.")

with t_hor:
    ranks = pd.DataFrame({hh: L.shap_importance(hh)["rank"] for hh in hs})
    default = [c for c in L.shap_selection(hs[0]).get("selected", [])[:3] if c in ranks.index]
    pick = st.multiselect("Highlight", ranks.index.tolist(), default=default, format_func=L.label)
    fig = go.Figure()
    palette = [theme.BLUE, theme.ORANGE, theme.AQUA, theme.YELLOW, "#e87ba4", "#008300"]
    k = 0
    for c in ranks.index:
        on = c in pick
        color = palette[k % len(palette)] if on else "#D3D9DD"
        k += int(on)
        fig.add_trace(go.Scatter(x=[f"{x} d" for x in ranks.columns], y=ranks.loc[c], mode="lines+markers+text",
                                 name=L.label(c), line=dict(color=color, width=2.6 if on else 1.2),
                                 marker=dict(size=9 if on else 6, color=color),
                                 text=[""] * (len(hs) - 1) + [L.label(c) if on else ""], textposition="middle right",
                                 textfont=dict(size=11, color=theme.INK if on else theme.MUTED),
                                 showlegend=False, hovertemplate=L.label(c) + ": rank %{y}<extra></extra>"))
    n = len(ranks)
    fig.update_layout(yaxis=dict(autorange="reversed", title="Rank (1 = most important)", dtick=1),
                      xaxis=dict(title="Horizon", range=[-0.2, len(hs) - 0.2 + 0.9]))
    theme.show(fig, height=480)
    st.caption("Rank of every input by horizon. Selected sets per horizon: " + "; ".join(
        f"{hh} d: " + ", ".join(L.label(c) for c in L.shap_selection(hh).get("selected", [])) for hh in hs) + ".")

with t_stab:
    seed_cols = [c for c in imp.columns if c.startswith("seed")]
    if len(seed_cols) < 2:
        st.info("Per-seed importances need at least two seeds.")
    else:
        rows = []
        for hh in hs:
            im = L.shap_importance(hh)
            for i, a_ in enumerate(seed_cols):
                for b_ in seed_cols[i + 1:]:
                    tau, p = kendalltau(im[a_], im[b_])
                    top_a = set(im[a_].drop(target).nlargest(s["top_k"]).index)
                    top_b = set(im[b_].drop(target).nlargest(s["top_k"]).index)
                    rows.append({"Horizon": f"{hh} d", "Seeds": f"{a_} vs {b_}", "Kendall tau": tau, "p": p,
                                 f"Shared top {s['top_k']}": len(top_a & top_b)})
        stab = pd.DataFrame(rows)
        c1, c2 = st.columns([1.2, 1], gap="large")
        with c1:
            fig = go.Figure()
            for i, hh in enumerate(hs):
                d = stab[stab.Horizon == f"{hh} d"]
                fig.add_trace(go.Bar(x=d.Seeds, y=d["Kendall tau"], name=f"{hh} d",
                                     marker_color=[theme.BLUE, theme.ORANGE, theme.AQUA][i % 3],
                                     hovertemplate="%{x}: tau %{y:.2f}<extra>" + f"{hh} d" + "</extra>"))
            fig.update_layout(barmode="group", yaxis=dict(title="Kendall rank correlation", range=[0, 1]),
                              bargap=0.3, bargroupgap=0.08)
            theme.show(fig, height=340)
        with c2:
            st.dataframe(stab.round(3), hide_index=True, width="stretch")
        st.caption("Each seed is a separately trained network. High tau means the seeds rank the inputs alike, "
                   "so the selection is not an accident of one initialisation. Averaging over seeds makes it steadier still.")

with t_png:
    figs = [f for f in (f"shap_beeswarm_h{h}.png", f"shap_bar_h{h}.png", f"shap_lag_heatmap_h{h}.png")
            if L.figure_path(f)]
    if not figs:
        st.info("No figures in results/figures yet.")
    for f in figs:
        st.image(str(L.figure_path(f)), caption=f"results/figures/{f}", width="stretch")

theme.footer(KEY)
