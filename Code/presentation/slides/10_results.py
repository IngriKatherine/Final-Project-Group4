"""Main result: every configuration against the baselines, plus Diebold-Mariano tests."""
import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import loaders as L
import theme

KEY = "10_results"
theme.header(KEY)

cfg = L.config()
e = cfg["experiments"]
comp = L.comparison()
if comp.empty:
    theme.pending("No forecasts found in results/predictions.", "python 03_baselines.py\npython 06_run_experiments.py")
    theme.footer(KEY)
    st.stop()

hs = sorted(comp.horizon.unique())
expected = set(itertools.product(e["models"], e["feature_sets"], cfg["framing"]["horizons"]))
have = {(r.model, r.features, r.horizon) for r in comp.itertuples() if r.model not in L.BASELINES}
missing = sorted(expected - have)

# --- Best model per horizon ------------------------------------------------------------------
best = comp.loc[comp.groupby("horizon")["RMSE"].idxmin()]
theme.figures([(b.config, f"lowest test RMSE at {b.horizon} day{'s' if b.horizon > 1 else ''} "
                          f"(${b.RMSE:.2f})") for b in best.itertuples()])

if missing:
    by = {}
    for m, f, h in missing:
        by.setdefault(L.config_label(m, f), []).append(h)
    st.info("Not run yet: " + "; ".join(f"{k} (h = {', '.join(map(str, v))})" for k, v in by.items())
            + ". This slide fills in automatically once they exist.", icon=":material/pending:")
    with st.popover("How to run them"):
        st.code("python 06_run_experiments.py      # full grid, resumable\n"
                "python 08_make_figures_tables.py  # report tables and figures", language="bash")

t_chart, t_table, t_dm = st.tabs(["Compare", "Full table", "Is it significant?"])

with t_chart:
    c1, c2 = st.columns([2, 1])
    metric = c1.segmented_control("Metric", ["Skill", "RMSE", "MAE", "MAPE", "DirAcc"], default="Skill",
                                  required=True, format_func=lambda x: {"Skill": "Skill vs. random walk",
                                                                        "DirAcc": "Direction"}.get(x, x))
    show_base = c2.toggle("Show baselines as bars", value=True)
    st.caption(L.METRIC_HELP[metric])

    fig = make_subplots(rows=1, cols=len(hs), subplot_titles=[f"{h} day" + ("s" if h > 1 else "") + " ahead" for h in hs],
                        shared_yaxes=metric in ("Skill", "DirAcc"), horizontal_spacing=0.06)
    seen = set()
    for j, h in enumerate(hs, 1):
        d = comp[comp.horizon == h]
        if not show_base:
            d = d[~d.model.isin(L.BASELINES)]
        d = d.dropna(subset=[metric])
        for r in d.itertuples():
            err = None
            if metric == "RMSE" and pd.notna(getattr(r, "RMSE_seed_std", np.nan)):
                err = dict(type="data", array=[r.RMSE_seed_std], color=theme.MUTED, thickness=1, width=4)
            name = r.config
            fig.add_trace(go.Bar(
                x=[name], y=[getattr(r, metric)], name=name, legendgroup=name, showlegend=name not in seen,
                marker=dict(color=theme.config_color(r.model, r.features),
                            pattern=dict(shape="/" if r.model == "gru" else "", fillmode="overlay",
                                         fgcolor="rgba(255,255,255,0.75)", size=7, solidity=0.35)),
                error_y=err, hovertemplate=f"{name}<br>{metric}: %{{y:.3f}}<extra>{h}-day</extra>"), row=1, col=j)
            seen.add(name)
        if metric == "Skill":
            fig.add_hline(y=0, line=dict(color=theme.INK, width=1), row=1, col=j)
        if metric in ("RMSE", "MAE", "MAPE"):
            rw = comp[(comp.horizon == h) & (comp.model == "rw")]
            if not rw.empty:
                fig.add_hline(y=float(rw[metric].iloc[0]), line=dict(color=theme.MUTED, width=1, dash="dot"),
                              annotation_text="random walk", annotation_position="top left",
                              annotation_font=dict(size=10, color=theme.INK_2), row=1, col=j)
        if metric == "DirAcc":
            fig.add_hline(y=0.5, line=dict(color=theme.MUTED, width=1, dash="dot"), annotation_text="coin flip",
                          annotation_position="top left", annotation_font=dict(size=10, color=theme.INK_2), row=1, col=j)
    fig.update_xaxes(showticklabels=False, showgrid=False)
    ytitle = {"Skill": "Skill vs. random walk (%)", "RMSE": "Test RMSE (USD/bbl)", "MAE": "Test MAE (USD/bbl)",
              "MAPE": "Test MAPE (%)", "DirAcc": "Share of correct directions"}[metric]
    fig.update_yaxes(title_text=ytitle, row=1, col=1)
    fig.update_layout(bargap=0.25, legend=dict(y=1.14))
    for a in fig.layout.annotations[:len(hs)]:
        a.update(font=dict(size=13, color=theme.INK), x=a.x, xanchor="center")
    theme.show(fig, height=440)
    st.caption("Colour is the input set (blue: price only, orange: price + market drivers, purple: selected "
               "drivers); hatching marks GRU, solid LSTM; yellow is the image CNN; grey bars are the statistical "
               "models. Forecasts are averaged over seeds; whiskers on RMSE show the spread of single-seed runs.")

    rnn = comp[~comp.model.isin(L.BASELINES)]
    if not rnn.empty:
        bestrnn = rnn.loc[rnn.groupby("horizon")["Skill"].idxmax()]
        words = []
        for b in bestrnn.itertuples():
            verb = "beats" if b.Skill > 0 else "trails"
            words.append(f"at {b.horizon} day{'s' if b.horizon > 1 else ''} the best model ({b.config}) {verb} "
                         f"the random walk by {abs(b.Skill):.1f}%")
        theme.callout("So far, " + "; ".join(words) + ".")

with t_table:
    show = comp.copy()
    show["RMSE (seeds)"] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else "" for m, s in
                            zip(show.get("RMSE_seed_mean", np.nan), show.get("RMSE_seed_std", np.nan))]
    show = show[["horizon", "config", "n_seeds", "RMSE", "RMSE (seeds)", "MAE", "MAPE", "DirAcc", "Skill"]]
    st.dataframe(
        show, hide_index=True, width="stretch", height=min(40 + 35 * len(show), 640),
        column_config={
            "horizon": "h", "config": "Model / input set", "n_seeds": "Seeds",
            "RMSE": st.column_config.NumberColumn("RMSE", help="Of the seed-averaged forecast", format="%.3f"),
            "RMSE (seeds)": st.column_config.TextColumn(help="Mean ± std of single-seed runs"),
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "MAPE": st.column_config.NumberColumn("MAPE %", format="%.2f"),
            "DirAcc": st.column_config.NumberColumn("Direction", format="%.3f",
                                                    help="Not defined for the random walk (it never predicts a move)"),
            "Skill": st.column_config.ProgressColumn("Skill vs. RW %", format="%.1f",
                                                     min_value=float(min(-20, comp.Skill.min())),
                                                     max_value=float(max(20, comp.Skill.max()))),
        })
    st.download_button("Download as CSV", show.to_csv(index=False), "final_comparison_app.csv", "text/csv",
                       icon=":material/download:")

with t_dm:
    dm = L.dm_tests()
    if dm.empty:
        st.info("No model forecasts to test yet.")
    else:
        alpha = st.select_slider("Significance level", [0.01, 0.05, 0.10], value=0.05)
        st.write("Diebold-Mariano test on squared errors, with the Harvey-Leybourne-Newbold small-sample "
                 "correction and h-1 Newey-West lags (oilfc/metrics.py). A negative statistic means the first "
                 "model is more accurate.")

        def reading(r):
            if r.p_value >= alpha:
                return "no significant difference"
            return f"{r.model} more accurate" if r.DM < 0 else f"{r.vs} more accurate"

        dm = dm.assign(Result=dm.apply(reading, axis=1))
        for kind in ("vs. baseline", "between input sets"):
            d = dm[dm.comparison == kind]
            st.markdown(f"**{'Against the statistical models' if kind == 'vs. baseline' else 'Extra inputs vs. price only, same network'}**")
            if d.empty:
                st.caption("Needs at least two input sets for the same network and horizon.")
                continue

            def color(row):
                c = "" if row.p_value >= alpha else ("background-color:#DDF2E6" if row.DM < 0 else "background-color:#FBE3E1")
                return [c if col == "Result" else "" for col in row.index]

            st.dataframe(d.drop(columns="comparison").style.apply(color, axis=1).format(
                {"DM": "{:+.2f}", "p_value": "{:.4f}"}), hide_index=True, width="stretch",
                column_config={"horizon": "h", "model": "Model", "vs": "Against", "p_value": "p-value", "n": "Days"})
        st.caption("Green: the model in the first column is significantly more accurate. Red: significantly less accurate.")

theme.footer(KEY)
