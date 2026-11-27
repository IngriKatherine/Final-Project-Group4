"""Training setup and learning curves from every saved run."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import loaders as L
import theme

KEY = "08_training"
theme.header(KEY)

cfg = L.config()
t = cfg["training"]
runs = L.main_runs()

if not runs.empty:
    theme.figures([
        (f"{len(runs)}", "training runs so far"),
        (f"{runs.epochs.median():.0f}", "median epochs before stopping"),
        (f"{runs.best_epoch.median():.0f}", "median epoch of the kept weights"),
        (f"{runs.train_seconds.median():.0f} s", "median training time per run"),
    ])

left, right = st.columns([1, 2.1], gap="large")
with left:
    st.subheader("Setup")
    rows = [
        ("Loss", f"{t['loss'].upper()} on the standardised log return"),
        ("Optimiser", f"Adam, learning rate {t['learning_rate']:g}, weight decay {t['weight_decay']:g}"),
        ("Schedule", f"halve the LR when validation loss stalls for {max(1, t['patience'] // 3)} epochs"),
        ("Batches", f"{t['batch_size']} windows, shuffled within the training set"),
        ("Stopping", f"patience {t['patience']} epochs, keep the best validation weights"),
        ("Clipping", f"gradient norm {t['grad_clip']:g}"),
        ("Dropout", f"{cfg['model']['dropout']:g} between layers and before the head"),
        ("Seeds", ", ".join(map(str, cfg["experiments"]["seeds"])) + " per configuration"),
        ("Hardware", "AWS GPU instance (g5.xlarge, A10G)"),
    ]
    st.html('<dl class="spec">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows) + "</dl>")
with right:
    if runs.empty:
        theme.pending("No training runs found in results/tables/runs.", "python 06_run_experiments.py")
    else:
        st.subheader("Learning curves")
        c1, c2 = st.columns([1.3, 1])
        h = c1.segmented_control("Horizon", sorted(runs.horizon.unique()), default=int(runs.horizon.min()),
                                 format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""), required=True)
        sub = runs[runs.horizon == h]
        cfgs = sorted({(r.model, r.features) for r in sub.itertuples()})
        pick = c2.selectbox("Configuration", cfgs, format_func=lambda p: L.config_label(*p))
        sel = sub[(sub.model == pick[0]) & (sub.features == pick[1])].sort_values("seed")
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Training loss", "Validation loss"],
                            horizontal_spacing=0.1)
        shades = [theme.BLUE, theme.ORANGE, theme.AQUA, theme.YELLOW]
        for k, r in enumerate(sel.itertuples()):
            hist = L.history(r.run)
            ep = np.arange(1, len(hist["train_loss"]) + 1)
            color = shades[k % len(shades)]
            fig.add_trace(go.Scatter(x=ep, y=hist["train_loss"], name=f"seed {r.seed}", mode="lines",
                                     line=dict(color=color, width=2), legendgroup=str(r.seed)), row=1, col=1)
            fig.add_trace(go.Scatter(x=ep, y=hist["val_loss"], name=f"seed {r.seed}", mode="lines", showlegend=False,
                                     line=dict(color=color, width=2), legendgroup=str(r.seed)), row=1, col=2)
            b = int(np.argmin(hist["val_loss"]))
            fig.add_trace(go.Scatter(x=[b + 1], y=[hist["val_loss"][b]], mode="markers", showlegend=False,
                                     marker=dict(size=10, color=color, line=dict(color="#fff", width=2)),
                                     hovertemplate=f"seed {r.seed}: best epoch %{{x}}<extra></extra>"), row=1, col=2)
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="MSE, standardised target", row=1, col=1)
        fig.update_layout(hovermode="x unified", legend=dict(y=1.16), margin=dict(t=70))
        for a in fig.layout.annotations:
            a.update(font=dict(size=13, color=theme.INK))
        theme.show(fig, height=380)
        st.caption("Each panel has its own scale. Validation loss sits far above training loss because the "
                   "2019 to 2021 period contains the April 2020 returns, which are huge in standardised units. "
                   f"Dots mark the epoch whose weights were kept; training stops {t['patience']} epochs later.")

if not runs.empty:
    st.subheader("Every run at a glance")
    show = runs.assign(config=[L.config_label(m, f) for m, f in zip(runs.model, runs.features)])
    show = show[["config", "horizon", "seed", "n_features", "n_params", "best_epoch", "epochs", "train_seconds",
                 "val_RMSE", "test_RMSE"]].sort_values(["horizon", "config", "seed"])
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "config": "Configuration", "horizon": "h", "seed": "Seed", "n_features": "Inputs",
        "n_params": st.column_config.NumberColumn("Parameters", format="%d"),
        "best_epoch": "Best epoch", "epochs": "Epochs run",
        "train_seconds": st.column_config.NumberColumn("Train time (s)", format="%.0f"),
        "val_RMSE": st.column_config.NumberColumn("Val. RMSE", format="%.3f"),
        "test_RMSE": st.column_config.NumberColumn("Test RMSE", format="%.3f"),
    })

theme.footer(KEY)
