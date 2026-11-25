"""Ships from space: the Strait of Hormuz, the satellite ship detector, and the oil-supply series
(all read from Code/satellite outputs; each part says what to run while it is missing)."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import loaders as L
import theme

KEY = "13_satellite"
theme.header(KEY)

CLOSURE = pd.Timestamp("2026-02-28")
pw = L.portwatch_hormuz()

# --- Headline numbers (from PortWatch when available) --------------------------------------------
if pw is not None and pw["date"].max() > CLOSURE:
    pre = pw[(pw.date >= "2025-01-01") & (pw.date < CLOSURE)]
    post = pw[pw.date >= CLOSURE + pd.Timedelta(days=3)]
    theme.figures([
        (f"{pre.n_tanker.mean():.0f}", "tankers a day through Hormuz in 2025 (AIS, PortWatch)"),
        (f"{post.n_tanker.mean():.1f}", "tankers a day since the 28 Feb 2026 closure"),
        (f"{1 - post.n_tanker.mean() / pre.n_tanker.mean():.0%}", "drop in AIS-visible tanker transits"),
    ])

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("From pixels to barrels")
    st.graphviz_chart(f"""
    digraph {{
      rankdir=TB; nodesep=0.25; ranksep=0.2;
      node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{theme.RULE}", fontname="Helvetica",
            fontsize=11, fontcolor="{theme.INK}", margin="0.18,0.07"];
      edge [color="{theme.MUTED}", arrowsize=0.6];
      a [label="Sentinel-2, 10 m, every 2-5 days: one cloud-masked image of the strait per date"];
      b [label="Faster R-CNN (ResNet-50 + FPN), pretrained on COCO", fillcolor="#E3F5EE"];
      c [label="Fine-tuned on 37k labelled ships (Ai2) + our hand-checked Hormuz chips", fillcolor="#E3F5EE"];
      d [label="Ships per zone; hull length from the near-infrared band"];
      e [label="Ships of 150 m or more -> tanker capacity (deadweight)"];
      f [label="IMF PortWatch tanker transits (AIS) = labels to calibrate capacity into barrels/day",
         fillcolor="{theme.CRUDE_SOFT}"];
      g [label="Oil-supply series: moving through the strait, and waiting at anchor"];
      a -> b -> c -> d -> e -> f -> g;
    }}""", width="stretch")
    st.caption("Code/satellite, scripts s01-s07. The calibration uses 2023-2025 only, so 2026 stays unseen.")
with right:
    st.subheader("Why look from space at all?")
    st.write("AIS transponders are how PortWatch and every ship-tracking service count transits. In 2026 many "
             "tankers crossed the strait with AIS switched off. A satellite image sees a ship whether or not it "
             "broadcasts, so the gap between the satellite count and the AIS count measures the oil that "
             "ship tracking missed.")
    if pw is not None:
        d = pw.set_index("date")["n_tanker"].rolling(7, min_periods=3).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d.values, mode="lines", line=dict(color=theme.PINK, width=1.8),
                                 name="Tanker transits, 7-day mean", hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra></extra>"))
        fig.add_vline(x=CLOSURE, line=dict(color=theme.ORANGE, width=1, dash="dash"))
        fig.add_annotation(x=CLOSURE, y=1, yref="paper", text="28 Feb 2026", showarrow=False, xanchor="left",
                           font=dict(size=11, color=theme.INK_2))
        fig.update_layout(yaxis_title="Tankers per day (AIS)", showlegend=False)
        theme.show(fig, height=260)
        st.caption("IMF PortWatch daily chokepoint transits, Strait of Hormuz.")
    else:
        theme.pending("PortWatch transits are not downloaded yet.",
                      "cd satellite\npython s07_calibrate_and_export.py   # also saves data/series/portwatch_hormuz_daily.csv")

t_img, t_det, t_ser = st.tabs(["One image", "The detector", "Supply series"])

with t_img:
    qls = L.sat_quicklooks()
    if qls:
        pick = st.select_slider("Image date", options=qls, value=qls[len(qls) // 2], format_func=lambda p: p.stem)
        st.image(str(pick), width="stretch",
                 caption="Detections on one Sentinel-2 image: red = 150 m or longer, orange = 60-150 m, "
                         "cyan = smaller craft; white outlines are the transit lanes and the anchorage.")
    else:
        prev = L.sat_figure("labels_preview_ai2.png")
        if prev:
            st.image(str(prev), width="stretch", caption="Training chips with their ship labels (Ai2 dataset).")
        theme.pending("No detection quicklooks yet.",
                      "cd satellite\npython s05_detect_scenes.py --ckpt runs/hormuz_ft/best.pt --quicklook")

with t_det:
    runs = L.detector_runs()
    if runs.empty:
        theme.pending("No trained ship detector yet.",
                      "cd satellite\npython s04_train_detector.py --sources ai2 --backbone coco --tag ai2_coco\n"
                      "python s04_train_detector.py --sources hormuz --init runs/ai2_coco/best.pt --tag hormuz_ft")
    else:
        show = runs[["run", "backbone", "sources", "epochs", "n_train", "val_ap", "val_f1", "val_count_mae",
                     "test_ap", "test_f1", "test_count_mae"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "run": "Run", "backbone": "Pretrained on", "sources": "Labels", "epochs": "Best epoch",
            "n_train": "Training chips",
            **{c: st.column_config.NumberColumn(c.replace("_", " ").replace("count mae", "count MAE")
                                                .replace("ap", "AP").replace("f1", "F1"), format="%.3f")
               for c in show.columns if c.startswith(("val_", "test_"))}})
        st.caption("A ship counts as found when the detection's centre is within 100 m of a label. Count MAE is "
                   "the average miscount per 5 x 5 km chip, the error that matters for the supply series. "
                   "Test = Hormuz chips from dates the detector never saw.")
        run = st.selectbox("Training curves", runs.run.tolist())
        hist = L.detector_history(run)
        if hist is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.epoch, y=hist.train_total, name="training loss", yaxis="y2",
                                     line=dict(color=theme.MUTED, width=1.5)))
            for k, c in (("val_ap", theme.AQUA), ("val_f1", theme.BLUE)):
                fig.add_trace(go.Scatter(x=hist.epoch, y=hist[k], name=k.replace("val_", "validation ").upper()
                                         .replace("VALIDATION", "validation"), mode="lines+markers",
                                         line=dict(color=c, width=2)))
            fig.update_layout(xaxis_title="Epoch", yaxis=dict(title="Score", range=[0, 1]),
                              yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False))
            theme.show(fig, height=280)

with t_ser:
    w = L.sat_vs_ais()
    sat = L.sat_series()
    if w is not None:
        cal = L.sat_calibration()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=w.index, y=w["ais_flow_mbd"], name="AIS transits (PortWatch)", mode="lines",
                                 line=dict(color=theme.PINK, width=1.6)))
        fig.add_trace(go.Scatter(x=w.index, y=w["sat_flow_mbd"], name="Satellite CNN estimate", mode="lines+markers",
                                 line=dict(color=theme.AQUA, width=2.2), marker=dict(size=4)))
        fig.add_vline(x=CLOSURE, line=dict(color=theme.ORANGE, width=1, dash="dash"))
        fig.update_layout(yaxis_title="Million barrels per day (tanker capacity)", hovermode="x unified")
        theme.show(fig, height=330)
        post = w[w.index >= CLOSURE]
        if len(post):
            theme.figures([
                (f"{cal.get('r2', float('nan')):.2f}" if not cal.get("fallback") else "n/a",
                 "R² of the 2023-2025 calibration against AIS"),
                (f"{post.sat_flow_mbd.mean():.2f}", "mb/d seen by satellite since 28 Feb 2026"),
                (f"{post.ais_flow_mbd.mean():.2f}", "mb/d visible on AIS since 28 Feb 2026"),
            ])
            theme.callout("The gap between the two lines after the closure is oil the satellite saw but ship "
                          "tracking did not: the dark-fleet effect.")
    elif sat is not None:
        fig = go.Figure(go.Scatter(x=sat.date, y=sat.transit_n_large, mode="markers+lines",
                                   line=dict(color=theme.AQUA), name="Large ships in the transit lanes"))
        fig.update_layout(yaxis_title="Ships of 150 m or more per image")
        theme.show(fig, height=300)
        theme.pending("Supply series not calibrated against PortWatch yet.",
                      "cd satellite\npython s07_calibrate_and_export.py --ais-lag 3")
    else:
        theme.pending("No supply series yet.", "cd satellite\npython s05_detect_scenes.py --ckpt runs/hormuz_ft/best.pt\n"
                                               "python s06_build_supply_series.py\npython s07_calibrate_and_export.py")

theme.footer(KEY)
