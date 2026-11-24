"""Two papers, two answers, and the experiment that tests both."""
import html

import streamlit as st

import content
import loaders as L
import theme

KEY = "03_question"
theme.header(KEY)

cols = st.columns(2, gap="large")
for col, p in zip(cols, content.PAPERS):
    with col:
        st.html(f"""
        <div class="paper">
          <div class="who">{html.escape(p['short'])}</div>
          <div class="venue"><a href="{p['url']}" target="_blank">{html.escape(p['venue'])}</a></div>
          <dl>
            <dt>Forecasts</dt><dd>{html.escape(p['target'])}</dd>
            <dt>Approach</dt><dd>{html.escape(p['method'])}</dd>
            <dt>Result</dt><dd>{html.escape(p['finding'])}</dd>
          </dl>
          <div class="finding">{html.escape(p['lesson'])}</div>
        </div>""")

st.space("medium")
st.html(f'<div class="callout"><p class="question">{html.escape(content.RESEARCH_QUESTION)}</p></div>')

st.subheader("Three input sets put both papers in one experiment")
st.write("The target's own history is always included. Pick a horizon to see which drivers SHAP chose for it.")

cfg = L.config()
target = cfg["data"]["target"]
all_cols = list(L.levels().columns)
drivers = [c for c in all_cols if c != target]
hs = L.shap_horizons()

h = st.segmented_control("SHAP selection for horizon", hs, default=hs[0] if hs else None,
                         format_func=lambda x: f"{x} day" + ("s" if x > 1 else ""), key="q_h") if hs else None
selected = L.shap_selection(h).get("selected", []) if h else []

rows = [
    ("price_only", "Price only", "Only WTI's own past returns.", [target]),
    ("all", "All features", f"WTI plus all {len(drivers)} candidate drivers (paper 1's idea).", [target] + drivers),
    ("shap_selected", "SHAP-selected", f"WTI plus the top {cfg['shap']['top_k']} drivers by mean |SHAP|, "
                                        "chosen on train and validation data only (paper 2's idea).",
     [target] + selected),
]
for key, name, desc, used in rows:
    c1, c2 = st.columns([1, 2.6], vertical_alignment="center")
    with c1:
        st.html(f'<div style="border-left:4px solid {theme.FEATURE_SET_COLORS[key]};padding-left:.7rem">'
                f'<strong>{name}</strong><br><span style="color:{theme.INK_2};font-size:.9rem">{html.escape(desc)}</span></div>')
    with c2:
        theme.chips([L.label(c) for c in [target] + drivers],
                    on={L.label(c): theme.FEATURE_SET_COLORS[key] for c in used})

st.subheader("Hypotheses")
v = L.verdicts()
for hyp in content.HYPOTHESES:
    kind, lab, why = v.get(hyp["key"], ("pending", "Pending", ""))
    st.html(f"<p><strong>{hyp['key']}</strong>&nbsp; {html.escape(hyp['text'])} {theme.verdict(kind, lab)}</p>")
st.caption("Verdicts are computed from the current results files and update as more runs finish. "
           "Details on the conclusions slide.")

theme.footer(KEY)
