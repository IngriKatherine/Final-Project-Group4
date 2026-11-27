"""Our approach: four model families, how they connect, and the hypotheses (verdicts computed live)."""
import html

import streamlit as st

import content
import loaders as L
import theme

KEY = "03_approach"
theme.header(KEY)

st.html(f'<div class="callout"><p class="question">{html.escape(content.GOAL)}</p></div>')

cols = st.columns(len(content.FAMILIES), gap="medium")
for col, fam in zip(cols, content.FAMILIES):
    done = any(any(L.CODE_DIR.glob(p)) for p in fam["files"])
    chip = theme.verdict("yes", "results in") if done else theme.verdict("pending", "in progress")
    who = content.ROLES.get(fam["key"], "")
    with col:
        st.html(f"""
        <div class="family" style="--fam:{fam['color']}">
          <div class="name">{html.escape(fam['name'])} {chip}</div>
          <div class="models">{html.escape(fam['models'])}</div>
          <dl>
            <dt>Reads</dt><dd>{html.escape(fam['input'])}</dd>
            <dt>Predicts</dt><dd>{html.escape(fam['output'])}</dd>
          </dl>
          <div class="who">{html.escape(who) if who and not who.startswith('[') else ''}</div>
        </div>""")

st.subheader("One price, four views of it")
st.graphviz_chart(f"""
digraph {{
  rankdir=LR; nodesep=0.22; ranksep=0.35;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="{theme.RULE}", fontname="Helvetica",
        fontsize=11, fontcolor="{theme.INK}", margin="0.16,0.07"];
  edge [color="{theme.MUTED}", arrowsize=0.6];
  px [label="WTI / Brent\\nprice history"];
  mk [label="12 market drivers\\n(dollar, rates, VIX, ...)"];
  img [label="Price window\\ndrawn as an image"];
  s2 [label="Sentinel-2 images\\nStrait of Hormuz"];
  pw [label="IMF PortWatch\\ntanker transits (labels)"];
  st [label="Random walk\\nARIMA / ARIMAX", color="{theme.BASELINE_COLORS['arima']}"];
  rn [label="LSTM / GRU", color="{theme.BLUE}"];
  ic [label="Image CNN\\n(up or down?)", color="{theme.YELLOW}"];
  sc [label="Ship-detector CNN\\noil-supply series", color="{theme.AQUA}"];
  fc [label="Price forecast\\n2022-2025 test, 2026 hold-out", fillcolor="{theme.CRUDE_SOFT}"];
  px -> st; mk -> st; px -> rn; mk -> rn; px -> img -> ic; s2 -> sc; pw -> sc; sc -> rn [label=" 2026", fontsize=10];
  st -> fc; rn -> fc; ic -> fc;
}}""", width="stretch")
st.caption("Every model is scored on the same test days and against the same benchmark: the random walk.")

st.subheader("Hypotheses")
v = L.verdicts()
for hyp in content.HYPOTHESES:
    kind, lab, why = v.get(hyp["key"], ("pending", "Pending", ""))
    st.html(f"<p><strong>{hyp['key']}</strong>&nbsp; {html.escape(hyp['text'])} {theme.verdict(kind, lab)}</p>")
st.caption("Verdicts are computed from the current results files and update as runs finish. Details on the conclusions slide.")

theme.footer(KEY)
