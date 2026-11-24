"""Conclusions: verdict per hypothesis (computed), lessons, limitations, next steps, references, questions."""
import html

import pandas as pd
import streamlit as st

import content
import loaders as L
import theme

KEY = "14_conclusions"
theme.header(KEY)

v = L.verdicts()
st.subheader("The hypotheses, judged by the current results")
for hyp in content.HYPOTHESES:
    kind, lab, why = v.get(hyp["key"], ("pending", "Pending", ""))
    st.html(f"""<div style="padding:.7rem 0;border-bottom:1px solid {theme.RULE}">
      <strong>{hyp['key']}</strong>&nbsp; {html.escape(hyp['text'])} {theme.verdict(kind, lab)}
      <div style="color:{theme.INK_2};font-size:.95rem;margin-top:.25rem">{html.escape(why)}</div></div>""")

comp = L.comparison()
rnn = comp[comp.model.isin(["lstm", "gru"])]
if not rnn.empty:
    beats = rnn[rnn.Skill > 0]
    best = rnn.loc[rnn.Skill.idxmax()]
    if beats.empty:
        msg = (f"No network configuration run so far beats the random walk on test RMSE. The closest is "
               f"{best.config} at {best.horizon} day{'s' if best.horizon > 1 else ''} ({best.Skill:+.1f}%).")
    else:
        msg = (f"{len(beats)} of {len(rnn)} network configurations beat the random walk on test RMSE; the best is "
               f"{best.config} at {best.horizon} day{'s' if best.horizon > 1 else ''} ({best.Skill:+.1f}%).")
    theme.callout(html.escape(msg) + " Check the significance tab on the main results slide before calling any "
                                     "difference real.")

st.text_area("Our answer to the research question (type it in before presenting; it is not saved)",
             placeholder="e.g. On crude oil at daily horizons, SHAP-guided selection ... compared with ...",
             height=90, key="answer")

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Limitations")
    for x in content.LIMITATIONS:
        st.markdown(f"- {x}")
with b:
    st.subheader("Next steps")
    for x in content.FUTURE_WORK:
        st.markdown(f"- {x}")

with st.expander("References", icon=":material/menu_book:"):
    for i, r in enumerate(content.REFERENCES, 1):
        st.markdown(f"{i}. {r}")

st.space("medium")
st.html(f"""<div class="title-hero"><h1 style="max-width:none">Questions?</h1>
<div class="by">{html.escape(content.GROUP)}: {", ".join(f"<strong>{html.escape(n)}</strong>" for n in content.TEAM)}<br>
Code and results: <a href="{html.escape(content.REPO_URL)}" target="_blank">{html.escape(content.REPO_URL)}</a></div></div>""")

theme.footer(KEY)
