"""Conclusions: verdict per hypothesis (computed), the headline numbers, limitations, next steps,
references, questions."""
import html

import streamlit as st

import content
import loaders as L
import theme

KEY = "15_conclusions"
theme.header(KEY)

v = L.verdicts()
st.subheader("The hypotheses, judged by the current results")
for hyp in content.HYPOTHESES:
    kind, lab, why = v.get(hyp["key"], ("pending", "Pending", ""))
    st.html(f"""<div style="padding:.7rem 0;border-bottom:1px solid {theme.RULE}">
      <strong>{hyp['key']}</strong>&nbsp; {html.escape(hyp['text'])} {theme.verdict(kind, lab)}
      <div style="color:{theme.INK_2};font-size:.95rem;margin-top:.25rem">{html.escape(why)}</div></div>""")

comp = L.comparison()
models = comp[~comp.model.isin(["rw"])] if not comp.empty else comp
msgs = []
if not models.empty and "Skill" in models:
    beats = models[models.Skill > 0]
    best = models.loc[models.Skill.idxmax()]
    hd = f"{best.horizon} day{'s' if best.horizon > 1 else ''}"
    msgs.append(f"2022-2025: {len(beats)} of {len(models)} model configurations beat the random walk on test RMSE; "
                f"the best is {best.config} at {hd} ({best.Skill:+.1f}%).")
c26 = L.comparison_2026()
if not c26.empty:
    b26 = c26[c26.model != "rw"].sort_values("Skill", ascending=False).iloc[0]
    msgs.append(f"2026: the best forecast is {b26.config} at {b26.horizon} day{'s' if b26.horizon > 1 else ''} "
                f"({b26.Skill:+.1f}% vs. the random walk).")
if msgs:
    theme.callout(html.escape(" ".join(msgs)) + " Check the significance tests before calling any difference real.")

st.text_area("Our answer (type it in before presenting; it is not saved)",
             placeholder="e.g. Price history alone does not beat the random walk; the satellite supply series ...",
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
<div class="by">{html.escape(content.GROUP)}: {" &middot; ".join(f"<strong>{html.escape(n)}</strong>" for n in content.TEAM)}<br>
Code and results: <a href="{html.escape(content.REPO_URL)}" target="_blank">{html.escape(content.REPO_URL)}</a></div></div>""")

theme.footer(KEY)
