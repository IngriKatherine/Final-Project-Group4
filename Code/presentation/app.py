"""
Final presentation of Group 4's project as an interactive Streamlit website.

    cd Code
    streamlit run presentation/app.py

Fifteen slides plus a backup, built from the files the pipeline scripts write
(data/processed, results/tables, results/predictions, results/predictions_2026, results/models,
and the satellite track in satellite/data, satellite/runs, satellite/figures).
Nothing is typed in by hand, so re-running a pipeline script updates the slides:
press "Reload data" in the sidebar.

Structure (same idea as the course apps: the page is separate from the work)
    app.py       navigation, sidebar, presenter tools        <- this file
    content.py   names, headlines, speaker notes, references  <- edit the story here
    loaders.py   every number on every slide (cached)
    theme.py     colours, type, chart template, slide chrome
    slides/      one file per slide

Move between slides with the links at the bottom of each page or the left and right
arrow keys. The sidebar has a presenter timer and speaker notes.
"""
import time
from pathlib import Path

import streamlit as st

import content
import loaders
import theme

HERE = Path(__file__).resolve().parent

st.set_page_config(
    page_title=f"{content.GROUP}: crude oil forecasting",
    page_icon=":material/oil_barrel:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": f"**{content.PROJECT_TITLE}**  \n{content.COURSE}, {content.GROUP}"},
)
theme.inject_css()

# ------------------------------------------------------------- navigation ---
pages = {}
for i, s in enumerate(content.SLIDES):
    pages[s["key"]] = st.Page(HERE / "slides" / f"{s['key']}.py", title=s["nav"], icon=s["icon"],
                              url_path=s["key"].split("_", 1)[1], default=(i == 0))
theme.PAGES.update(pages)

nav = {sec: [pages[s["key"]] for s in content.SLIDES if s["section"] == sec] for sec in content.SECTIONS}
current = st.navigation(nav, position="sidebar", expanded=True)


# --------------------------------------------------------------- sidebar ---
@st.fragment(run_every=1)
def presenter_timer():
    """Re-runs on its own every second; the rest of the page is left alone (see the course's fragment lesson)."""
    start = st.session_state.get("timer_start")
    budget = sum(s["minutes"] for s in content.SLIDES if s["section"] != "Appendix")
    if start is None:
        st.caption(f"Talk budget: 12 to 15 min (slides add up to {budget:.0f} min)")
        return
    elapsed = time.time() - start
    mm, ss = divmod(int(elapsed), 60)
    status = "on time" if elapsed < 12 * 60 else "wrap up" if elapsed < 15 * 60 else "over time"
    st.metric("Elapsed", f"{mm:02d}:{ss:02d}", delta=status,
              delta_color="off" if status == "on time" else "inverse")
    st.progress(min(elapsed / (15 * 60), 1.0))


with st.sidebar:
    st.caption(f"{content.GROUP}, {content.COURSE}")
    st.divider()
    st.toggle("Speaker notes", key="show_notes", help="Shows the notes for each slide at the bottom of the page.")
    c1, c2 = st.columns(2)
    if c1.button("Start timer", icon=":material/timer:", width="stretch"):
        st.session_state.timer_start = time.time()
    if c2.button("Reset", icon=":material/restart_alt:", width="stretch"):
        st.session_state.pop("timer_start", None)
    presenter_timer()
    st.divider()
    if st.button("Reload data", icon=":material/refresh:", width="stretch",
                 help="Clear the cache after the pipeline scripts have written new results."):
        loaders.clear_all_caches()
        st.rerun()
    st.caption("Reads `Code/data`, `Code/results` and `Code/satellite`. Run with  \n"
               "`streamlit run presentation/app.py` from `Code/`.")

current.run()
