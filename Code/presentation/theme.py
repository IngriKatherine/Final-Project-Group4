"""
Look and feel: colours, type, the Plotly template, and the slide chrome
(progress rail at the top, previous/next links and arrow-key navigation at the bottom).

Design notes
  * Light "assay sheet" surface so charts stay readable on a projector.
  * One type family, Archivo, used with its width axis: headlines are set wide
    (font-stretch 118%), body text at normal width. Figures use tabular numbers.
  * Data colours follow the entity, never its rank: a feature set keeps its colour
    on every slide. Baselines are grey (they are the reference, not the story).
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import content

# ----------------------------------------------------------------- palette ---
PAGE = "#F3F5F6"
SURFACE = "#FFFFFF"
INK = "#15212B"
INK_2 = "#4E5A64"
MUTED = "#86919A"
RULE = "#E1E6E9"
CRUDE = "#A85D0C"        # the deck's one accent: progress rail, current slide
CRUDE_SOFT = "#F4E6D3"

# Categorical slots (validated for colour-vision deficiency, adjacent pairs).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

FEATURE_SET_COLORS = {"price_only": BLUE, "all": ORANGE, "shap_selected": AQUA}
BASELINE_COLORS = {"rw": "#9AA4AB", "arima": "#5F6B74"}
MODEL_DASH = {"lstm": "solid", "gru": "dot"}
SERIES_COLORS = {"WTI": BLUE, "BRENT": ORANGE}

# Sequential (magnitude) and diverging (polarity) ramps.
SEQ_BLUE = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#1c5cab"], [0.25, "#86b6ef"], [0.5, "#f0efec"], [0.75, "#f29b98"], [1.0, "#b3261e"]]

SPLIT_FILL = {"train": "rgba(0,0,0,0)", "val": "rgba(168,93,12,0.075)", "test": "rgba(42,120,214,0.075)"}

FONT = "Archivo, 'Helvetica Neue', Arial, sans-serif"


def config_color(model: str, features: str) -> str:
    return BASELINE_COLORS.get(model) or FEATURE_SET_COLORS.get(features, INK)


def config_dash(model: str) -> str:
    return {"rw": "solid", "arima": "dash"}.get(model, MODEL_DASH.get(model, "solid"))


# ---------------------------------------------------------- plotly template ---
def _register_template():
    t = go.layout.Template()
    axis = dict(gridcolor=RULE, linecolor=RULE, zerolinecolor=RULE, tickcolor=RULE,
                tickfont=dict(color=INK_2, size=12), title=dict(font=dict(color=INK_2, size=12)),
                showline=True, ticks="outside", ticklen=4, automargin=True)
    t.layout = go.Layout(
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        colorway=[BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"],
        xaxis=axis, yaxis=axis,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=RULE, font=dict(family=FONT, color=INK, size=12)),
        margin=dict(l=12, r=12, t=40, b=12),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left"),
    )
    t.data.scatter = [go.Scatter(line=dict(width=2))]
    t.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    pio.templates["oil"] = t
    pio.templates.default = "oil"


_register_template()


def show(fig: go.Figure, height: int | None = None, key: str | None = None):
    """Every chart goes through here so they share size, toolbar and theme."""
    if height:
        fig.update_layout(height=height)
    # Streamlit fills in its own secondary colour unless the figure sets these itself.
    if fig.layout.plot_bgcolor is None:
        fig.update_layout(plot_bgcolor=SURFACE)
    if fig.layout.paper_bgcolor is None:
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def add_split_bands(fig: go.Figure, start=None, end=None, labels: bool = True, row=None, col=None):
    """Shade the train / validation / test periods behind a time-series chart."""
    import loaders

    train_end, val_end = loaders.split_dates()
    lv = loaders.levels()
    start = start or lv.index[0]
    end = end or lv.index[-1]
    spans = [("train", start, train_end, "Train"), ("val", train_end, val_end, "Validation"),
             ("test", val_end, end, "Test")]
    for key, a, b, name in spans:
        if b <= a:
            continue
        kw = dict(row=row, col=col) if row else {}
        fig.add_vrect(x0=a, x1=b, fillcolor=SPLIT_FILL[key], line_width=0, layer="below",
                      annotation_text=name if labels else None, annotation_position="top left",
                      annotation_font=dict(size=11, color=INK_2), **kw)


# --------------------------------------------------------------------- CSS ---
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:ital,wdth,wght@0,62..125,100..900;1,62..125,100..900&display=swap');

:root {{
  --ink: {INK}; --ink-2: {INK_2}; --muted: {MUTED}; --rule: {RULE};
  --crude: {CRUDE}; --crude-soft: {CRUDE_SOFT}; --surface: {SURFACE};
}}
html, body, .stApp, .stMarkdown, button, input, textarea, select, [data-testid="stSidebar"] {{
  font-family: {FONT};
}}
.stApp {{ background: {PAGE}; color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; height: 2.6rem; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1280px; }}
p, li {{ line-height: 1.55; }}
.stMarkdown p, .stMarkdown li {{ max-width: 78ch; }}

/* Headings: the type itself carries the personality. */
h1, h2, h3 {{ color: var(--ink); letter-spacing: -0.012em; }}
.slide-headline {{
  font-stretch: 118%; font-weight: 760; font-size: clamp(2rem, 3.4vw, 3.05rem);
  line-height: 1.04; margin: 0.35rem 0 0.55rem; max-width: 22ch; text-wrap: balance;
}}
.slide-lede {{ font-size: 1.14rem; color: var(--ink-2); max-width: 70ch; margin: 0 0 1.2rem; }}
.block-container h3 {{ font-stretch: 108%; font-weight: 680; font-size: 1.28rem; margin-top: 0.6rem; }}
.block-container h4 {{ font-stretch: 104%; font-weight: 650; font-size: 1.05rem; }}

/* Progress rail: one segment per section, one tick per slide. */
.rail {{ display: flex; gap: 10px; margin: 0 0 0.4rem; }}
.rail-sec {{ display: flex; flex-direction: column; gap: 5px; }}
.rail-ticks {{ display: flex; gap: 3px; }}
.rail-tick {{ height: 5px; flex: 1; border-radius: 3px; background: var(--rule); min-width: 14px; }}
.rail-tick.done {{ background: #D9B486; }}
.rail-tick.now {{ background: var(--crude); }}
.rail-label {{ font-size: 0.74rem; color: var(--muted); white-space: nowrap; }}
.rail-label.now {{ color: var(--ink); font-weight: 600; }}
.rail-count {{ margin-left: auto; align-self: flex-end; font-size: 0.78rem; color: var(--ink-2);
  font-variant-numeric: tabular-nums; white-space: nowrap; }}

/* Small building blocks */
.figure-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0;
  border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule); margin: 0.4rem 0 1rem; }}
.figure {{ padding: 0.8rem 1rem 0.75rem 0; }}
.figure + .figure {{ padding-left: 1rem; border-left: 1px solid var(--rule); }}
.figure .v {{ font-stretch: 112%; font-weight: 700; font-size: 1.85rem; line-height: 1.05;
  font-variant-numeric: tabular-nums; color: var(--ink); }}
.figure .k {{ font-size: 0.86rem; color: var(--ink-2); margin-top: 0.25rem; }}

.callout {{ border-left: 3px solid var(--crude); background: var(--crude-soft); padding: 0.85rem 1.1rem;
  border-radius: 0 6px 6px 0; margin: 0.3rem 0 1rem; }}
.callout p {{ margin: 0; }}
.question {{ font-stretch: 104%; font-size: 1.32rem; line-height: 1.35; font-weight: 520; max-width: 60ch; }}

.chip {{ display: inline-block; padding: 0.12rem 0.55rem; margin: 0 0.3rem 0.35rem 0; border-radius: 999px;
  font-size: 0.84rem; border: 1px solid var(--rule); background: var(--surface); color: var(--ink); }}
.chip.on {{ border-color: transparent; color: #fff; }}
.verdict {{ display: inline-block; font-size: 0.8rem; font-weight: 650; padding: 0.1rem 0.55rem;
  border-radius: 4px; margin-left: 0.4rem; vertical-align: 2px; }}
.verdict.yes {{ background: #DDF2E6; color: #0B5E36; }}
.verdict.no {{ background: #FBE3E1; color: #8E1F17; }}
.verdict.mixed {{ background: #FBEFD5; color: #7A4D00; }}
.verdict.pending {{ background: #E9EDF0; color: var(--ink-2); }}

.paper {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 1.1rem 1.2rem; height: 100%; }}
.paper .who {{ font-stretch: 110%; font-weight: 700; font-size: 1.15rem; }}
.paper .venue {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.7rem; }}
.paper dl {{ display: grid; grid-template-columns: 6.5rem 1fr; gap: 0.35rem 0.8rem; margin: 0; }}
.paper dt {{ color: var(--ink-2); font-size: 0.88rem; }}
.paper dd {{ margin: 0; font-size: 0.95rem; }}
.paper .finding {{ margin-top: 0.8rem; padding-top: 0.7rem; border-top: 1px solid var(--rule); font-weight: 600; }}

.spec {{ display: grid; grid-template-columns: max-content 1fr; gap: 0.3rem 1.2rem; font-size: 0.98rem; }}
.spec dt {{ color: var(--ink-2); }}
.spec dd {{ margin: 0; }}

.title-hero h1 {{ font-stretch: 125%; font-weight: 800; font-size: clamp(2.3rem, 4.6vw, 4.1rem);
  line-height: 0.98; letter-spacing: -0.02em; max-width: 17ch; margin: 0.2rem 0 0.8rem; }}
.title-hero .by {{ font-size: 1.05rem; color: var(--ink-2); }}
.title-hero .by strong {{ color: var(--ink); font-weight: 650; }}

/* Navigation links at the bottom of each slide */
.st-key-slide_nav {{ border-top: 1px solid var(--rule); padding-top: 0.9rem; margin-top: 2rem; }}
.st-key-slide_nav [data-testid="stPageLink-NavLink"] {{ border: 1px solid var(--rule); background: var(--surface);
  padding: 0.35rem 0.9rem; border-radius: 6px; }}
.st-key-slide_nav [data-testid="stPageLink-NavLink"]:hover {{ border-color: var(--crude); }}
.nav-hint {{ font-size: 0.78rem; color: var(--muted); text-align: center; padding-top: 0.55rem; }}

[data-testid="stSidebar"] {{ background: #EAEEF0; }}
[data-testid="stMetricValue"] {{ font-stretch: 110%; font-variant-numeric: tabular-nums; }}
[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
:focus-visible {{ outline: 2px solid var(--crude) !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
@media (max-width: 640px) {{ .rail-label {{ display: none; }} .figure + .figure {{ border-left: 0; padding-left: 0; }} }}
</style>
"""

# Arrow keys move between slides (ignored while typing or while a slider has focus).
# Installed once on the page; it looks up the links at key-press time.
KEYS_JS = """
<script>
(function () {
  const doc = window.document;
  if (window.__oilDeckKeys) { doc.removeEventListener('keydown', window.__oilDeckKeys); }
  window.__oilDeckKeys = function (e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const a = doc.activeElement;
    if (a && (a.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(a.tagName) ||
              a.getAttribute('role') === 'slider' || a.getAttribute('role') === 'tab')) return;
    let want = null;
    if (e.key === 'ArrowRight' || e.key === 'PageDown') want = 'next';
    if (e.key === 'ArrowLeft' || e.key === 'PageUp') want = 'prev';
    if (!want) return;
    const link = doc.querySelector('.st-key-slide_nav .st-key-nav_' + want + ' a');
    if (link) { e.preventDefault(); link.click(); }
  };
  doc.addEventListener('keydown', window.__oilDeckKeys);
})();
</script>
"""


def inject_css():
    st.html(CSS)


# ------------------------------------------------------------ slide chrome ---
PAGES: dict = {}          # key -> st.Page, filled in by app.py


def _slide(key: str) -> tuple[int, dict]:
    for i, s in enumerate(content.SLIDES):
        if s["key"] == key:
            return i, s
    raise KeyError(key)


def rail(key: str) -> str:
    i, cur = _slide(key)
    parts = []
    for sec in content.SECTIONS:
        slides = [(j, s) for j, s in enumerate(content.SLIDES) if s["section"] == sec]
        ticks = "".join(
            f'<span class="rail-tick {"now" if j == i else "done" if j < i else ""}" title="{html.escape(s["nav"])}"></span>'
            for j, s in slides)
        now = " now" if sec == cur["section"] else ""
        parts.append(f'<div class="rail-sec" style="flex:{len(slides)}"><div class="rail-ticks">{ticks}</div>'
                     f'<span class="rail-label{now}">{sec}</span></div>')
    count = f'<span class="rail-count">{i + 1} of {len(content.SLIDES)}</span>'
    return f'<div class="rail" role="navigation" aria-label="Presentation progress">{"".join(parts)}{count}</div>'


def header(key: str, headline: str | None = None, lede: str | None = None):
    """Top of every slide: progress rail, the slide's claim, and one supporting sentence."""
    _, s = _slide(key)
    st.html(rail(key))
    st.html(f'<h1 class="slide-headline">{html.escape(headline or s["headline"])}</h1>')
    text = s["lede"] if lede is None else lede
    if text:
        st.html(f'<p class="slide-lede">{html.escape(text)}</p>')


def footer(key: str):
    """Previous / next links, speaker notes and keyboard navigation."""
    i, s = _slide(key)
    prev_s = content.SLIDES[i - 1] if i > 0 else None
    next_s = content.SLIDES[i + 1] if i + 1 < len(content.SLIDES) else None

    if st.session_state.get("show_notes"):
        speaker = content.ROLES.get(s["speaker"], "Everyone") if s["speaker"] != "all" else "Everyone"
        with st.expander(f"Speaker notes ({speaker}, about {s['minutes']:g} min)", expanded=True):
            st.write(s["notes"])

    with st.container(key="slide_nav"):
        c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="center")
        with c1:
            if prev_s and prev_s["key"] in PAGES:
                with st.container(key="nav_prev"):
                    st.page_link(PAGES[prev_s["key"]], label=f"Previous: {prev_s['nav']}",
                                 icon=":material/arrow_back:")
        with c2:
            st.html('<div class="nav-hint">Use the left and right arrow keys to move between slides</div>')
        with c3:
            if next_s and next_s["key"] in PAGES:
                with st.container(key="nav_next", horizontal=True, horizontal_alignment="right"):
                    st.page_link(PAGES[next_s["key"]], label=f"Next: {next_s['nav']}",
                                 icon=":material/arrow_forward:", icon_position="right")
    st.html(KEYS_JS, unsafe_allow_javascript=True)


# ------------------------------------------------------------- components ---
def figures(items: list[tuple[str, str]]):
    """A row of headline numbers: [(value, what it is), ...]."""
    cells = "".join(f'<div class="figure"><div class="v">{html.escape(v)}</div>'
                    f'<div class="k">{html.escape(k)}</div></div>' for v, k in items)
    st.html(f'<div class="figure-row">{cells}</div>')


def callout(markdown_html: str):
    st.html(f'<div class="callout"><p>{markdown_html}</p></div>')


def chips(items: list[str], on: dict[str, str] | None = None):
    """Feature chips; `on` maps an item to a fill colour to mark it as selected."""
    on = on or {}
    out = []
    for it in items:
        if it in on:
            out.append(f'<span class="chip on" style="background:{on[it]}">{html.escape(it)}</span>')
        else:
            out.append(f'<span class="chip">{html.escape(it)}</span>')
    st.html("<div>" + "".join(out) + "</div>")


def verdict(kind: str, text: str) -> str:
    return f'<span class="verdict {kind}">{html.escape(text)}</span>'


def pending(what: str, command: str):
    """Shown where a result does not exist yet: say what is missing and how to make it."""
    st.info(f"{what} Run this from the `Code/` folder, then press **Reload data** in the sidebar:",
            icon=":material/pending:")
    st.code(command, language="bash")
