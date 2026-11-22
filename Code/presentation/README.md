# Presentation app

The group's final presentation as an interactive Streamlit website: 14 slides in five
sections, every chart and number read live from what the pipeline scripts wrote into
`Code/data/` and `Code/results/`.

## Run it

```bash
cd Code
pip install -r requirements.txt -r presentation/requirements-presentation.txt
streamlit run presentation/app.py
```

Start it **from the `Code/` folder** so Streamlit picks up the theme in `Code/.streamlit/config.toml`.
The browser opens at <http://localhost:8501>. If the `streamlit` command is not found, use
`python -m streamlit run presentation/app.py`.

## Presenting

* **Left / right arrow keys** (or Page Up / Page Down, which most clickers send) move between slides.
  The links at the bottom of each slide do the same.
* The **sidebar** lists every slide by section and has the presenter tools:
  * **Speaker notes**: shows each slide's notes, owner and time budget at the bottom of the page.
  * **Start timer**: elapsed time, turning to "wrap up" at 12 min and "over time" at 15 min.
  * **Reload data**: clears the cache after you re-run a pipeline script.
* Collapse the sidebar (the `<<` at its top) for a cleaner view on the projector.
* Fonts come from Google Fonts; without internet the app falls back to a system sans-serif.

## Before the talk

1. Put the real names, roles and the GitHub link in `content.py` (`TEAM`, `ROLES`, `REPO_URL`).
2. Run the rest of the pipeline. Slides that need missing results say what is missing and which
   command produces it, then fill in automatically:
   * `python 06_run_experiments.py`: GRU and the price-only / SHAP-selected input sets
     (main results, significance tests, hypothesis verdicts, forecast explorer)
   * `python 07_ablations.py --features shap_selected --horizon 5`: the ablations slide
3. Copy `results/models/*.pt` to the machine you present from (they are git-ignored). Only the
   Model lab slide needs them.
4. Rewrite headlines or notes in `content.py` once the final numbers are in.

## The slides

| # | Slide | What is interactive |
|---|-------|---------------------|
| 1 | Title | Hover the 20-year WTI line (negative print included) |
| 2 | Why oil prices? | Range buttons, event markers, autocorrelation of returns |
| 3 | Two papers, one question | Pick a horizon to see which drivers SHAP selected; live hypothesis verdicts |
| 4 | Data | Sources table, series explorer (indexed / levels / daily change), stats, correlation heatmap |
| 5 | Preprocessing | Pipeline diagram, the April 2020 negative price, a sliding-window demo |
| 6 | Baselines | Random walk vs. rolling ARIMA by horizon and up close |
| 7 | LSTM and GRU | Equations, architecture, parameter calculator, one-cell gate playground |
| 8 | Training | Learning curves per seed, table of every run |
| 9 | SHAP | Importance per horizon, lag heatmap, rank changes across horizons, seed agreement |
| 10 | Main results | Skill / RMSE / MAE / MAPE / direction by input set, full table, Diebold-Mariano tests |
| 11 | Ablations | Depth, width, lookback, dropout (validation vs. test) |
| 12 | Forecast explorer | Actual vs. predicted, zoom presets, rolling RMSE, error spread, single-day inspector |
| 13 | Model lab | Live PyTorch inference: input heatmap, shock one input, sensitivity vs. SHAP |
| 14 | Conclusions | Computed verdicts, limitations, next steps, references, questions |

## Files

| File | Contents |
|------|----------|
| `app.py` | Navigation (`st.navigation`), sidebar, presenter timer (`st.fragment`) |
| `content.py` | Names, headlines, speaker notes, papers, references: **edit the story here** |
| `loaders.py` | Every number on every slide, cached with `st.cache_data` / `st.cache_resource`; reuses `oilfc.metrics` for metrics and Diebold-Mariano tests and `oilfc.models` for the checkpoints |
| `theme.py` | Colours, type, Plotly template, progress rail, previous/next links, keyboard navigation |
| `slides/*.py` | One file per slide |
| `../.streamlit/config.toml` | Theme colours and font |

The structure follows the course's Streamlit examples in `Data-Visualization/streamlit`
(page separate from the work, numbered steps, `@st.cache_data` / `@st.cache_resource`,
`st.fragment`, `column_config`, Plotly charts).
