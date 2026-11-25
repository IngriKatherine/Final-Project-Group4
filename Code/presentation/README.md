# Presentation app

The group's final presentation as an interactive Streamlit website: 15 slides in six sections
plus a backup slide, every chart and number read live from what the pipeline scripts wrote into
`Code/data/`, `Code/results/` and `Code/satellite/`.

**Goal: oil price forecasting.** Four families of models forecast the WTI crude price:

| Family | Models | Input | Results files |
|--------|--------|-------|---------------|
| Statistical | Random walk, ARIMA, ARIMAX | Price history (ARIMAX adds market drivers) | `results/predictions/baseline_(rw\|arima\|arimax)_h<h>.csv` |
| Recurrent networks | LSTM, GRU | 30-day windows, price only or with market drivers | `results/predictions/(lstm\|gru)_<inputs>_h<h>_s<seed>.csv` |
| Image CNN | CNN on price-chart images (Ren et al., 2024) | The price path drawn as a picture | `results/predictions/imgcnn_price_only_h<h>_s<seed>.csv` |
| Satellite CNN | Faster R-CNN ship detector, then LSTM / GRU | Sentinel-2 images of the Strait of Hormuz, calibrated with IMF PortWatch | `Code/satellite/...`, `data/processed/satellite_features.csv`, `results/predictions_2026/*.csv` |

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

1. In `content.py`, fill in who owns which part (`ROLES`: ARIMA / ARIMAX, LSTM / GRU, image CNN)
   and the GitHub link (`REPO_URL`). Team names are already in `TEAM`.
2. Run the pipeline. Slides that need missing results say what is missing and which command or
   file produces it, then fill in automatically after **Reload data**. Every prediction file has the
   columns `origin_date, target_date, base_price, true_price, pred_price`:
   * **ARIMAX**: `results/predictions/baseline_arimax_h<h>.csv` (same days as ARIMA).
   * **LSTM / GRU**: `python 06_run_experiments.py`.
   * **Image CNN**: `results/predictions/imgcnn_price_only_h<h>_s<seed>.csv`, optionally with a
     `prob_up` column (probability the price goes up). `pred_price` can be `base_price` moved in the
     predicted direction if the network only predicts up / down.
   * **Satellite track**: `Code/satellite` scripts s01 to s07 (see `Code/satellite/README.md`).
     They write the detector runs, the supply series, PortWatch transits and
     `data/processed/satellite_features.csv`.
   * **2026 forecasts**: `results/predictions_2026/<model>_<inputs>_h<h>_s<seed>.csv` with inputs
     `price_only`, `ais`, `satellite` or `satellite_ais`. For 2026 prices, set `data.end_date` in
     `Code/config.yaml` to a 2026 date and rerun the download and preprocessing scripts.
   * Ablations: `python 07_ablations.py --features shap_selected --horizon 5`.
3. Copy `results/models/*.pt` to the machine you present from (they are git-ignored). Only the
   Model lab slide (backup) needs them.
4. Rewrite headlines or notes in `content.py` once the final numbers are in.

## The slides

| # | Slide | What is interactive |
|---|-------|---------------------|
| 1 | Title | Hover the WTI line with train / validation / test periods and the 2026 Hormuz closure |
| 2 | Why oil prices? | Range buttons, event markers, autocorrelation of returns |
| 3 | Our approach | The four model families with status chips, overview diagram, live hypothesis verdicts |
| 4 | Data | Sources table, series explorer (indexed / levels / daily change), stats, correlation heatmap |
| 5 | Preprocessing | Pipeline diagram, the April 2020 negative price, a sliding-window demo |
| 6 | ARIMA and ARIMAX | Random walk vs. ARIMA vs. ARIMAX by horizon and up close |
| 7 | LSTM and GRU | Equations, architecture, parameter calculator, one-cell gate playground |
| 8 | Training | Learning curves per seed, table of every run |
| 9 | Price charts as images | Turn any price window into the CNN's input image, guess-the-direction quiz, image CNN results |
| 10 | Main results | Skill / RMSE / MAE / MAPE / direction for every model, full table, Diebold-Mariano tests |
| 11 | Ablations | Depth, width, lookback, dropout (validation vs. test) |
| 12 | Forecast explorer | Actual vs. predicted, zoom presets, rolling RMSE, error spread, single-day inspector |
| 13 | Ships from space | PortWatch transits, detection pipeline, image quicklooks, detector metrics, satellite vs. AIS supply |
| 14 | 2026 forecast | Price vs. satellite supply, four input sets, 2026 skill vs. random walk, DM tests, forecast plot |
| 15 | Conclusions | Computed verdicts, limitations, next steps, references, questions |
| 16 | Model lab (appendix) | Live PyTorch inference: input heatmap, shock one input, sensitivity |

## Files

| File | Contents |
|------|----------|
| `app.py` | Navigation (`st.navigation`), sidebar, presenter timer (`st.fragment`) |
| `content.py` | Names, roles, headlines, speaker notes, hypotheses, references: **edit the story here** |
| `loaders.py` | Every number on every slide, cached with `st.cache_data` / `st.cache_resource`; reuses `oilfc.metrics` for metrics and Diebold-Mariano tests and `oilfc.models` for the checkpoints |
| `theme.py` | Colours, type, Plotly template, progress rail, previous/next links, keyboard navigation |
| `slides/*.py` | One file per slide |
| `../.streamlit/config.toml` | Theme colours and font |

The structure follows the course's Streamlit examples in `Data-Visualization/streamlit`
(page separate from the work, numbered steps, `@st.cache_data` / `@st.cache_resource`,
`st.fragment`, `column_config`, Plotly charts).
