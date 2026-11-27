# Code

All code for the group project: forecasting the WTI crude oil price with ARIMA, ARIMAX, LSTM and
GRU, a CNN that reads price-chart images, and a satellite CNN that measures oil supply through the
Strait of Hormuz for the 2026 forecasts. Scripts are numbered in the order they must be run.
Every setting (dates, tickers, splits, horizons, model sizes) lives in [`config.yaml`](config.yaml)
(satellite track: [`satellite/config_satellite.yaml`](satellite/config_satellite.yaml)), so you should
never need to edit numbers inside a script.

## Setup

```bash
cd Code
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build (AWS); omit on CPU
pip install -r requirements.txt
```

Run every script from inside `Code/`.

## Run order

| # | Script | What it does | Main outputs | Owner |
|---|--------|--------------|--------------|-------|
| 1 | `01_fetch_data.py` | Downloads WTI, Brent, rates and the broad dollar index from FRED, plus DXY, gold, S&P 500, VIX, natural gas and XLE from Yahoo Finance | `data/raw/fred.csv`, `data/raw/yahoo.csv` | Data & baselines |
| 2 | `02_preprocess.py` | Aligns calendars, fills short gaps, handles the negative WTI price of 20 April 2020, trims to the common sample, writes descriptive statistics | `data/processed/levels.csv`, `results/tables/descriptive_stats.csv`, `correlations.csv` | Data & baselines |
| 3 | `03_baselines.py` | Random walk, rolling-origin ARIMA and ARIMAX (lagged drivers as exogenous inputs) on the same test origins the networks use; ARIMA order fixed or chosen by `auto_arima` on train | `results/predictions/baseline_*.csv`, `results/tables/baselines_metrics.csv`, `baselines_arima_order.json` | Data & baselines |
| 4 | `04_train_rnn.py` | Trains one LSTM or GRU for a given feature set, horizon and seed. Other scripts call it in loops | `results/models/*.pt`, `results/predictions/*.csv`, `results/tables/runs/*.json` | Modeling |
| 5 | `05_shap_analysis.py` | Ranks the drivers on validation data and saves a reduced input set (`shap_selected`) per horizon. Needed by 06 only while `shap_selected` is in `experiments.feature_sets` | `results/shap/*` | Modeling |
| 6 | `06_run_experiments.py` | Main grid: {LSTM, GRU} × {price only, all drivers, reduced set} × horizons × seeds | `results/tables/main_results.csv` | Modeling |
| 7 | `07_ablations.py` | One-factor ablations: hidden size, layers, lookback, dropout | `results/tables/ablations.csv`, `results/figures/ablation_*.png` | Modeling |
| 8 | `08_make_figures_tables.py` | Final comparison against the baselines, Diebold–Mariano tests, report figures | `results/tables/final_comparison.*`, `dm_tests.csv`, `results/figures/*` | Shared |
| – | `sample_start_check.py` | Does dropping USD_BROAD (and starting the sample in 2003) change accuracy? Decided on validation data only | `results/sample_start_check/*` | Data & baselines |
| – | image CNN script *(in progress)* | CNN on price-chart images (Ren et al., 2024): each window of prices drawn as a picture, predicts up / down | `results/predictions/imgcnn_price_only_h<h>_s<seed>.csv` | Image CNN |
| – | `satellite/s00`–`s07` | Strait of Hormuz track: Sentinel-2 images → ship detector (Faster R-CNN) → oil-supply series calibrated with IMF PortWatch → `data/processed/satellite_features.csv` | see [`satellite/README.md`](satellite/README.md) | Satellite |
| – | 2026 forecasts | LSTM / GRU on 2026 with price only, + AIS supply, + satellite supply, + both | `results/predictions_2026/*.csv` | Satellite + Modeling |

Quick version:

```bash
python 01_fetch_data.py
python 02_preprocess.py
python 03_baselines.py
python 04_train_rnn.py --model lstm --features all --horizon 1    # smoke test of one run
python 05_shap_analysis.py                 # reduced input set (skip if shap_selected is not in feature_sets)
python 06_run_experiments.py
python 07_ablations.py --features shap_selected --horizon 5
python 08_make_figures_tables.py
```

### Forecast file format

Every model writes one CSV per run with the columns
`origin_date, target_date, base_price, true_price, pred_price`, so the evaluation, the
Diebold–Mariano tests and the presentation treat all models the same way:

| Model | File |
|-------|------|
| Random walk, ARIMA, ARIMAX | `results/predictions/baseline_(rw\|arima\|arimax)_h<h>.csv` |
| LSTM, GRU | `results/predictions/(lstm\|gru)_<inputs>_h<h>_s<seed>.csv` |
| Image CNN | `results/predictions/imgcnn_price_only_h<h>_s<seed>.csv` (optional `prob_up` column) |
| 2026 forecasts | `results/predictions_2026/<model>_<inputs>_h<h>_s<seed>.csv`, inputs `price_only`, `ais`, `satellite`, `satellite_ais` |

### Satellite track and 2026

```bash
cd satellite
pip install -r requirements_satellite.txt
python s00_check_setup.py          # packages, data access, PortWatch
# s01 search → s02 download → s03 labels → s04 train detector → s05 detect → s06 series → s07 calibrate
python s07_calibrate_and_export.py # writes ../data/processed/satellite_features.csv
```

For 2026 prices, set `data.end_date` in `config.yaml` to a 2026 date and rerun `01_fetch_data.py`
and `02_preprocess.py`. Full commands, data sources and experiments are in
[`satellite/README.md`](satellite/README.md).

Scripts 04–07 skip runs that already have results, so you can stop and restart them.
Use `--force` to retrain.

## Presentation app

The final presentation is an interactive Streamlit site that reads everything above from
`data/`, `results/` and `satellite/`. Slides whose results are not ready yet say which command or
file fills them. From this folder:

```bash
pip install -r presentation/requirements-presentation.txt
streamlit run presentation/app.py
```

See [`presentation/README.md`](presentation/README.md) for the slides and presenter tools.

## Shared package `oilfc/`

| Module | Contents |
|--------|----------|
| `utils.py` | config loading and overrides, seeding, device selection, paths, run names |
| `data.py` | transforms (log returns / differences), targets, feature-set resolution, chronological split, train-only scaling, sliding windows |
| `models.py` | `RNNForecaster`: stacked LSTM or GRU with a linear head on the last hidden state |
| `train.py` | Adam training loop with early stopping, LR schedule, gradient clipping; `predict` |
| `metrics.py` | RMSE, MAE, MAPE, directional accuracy, Diebold–Mariano test with the HLN correction |

## Design decisions (and why)

* **Target = h-step log return, metrics on prices.** A network trained on price levels mostly learns
  "tomorrow ≈ today". Predicting the return and converting back with `P_t · exp(ŷ)` makes the
  comparison with the random walk honest. Every metric is computed in USD/bbl.
* **Direct multi-horizon strategy.** One model per horizon (1, 5, 20 business days by default).
* **No look-ahead.** Samples are assigned to a split by their *target* date, and scalers are fit on
  training rows only. The reduced input set is chosen on train and validation data only. The satellite
  features use only images taken on or before each day, and the PortWatch calibration uses 2023–2025
  only, so 2026 stays unseen.
* **Same test origins for every model**, so Diebold–Mariano tests compare like with like.
* **Several seeds per configuration.** Results are reported as mean ± std.
* **Satellite vs. AIS supply.** In 2026 many tankers crossed Hormuz with AIS switched off, so the 2026
  study compares price + satellite supply against price + AIS supply (PortWatch) on the same days.

## Folder layout

```
Code/
├── config.yaml
├── requirements.txt
├── 01_fetch_data.py … 08_make_figures_tables.py
├── sample_start_check.py
├── oilfc/                # shared package
├── satellite/            # Strait of Hormuz track (s00–s07, hormuz/ package, tests/)
├── presentation/         # Streamlit presentation
├── data/
│   ├── raw/              # downloaded CSVs (01)
│   └── processed/        # levels.csv (02), satellite_features.csv (satellite/s07)
├── results/
│   ├── tables/           # CSV/Markdown tables; runs/ = one JSON per training run
│   ├── figures/          # PNGs for the report and slides
│   ├── predictions/      # test-period forecasts, one CSV per run
│   ├── predictions_2026/ # 2026 forecasts with satellite / AIS supply
│   ├── models/           # .pt checkpoints (git-ignored)
│   └── shap/             # (optional) driver ranking for the reduced input set
└── notebooks/            # exploratory work only; final code goes in the scripts
```
