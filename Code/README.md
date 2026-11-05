# Code

All code for the group project. Scripts are numbered in the order they must be run.
Every setting (dates, tickers, splits, horizons, model sizes, SHAP options) lives in
[`config.yaml`](config.yaml), so you should never need to edit numbers inside a script.

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
| 3 | `03_baselines.py` | Random walk and rolling-origin ARIMA on the same test origins the networks use | `results/predictions/baseline_*.csv`, `results/tables/baselines_metrics.csv` | Data & baselines |
| 4 | `04_train_rnn.py` | Trains one LSTM or GRU for a given feature set, horizon and seed. Other scripts call it in loops | `results/models/*.pt`, `results/predictions/*.csv`, `results/tables/runs/*.json` | Modeling |
| 5 | `05_shap_analysis.py` | Explains the all-features LSTM with SHAP (on validation data), ranks the features and saves the selected subset per horizon | `results/shap/*`, `results/figures/shap_*.png` | SHAP |
| 6 | `06_run_experiments.py` | Main grid: {LSTM, GRU} × {price only, all, SHAP-selected} × horizons × seeds | `results/tables/main_results.csv` | Modeling + SHAP |
| 7 | `07_ablations.py` | One-factor ablations: hidden size, layers, lookback, dropout | `results/tables/ablations.csv`, `results/figures/ablation_*.png` | Modeling |
| 8 | `08_make_figures_tables.py` | Final comparison against the baselines, Diebold–Mariano tests, report figures | `results/tables/final_comparison.*`, `dm_tests.csv`, `results/figures/*` | Shared |

Quick version:

```bash
python 01_fetch_data.py
python 02_preprocess.py
python 03_baselines.py
python 04_train_rnn.py --model lstm --features all --horizon 1    # smoke test of one run
python 05_shap_analysis.py
python 06_run_experiments.py
python 07_ablations.py --features shap_selected --horizon 5
python 08_make_figures_tables.py
```

Scripts 04–07 skip runs that already have results, so you can stop and restart them.
Use `--force` to retrain.

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
  training rows only. SHAP selection uses train and validation data only and never sees the test period.
* **Same test origins for every model**, so Diebold–Mariano tests compare like with like.
* **Several seeds per configuration.** Results are reported as mean ± std, and the SHAP importances
  are averaged over the seeds.
* **SHAP on CPU.** cuDNN cannot backpropagate through an RNN in eval mode, so the explainer loads
  the checkpoints on CPU.

## Folder layout

```
Code/
├── config.yaml
├── requirements.txt
├── 01_fetch_data.py … 08_make_figures_tables.py
├── oilfc/                # shared package
├── data/
│   ├── raw/              # downloaded CSVs (01)
│   └── processed/        # levels.csv (02)
├── results/
│   ├── tables/           # CSV/Markdown tables; runs/ = one JSON per training run
│   ├── figures/          # PNGs for the report and slides
│   ├── predictions/      # test-period forecasts, one CSV per run
│   ├── models/           # .pt checkpoints (git-ignored)
│   └── shap/             # importances and selected features
└── notebooks/            # exploratory work only; final code goes in the scripts
```
