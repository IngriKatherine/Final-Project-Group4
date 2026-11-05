# Final-Project-Group4 — Forecasting Crude Oil Prices with LSTM/GRU and SHAP Feature Selection

Deep Learning and Fundamentals of AI · Final Project · Group 4
**Members:** [Name 1] · [Name 2] · [Name 3]

We forecast crude oil spot prices (WTI, with Brent as a robustness check) with LSTM and GRU networks
in PyTorch, and use **SHAP** to understand which inputs the networks rely on and to **select** them.

---

## Research question

> **Does SHAP-guided feature selection improve recurrent-network forecasts of crude oil prices in cases
> where simply adding more features does not?**

Two papers point in different directions:

* **Firouzjaee & Khalilian** added correlated assets (crude oil, gold, the US dollar) to an LSTM that
  forecasts oil-company stocks. The extra inputs mostly **did not** improve the predictions.
* **Okasha & Zaher (2026, *Journal of Shipping and Trade*)** used SHAP to rank and keep only the most
  important drivers. Their LSTM forecast marine gas oil prices 8 weeks ahead and **beat** ARIMA, ARIMAX
  and an LSTM without feature selection.

We test both ideas on the same problem by comparing three input sets for each network:

| Input set | Description |
|-----------|-------------|
| `price_only` | Only the oil price's own history |
| `all` | Oil price plus every candidate driver |
| `shap_selected` | Oil price plus the top drivers ranked by SHAP (chosen on train/validation data only) |

## Approach at a glance

| Item | Choice |
|------|--------|
| Target | WTI spot price (`DCOILWTICO`); Brent (`DCOILBRENTEU`) as a robustness check |
| Candidate drivers | US dollar (DXY, broad index), gold, S&P 500, VIX, natural gas, energy ETF (XLE), 2y/10y Treasury yields, fed funds rate, 10y breakeven inflation |
| Sources | [FRED](https://fred.stlouisfed.org/) and Yahoo Finance (`yfinance`), all free |
| Frequency and sample | Business-daily, about 2006–2025 (limited by the latest-starting series) |
| Split (chronological) | Train ≤ 2018 · Validation 2019–2021 · Test 2022–2025 |
| Horizons | 1, 5 and 20 business days ahead (one model per horizon) |
| Networks | Stacked LSTM and GRU, PyTorch, trained on the AWS GPU machine |
| Benchmarks | Naive random walk, rolling-origin ARIMA |
| Explainability | SHAP `GradientExplainer` on the trained LSTM (LightGBM + `TreeExplainer` as an alternative, as in the 2026 paper) |
| Ablations | LSTM vs. GRU, depth, width, dropout, lookback window, forecast horizon |
| Metrics | RMSE, MAE, MAPE (on prices, USD/bbl), directional accuracy; Diebold–Mariano tests; mean ± std over 3 seeds |

All settings live in [`Code/config.yaml`](Code/config.yaml).

## Repository structure

```
Final-Project-Group4/
├── README.md                          ← this file
├── .gitignore
├── Group-Proposal/
│   ├── Group-Proposal-DRAFT.md        ← draft text for the proposal
│   └── <Group4-Proposal>.pdf          ← final proposal (Word → PDF)
├── Final-Group-Project-Report/
│   ├── Final-Group-Report-OUTLINE.md  ← section-by-section outline
│   └── <Group4-Final-Report>.pdf      ← final report (Word → PDF)
├── Final-Group-Presentation/
│   ├── Presentation-OUTLINE.md        ← slide plan for 12–15 minutes
│   └── <Group4-Presentation>.pdf      ← slides (PowerPoint → PDF)
├── Code/                              ← all group code (see Code/README.md)
│   ├── README.md                      ← run order and description of every script
│   ├── config.yaml                    ← every setting in one place
│   ├── requirements.txt
│   ├── 01_fetch_data.py               ← download FRED + Yahoo data
│   ├── 02_preprocess.py               ← clean, align, descriptive stats
│   ├── 03_baselines.py                ← random walk + ARIMA
│   ├── 04_train_rnn.py                ← train one LSTM/GRU run
│   ├── 05_shap_analysis.py            ← SHAP importance + feature selection
│   ├── 06_run_experiments.py          ← main grid: models × feature sets × horizons × seeds
│   ├── 07_ablations.py                ← architecture / lookback / dropout ablations
│   ├── 08_make_figures_tables.py      ← final tables, DM tests, figures
│   ├── oilfc/                         ← shared package (data, models, training, metrics)
│   ├── data/{raw,processed}/
│   ├── results/{tables,figures,predictions,models,shap}/
│   └── notebooks/                     ← exploration only
└── firstname-lastname-individual-project/   ← TEMPLATE: each member copies and renames it
    ├── README.md
    ├── Individual-Final-Project-Report/
    │   └── Individual-Report-OUTLINE.md     → firstname-lastname-final-project.pdf
    └── Code/
        └── README.md
```

## How to run

```bash
git clone https://github.com/<owner>/Final-Project-Group4.git
cd Final-Project-Group4/Code
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build; omit on CPU
pip install -r requirements.txt

python 01_fetch_data.py            # 1. download data
python 02_preprocess.py            # 2. clean and align
python 03_baselines.py             # 3. random walk + ARIMA
python 05_shap_analysis.py         # 4. trains all-features LSTMs (via 04) and runs SHAP selection
python 06_run_experiments.py       # 5. full grid (resumable)
python 07_ablations.py             # 6. ablations
python 08_make_figures_tables.py   # 7. tables and figures for the report
```

`04_train_rnn.py` can also be run by itself for a single experiment, e.g.
`python 04_train_rnn.py --model gru --features shap_selected --horizon 5 --seed 0`.
Run every script from inside `Code/`. Scripts 04–07 skip runs that already have results, so you can
stop and restart them; use `--force` to retrain.

### Scripts in run order

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

### Shared package `Code/oilfc/`

| Module | Contents |
|--------|----------|
| `utils.py` | config loading and overrides, seeding, device selection, paths, run names |
| `data.py` | transforms (log returns / differences), targets, feature-set resolution, chronological split, train-only scaling, sliding windows |
| `models.py` | `RNNForecaster`: stacked LSTM or GRU with a linear head on the last hidden state |
| `train.py` | Adam training loop with early stopping, LR schedule, gradient clipping; `predict` |
| `metrics.py` | RMSE, MAE, MAPE, directional accuracy, Diebold–Mariano test with the HLN correction |

### Design decisions (and why)

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

**On AWS:** run long scripts inside `tmux` so they survive a dropped SSH connection. Finished runs are
skipped on restart. Copy `Code/results/` back before stopping the instance.

## Team and division of work

| Part | Owner | Scripts | Individual-report focus |
|------|-------|---------|-------------------------|
| Data pipeline and baselines | [TODO] | `01`, `02`, `03`, `oilfc/data.py`, `oilfc/metrics.py` | Data, stationarity, ARIMA, evaluation design, DM tests |
| LSTM/GRU modeling | [TODO] | `04`, `06`, `07`, `oilfc/models.py`, `oilfc/train.py` | LSTM/GRU theory, training, hyper-parameters, ablations |
| SHAP analysis | [TODO] | `05`, SHAP parts of `08` | Shapley values, explainers, feature selection, stability |

## Deliverables checklist

- [ ] `Group-Proposal/` — proposal PDF
- [ ] `Final-Group-Project-Report/` — group report PDF (sections 1–8 of the guidelines)
- [ ] `Final-Group-Presentation/` — slides PDF (12–15 min)
- [ ] `Code/` — all code + README with run order
- [ ] One `firstname-lastname-individual-project/` folder per member, each with
      `Individual-Final-Project-Report/firstname-lastname-final-project.pdf` and `Code/`
- [ ] Template folder deleted; `[TODO]`s and placeholder names replaced
- [ ] Repo link uploaded to Blackboard (and checked that it opens)

## References

1. Firouzjaee, J. T., & Khalilian, P. (2022). *The Interpretability of LSTM Models for Predicting Oil
   Company Stocks: Impact of Correlated Features.* arXiv:2201.00350. https://arxiv.org/html/2201.00350v5
2. Okasha, S. H., & Zaher, H. (2026). *An explainable deep learning framework for marine gas oil price
   forecasting: integrating LSTM with SHAP-based feature importance.* Journal of Shipping and Trade, 11, 15.
   https://doi.org/10.1186/s41072-026-00237-3
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
4. Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical
   machine translation. *EMNLP*.
5. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
6. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
   Economic Statistics*, 13(3).
7. Data: Federal Reserve Bank of St. Louis, FRED; Yahoo Finance via `yfinance`.

*Code borrowed from websites or GitHub repos must be linked here and in the reports.*
