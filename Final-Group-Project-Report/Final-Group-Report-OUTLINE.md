# Final Group Project Report — OUTLINE

> **How to use this file:** this is the skeleton for the Word report, following the eight sections the
> guidelines require. Write the report in Word, export it as PDF and save it in this folder
> (e.g. `Final-Group-Project-Report/Group4-Final-Report.pdf`). Figures and tables come from
> `Code/results/`. **All results (figures and tables) go in the main body, not in appendices.**

**Title:** Does SHAP-Guided Feature Selection Improve LSTM/GRU Forecasts of Crude Oil Prices?
**Group 4:** [Name 1], [Name 2], [Name 3]

---

## 1. Introduction
- Motivation: why oil price forecasting matters, and why the random walk is hard to beat
- The research gap: Firouzjaee & Khalilian (adding correlated features did not help) vs.
  Okasha & Zaher 2026 (SHAP-selected features helped)
- Research question and hypotheses:
  - H1: LSTM/GRU with price-only input does not beat the random walk at h = 1
  - H2: adding all features does not reliably improve on price only (replicating paper 1)
  - H3: SHAP-selected features improve on both (testing paper 2's claim on crude oil)
- Contributions (bullets) and an outline of the rest of the report

## 2. Description of the Data Set
- Sources and series table (name, source, ID/ticker, unit, start date) → `Code/config.yaml`, `data/raw/download_log.json`
- Cleaning: calendar alignment, forward fill, the negative WTI price (20 April 2020) and how we handled it
- Sample period, number of observations, train/val/test split with dates
- **Figure:** `prices_overview.png` with split boundaries
- **Table:** `descriptive_stats.csv`; **Table/Figure:** `correlations.csv` of returns
- Transformations: log returns for prices, first differences for rates (stationarity)

## 3. Deep Learning Network and Training Algorithm
- RNN background, and why gating solves vanishing gradients
- LSTM equations (input/forget/output gates, cell state); GRU equations (update/reset gates)
- **Figure:** LSTM and GRU cell diagrams; our architecture (window → stacked RNN → linear head)
- Training: MSE loss, backpropagation through time, Adam, gradient clipping, LR schedule
- SHAP: Shapley values, additivity, expected gradients (`GradientExplainer`), how we aggregate
  |SHAP| over samples and lags, and the selection rule (top-k / cumulative share)
- Baselines: random walk, ARIMA(p,d,q)

## 4. Experimental Setup
- Supervised framing: lookback window, direct multi-horizon targets (h = 1, 5, 20), log-return target
  converted back to price
- Avoiding leakage: split by target date, scalers fit on train only, SHAP on train/val only
- Implementation: PyTorch, hardware (AWS GPU, instance type), run time
- Mini-batches: batch size and how it was chosen; learning rate and how it was chosen (validation)
- Overfitting: early stopping on validation loss, dropout, weight decay, small models; learning curves
- Extrapolation: the test period (2022–2025) contains regimes not seen in training (2022 price spike);
  discuss
- Metrics: RMSE, MAE, MAPE (and its problem near zero prices), directional accuracy
- Statistical testing: Diebold–Mariano with the HLN correction; 3 seeds, mean ± std
- Experiment grid table: models × feature sets × horizons × seeds; ablation grid

## 5. Results  *(most important section)*
- 5.1 Baselines: random walk vs. ARIMA → `baselines_metrics.csv`
- 5.2 Main comparison: price only vs. all vs. SHAP-selected, for LSTM and GRU, per horizon
  → `final_comparison.md`, `rmse_by_featureset.png`
- 5.3 Statistical significance → `dm_tests.csv`
- 5.4 SHAP findings: which drivers matter, how importance changes with horizon and lag
  → `shap_bar_h*.png`, `shap_lag_heatmap_h*.png`, `shap_beeswarm_h*.png`, `importance_h*.csv`
- 5.5 Ablations: architecture, depth/width, lookback, dropout → `ablation_*.png`
- 5.6 Forecast plots in the test period → `forecast_vs_actual_h*.png`; learning curves
- Explain **every** figure and table in the text; connect each result back to H1–H3 and the two papers

## 6. Summary and Conclusions
- Answer to the research question, in one paragraph
- What we learned (technical and about the problem)
- Limitations: one commodity, a single test period, daily frequency, SHAP on correlated inputs
- Future work: weekly frequency as in paper 2, attention/Transformer models, rolling re-training,
  news and sentiment features, SHAP with grouped or correlated features

## 7. References
- Papers (APA or IEEE style, used consistently)
- Data sources (FRED series pages, Yahoo Finance)
- **Links to any websites / GitHub repos we borrowed code from** (required by the guidelines)

## 8. Appendix — Code Listings
- Separate appendix with documented code: the scripts in `Code/` (or a link to the repo plus key listings)
