# Final Group Project Report — OUTLINE

> **How to use this file:** this is the skeleton for the Word report, following the eight sections the
> guidelines require. Write the report in Word, export it as PDF and save it in this folder
> (e.g. `Final-Group-Project-Report/Group4-Final-Report.pdf`). Figures and tables come from
> `Code/results/`. **All results (figures and tables) go in the main body, not in appendices.**
>
> **Results status:** the numbers below come from the full pipeline run (scripts 01–08) on AWS on
> 22 Nov 2026 with the settings in `Code/config.yaml`. If anything in the config changes, rerun and
> update them.

**Title:** Does SHAP-Guided Feature Selection Improve LSTM/GRU Forecasts of Crude Oil Prices?
**Group 4:** [Name 1], [Name 2], [Name 3]

---

## 0. Key findings at a glance (for the abstract and conclusion)

- **Nothing beats the random walk in a statistically significant way.** The best network at each
  horizon is the **price-only GRU**, and it is within 0.4% of the random walk's RMSE
  (h=1: 1.920 vs 1.925; h=5: 4.190 vs 4.217; h=20: 6.775 vs 6.801). No Diebold–Mariano (DM)
  test against the random walk is significant (all p > 0.35).
- **Adding all 13 features makes forecasts worse,** and significantly so at h=1 and h=20
  (DM vs random walk: p ≈ 0.015–0.019 at h=1 and p ≈ 0.005–0.017 at h=20).
- **SHAP selection partly repairs the damage but never beats price only.** It lowers RMSE compared
  with all features for both models at h=1 and for GRU at h=5 and h=20 (LSTM h=5 is a tie, and LSTM
  h=20 is worse), but it never beats price-only input. The
  SHAP-selected LSTM at h=20 is still significantly worse than the random walk (p = 0.046).
- **Hypotheses:** the results support H1 and H2. They **do not support H3**, so they agree with
  Firouzjaee & Khalilian rather than Okasha & Zaher (2026).
- **Directional accuracy is about 0.47–0.55,** which is essentially a coin flip.

---

## 1. Introduction
- Motivation: why oil price forecasting matters, and why the random walk is hard to beat
- The research gap: Firouzjaee & Khalilian (adding correlated features did not help) vs.
  Okasha & Zaher 2026 (SHAP-selected features helped)
- Research question and hypotheses:
  - H1: LSTM/GRU with price-only input does not beat the random walk at h = 1 → **supported**
  - H2: adding all features does not reliably improve on price only (replicating paper 1) → **supported**
    (it is significantly worse at h=1 and h=20)
  - H3: SHAP-selected features improve on both (testing paper 2's claim on crude oil) → **not supported**
    (better than all features, but not better than price only or the random walk)
- Contributions (bullets) and an outline of the rest of the report
  - A controlled, leakage-free comparison of three feature sets × two RNNs × three horizons × three seeds
  - SHAP computed on the RNN itself (expected gradients), not on a tree surrogate
  - Significance tests (DM with the HLN correction) against both the random walk and ARIMA

## 2. Description of the Data Set
- Sources and series table (name, source, ID/ticker, unit, start date) → `Code/config.yaml`, `data/raw/download_log.json`
  - 13 series: WTI and Brent (FRED); USD_BROAD, UST10Y, UST2Y, FEDFUNDS and BREAKEVEN10Y (FRED);
    DXY, GOLD, SP500, VIX, NATGAS and XLE (Yahoo Finance)
- Cleaning: calendar alignment, forward fill, the negative WTI price (20 April 2020) and how we handled it
  - WTI settled at −37.63 USD/bbl on 2020-04-20; the policy is **clip** at 1.0 USD/bbl (the WTI minimum in
    `descriptive_stats.csv` is 1.0 because of this)
  - Dropping rows with missing values removed 1,566 rows. The binding constraint is USD_BROAD, which
    starts in 2006.
- Sample period, number of observations, train/val/test split with dates
  - **5,217 daily observations, 2006-01-02 to 2025-12-31**
  - Train: 2006 to 2018-12-31 (3,359 windows) · Validation: 2019–2021 (784) · Test: 2022–2025 (1,043)
  - **Important:** the validation period contains COVID and the negative WTI price. Validation RMSE and
    MAPE are therefore much higher than on test (at h=1, val MAPE ≈ 4.5% vs test ≈ 1.7%), and the
    validation loss in the training logs is about 35× the training loss. Discuss how this affects early
    stopping, SHAP and the ablations (all of which use validation data).
- **Figure:** `prices_overview.png` with split boundaries
- **Table:** `descriptive_stats.csv` (e.g. WTI mean 72.3, sd 21.3, max 145.3); **Table/Figure:** `correlations.csv` of returns
- Transformations: log returns for prices, first differences for rates (stationarity)

## 3. Deep Learning Network and Training Algorithm
- RNN background, and why gating solves vanishing gradients
- LSTM equations (input/forget/output gates, cell state); GRU equations (update/reset gates)
- **Figure:** LSTM and GRU cell diagrams; our architecture (window → stacked RNN → linear head)
  - Default: lookback 30, hidden size 64, 2 layers, dropout 0.2 (51,521 parameters for LSTM, 38,657 for GRU)
- Training: MSE loss, backpropagation through time, Adam, gradient clipping, LR schedule
  - Adam with lr 1e-3, batch size 64, up to 200 epochs, early stopping with patience 20,
    gradient clipping at 1.0, no weight decay
- SHAP: Shapley values, additivity, expected gradients (`GradientExplainer`), how we aggregate
  |SHAP| over samples and lags, and the selection rule (top-k / cumulative share)
  - Background: 200 training samples; explained on validation data; averaged over 3 seeds;
    rule **top_k = 4** plus WTI, which is always included
- Baselines: random walk, ARIMA(p,d,q) (rolling origin, refit every 20 test steps)

## 4. Experimental Setup
- Supervised framing: lookback window, direct multi-horizon targets (h = 1, 5, 20), log-return target
  converted back to price
- Avoiding leakage: split by target date, scalers fit on train only, SHAP on train/val only
- Implementation: PyTorch, hardware (AWS GPU, instance type), run time
  - **AWS g5.xlarge (NVIDIA A10G, 24 GB)**, PyTorch 2.13 + CUDA, Python 3.13
  - Run time of the full pipeline: about 30 minutes (baselines ≈ 2 min, SHAP ≈ 13 min,
    main grid ≈ 5 min, ablations ≈ 6.5 min); a single training run takes about 15 s
- Mini-batches: batch size and how it was chosen; learning rate and how it was chosen (validation)
- Overfitting: early stopping on validation loss, dropout, weight decay, small models; learning curves
  (`learning_curves_h*.png`); training loss keeps falling while validation loss is flat from the first epochs
- Extrapolation: the test period (2022–2025) contains regimes not seen in training (2022 price spike);
  discuss
- Metrics: RMSE, MAE, MAPE (and its problem near zero prices), directional accuracy
  - Note: the random walk's directional accuracy is about 0 because it always predicts "no change".
    Explain this so readers don't take it for a bug.
- Statistical testing: Diebold–Mariano with the HLN correction; 3 seeds, mean ± std
  - DM tests use the **seed-averaged forecast** (`final_comparison.md`), while `main_results.csv`
    reports the mean ± std across seeds. State this, because the two RMSE numbers differ slightly.
  - Sign: DM < 0 means the RNN is more accurate than the benchmark; DM > 0 means it is less accurate.
- Experiment grid table: models × feature sets × horizons × seeds; ablation grid
  - Main grid: {LSTM, GRU} × {price only, all, SHAP-selected} × {1, 5, 20} × {0, 1, 2} = 54 runs
  - Ablations (SHAP-selected, h=5, 3 seeds): hidden size {32, 64, 128}, layers {1, 2, 3},
    lookback {10, 30, 60}, dropout {0.0, 0.2, 0.4}, each varied one at a time

## 5. Results  *(most important section)*

### 5.1 Baselines: random walk vs. ARIMA → `baselines_metrics.csv`

| h | Random walk RMSE | ARIMA RMSE |
|---|---|---|
| 1 | **1.925** | 2.121 |
| 5 | 4.217 | **4.174** |
| 20 | **6.801** | 6.802 |

- The random walk is the strongest baseline at h=1. ARIMA is slightly better at h=5, and the two
  tie at h=20.

### 5.2 Main comparison → `final_comparison.md`, `rmse_by_featureset.png`

Test RMSE of the seed-averaged forecasts:

| h | Model | Price only | All | SHAP-selected |
|---|---|---|---|---|
| 1 | LSTM | 1.925 | 1.977 | 1.931 |
| 1 | GRU | **1.920** | 1.981 | 1.932 |
| 5 | LSTM | 4.203 | 4.291 | 4.290 |
| 5 | GRU | **4.190** | 4.346 | 4.246 |
| 20 | LSTM | 6.844 | 7.748 | 8.060 |
| 20 | GRU | **6.775** | 9.124 | 7.699 |

- Price only is the best feature set in all 6 model × horizon cells.
- The cost of adding features grows with the horizon: about +3% RMSE at h=1, +2–4% at h=5 and
  +13–35% at h=20.
- SHAP-selected input is better than all features in 4 of 6 cells (the exceptions are LSTM h=20 and
  LSTM h=5, which is about a tie).
- GRU is slightly better than or equal to LSTM with price-only input at every horizon.
- Variance across seeds grows with more features and longer horizons (`main_results.csv`, e.g. GRU all
  h=20 test RMSE std 0.63 vs 0.06 for price only), so extra features also make training less stable.

### 5.3 Statistical significance → `dm_tests.csv`

| Comparison | h=1 | h=5 | h=20 |
|---|---|---|---|
| Price only vs random walk | n.s. (p 0.61–0.95) | n.s. (p 0.36–0.47) | n.s. (p 0.76–0.81) |
| All vs random walk | **worse** (p 0.015–0.019) | n.s. | **worse** (p 0.005–0.017) |
| SHAP-selected vs random walk | n.s. | n.s. | LSTM **worse** (p 0.046); GRU n.s. (p 0.074) |
| Every RNN vs ARIMA at h=1 | **better** (DM −3.0 to −4.0, p < 0.003) | — | — |

- At h=1 every network significantly beats ARIMA, but none beats the random walk. ARIMA is the
  weaker baseline at h=1.

### 5.4 SHAP findings → `shap_bar_h*.png`, `shap_lag_heatmap_h*.png`, `shap_beeswarm_h*.png`, `importance_h*.csv`

| Rank | h=1 | h=5 | h=20 |
|---|---|---|---|
| 1 (always kept) | WTI (20%) | WTI (20%) | WTI (24%) |
| 2 | Brent (14%) | Brent (16%) | S&P 500 (13%) |
| 3 | XLE (9%) | S&P 500 (15%) | XLE (12%) |
| 4 | DXY (9%) | UST10Y (8%) | Brent (9%) |
| 5 | VIX (8%) | VIX (8%) | DXY (8%) |
| Last | FEDFUNDS (1%) | FEDFUNDS (1%) | USD_BROAD (2%) |

- WTI's own history dominates at every horizon. Brent is near the top at every horizon. Equity and
  energy-equity factors (S&P 500, XLE) gain weight at longer horizons. The policy rate (FEDFUNDS) is
  close to irrelevant throughout.
- **Caveat:** importance is unstable across seeds. At h=5, seed 0's |SHAP| values are about 5–10× larger
  than seeds 1–2, and at h=20 seed 0 is smaller. The ranking comes from the mean, so discuss it as
  indicative. A second caveat is that SHAP spreads credit between correlated inputs (WTI/Brent,
  DXY/USD_BROAD).
- High SHAP importance did not translate into better forecasts. This is the central point of the
  discussion: importance on the validation period ≠ predictive value on the test period.

### 5.5 Ablations → `ablations.csv`, `ablation_*.png` (SHAP-selected, h=5)

| Factor | Values tried | Test RMSE range (LSTM / GRU) | Takeaway |
|---|---|---|---|
| Hidden size | 32 / 64 / 128 | 4.28–4.30 / 4.25–4.28 | Smallest (32) is as good or best |
| Layers | 1 / 2 / 3 | 4.30–4.36 / 4.24–4.27 | No consistent gain from depth |
| Lookback | 10 / 30 / 60 | 4.30–4.37 / 4.25–4.31 | 30 is best for LSTM; 10 is worst for both |
| Dropout | 0.0 / 0.2 / 0.4 | 4.30–4.43 / 4.25–4.28 | Small effect, but LSTM gets worse at 0.4 |

- All differences are within about 3% and smaller than the gap to the random walk (4.217), so
  architecture choices are not the bottleneck; the signal is. GRU is better than LSTM in every
  ablation setting with fewer parameters.

### 5.6 Forecast plots → `forecast_vs_actual_h*.png`; learning curves → `learning_curves_h*.png`
- Expect the forecasts to track the lagged actual price (random-walk-like behaviour). Point this out
  in the plots, especially around the 2022 spike.

- Explain **every** figure and table in the text; connect each result back to H1–H3 and the two papers

## 6. Summary and Conclusions
- Answer to the research question, in one paragraph
  - No. On daily WTI over 2022–2025, SHAP-guided selection improves on throwing in all features,
    but neither LSTM nor GRU beats a random walk, and price-only input is always best.
    This is consistent with the efficient-market view and with Firouzjaee & Khalilian.
- What we learned (technical and about the problem)
  - Extra correlated inputs mainly add variance (unstable seeds, worse long-horizon RMSE)
  - A strong naive benchmark and DM tests are essential; ARIMA alone would have made the RNNs look good
  - A validation window containing a crash (2020) distorts early stopping and SHAP
- Limitations: one commodity, a single test period, daily frequency, SHAP on correlated inputs,
  a validation period that includes COVID and the negative price, SHAP rankings that change with the seed
- Future work: weekly frequency as in paper 2, attention/Transformer models, rolling re-training,
  news and sentiment features, SHAP with grouped or correlated features, a validation window
  without the 2020 crash (or rolling validation), the LightGBM TreeExplainer route used by Okasha & Zaher

## 7. References
- Papers (APA or IEEE style, used consistently)
- Data sources (FRED series pages, Yahoo Finance)
- **Links to any websites / GitHub repos we borrowed code from** (required by the guidelines)

## 8. Appendix — Code Listings
- Separate appendix with documented code: the scripts in `Code/` (or a link to the repo plus key listings)
