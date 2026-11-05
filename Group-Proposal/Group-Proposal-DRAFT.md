# Group Proposal — Group 4 (DRAFT)

> **How to use this file:** edit the draft below, paste it into Word, export as PDF and save it in this
> folder (e.g. `Group-Proposal/Group4-Proposal.pdf`). The guidelines ask for "a few hundred words"
> covering the seven questions below. Delete this note and the `[TODO]` markers before exporting.

**Title:** Does SHAP-Guided Feature Selection Improve LSTM/GRU Forecasts of Crude Oil Prices?
**Members:** [TODO: Name 1], [TODO: Name 2], [TODO: Name 3]
**Course:** Deep Learning and Fundamentals of AI — Prof. Amir Jafari

---

## 1. What problem did you select and why?

We will forecast crude oil spot prices (WTI, with Brent as a robustness check) 1, 5 and 20 trading days
ahead using recurrent neural networks. We will also use SHAP to explain which inputs the networks rely on
and to select those inputs. Oil prices matter for inflation, energy budgets and markets, and they are hard
to forecast: the random walk is a strong benchmark.

The literature points in two directions. Firouzjaee & Khalilian found that adding correlated assets
(crude oil, gold, the US dollar) to an LSTM mostly did **not** improve forecasts of oil-company stocks.
Okasha & Zaher (2026) instead used SHAP-based feature selection with an LSTM to forecast marine gas oil
prices eight weeks ahead and beat ARIMA, ARIMAX and an LSTM without feature selection. Our project tests
whether SHAP-guided selection helps where simply adding more features does not.

## 2. What dataset will you use? Is it large enough?

Free daily data from FRED (WTI, Brent, broad dollar index, 2- and 10-year Treasury yields, fed funds rate,
breakeven inflation) and Yahoo Finance (dollar index DXY, gold, S&P 500, VIX, natural gas, energy-sector
ETF). The common sample from about 2006 to 2025 gives roughly 5,000 daily observations per series. With a
30-day sliding window, that is about 3,000 training windows. The size is moderate, so we will keep the
networks small (tens of thousands of parameters) and regularize them with dropout and early stopping.
Chronological split: train ≤ 2018, validation 2019–2021, test 2022–2025.

## 3. What deep network will you use? Standard or customized?

Standard stacked LSTM and GRU networks (1–3 layers, 32–128 hidden units) with a linear output head,
trained to predict the h-step log return. We will compare three input sets: price only, all candidate
features, and SHAP-selected features. Ablations cover architecture (LSTM vs. GRU, depth, width, dropout),
lookback window and forecast horizon. Benchmarks: naive random walk and ARIMA.

## 4. What framework will you use and why?

PyTorch: it is used in the course, it gives full control over the training loop, and SHAP's
`GradientExplainer` supports PyTorch models directly. Supporting libraries: statsmodels (ARIMA), shap,
scikit-learn and pandas. Training will run on our AWS GPU instance.

## 5. What reference materials will you use?

1. Firouzjaee, J. T. & Khalilian, P. *The Interpretability of LSTM Models for Predicting Oil Company Stocks:
   Impact of Correlated Features.* arXiv:2201.00350.
2. Okasha, S. H. & Zaher, H. (2026). *An explainable deep learning framework for marine gas oil price
   forecasting: integrating LSTM with SHAP-based feature importance.* Journal of Shipping and Trade, 11, 15.
3. Hochreiter & Schmidhuber (1997), LSTM; Cho et al. (2014), GRU.
4. Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions* (SHAP).
5. Diebold & Mariano (1995), *Comparing Predictive Accuracy*.
6. [TODO: course lecture notes / additional oil-forecasting papers]

## 6. How will you judge performance? What metrics?

RMSE, MAE and MAPE on the held-out test period, computed on prices in USD/bbl. We will also report
directional accuracy. Diebold–Mariano tests will check whether differences against the random walk and
ARIMA are statistically significant. Each configuration is trained with 3 random seeds and reported as
mean ± standard deviation.

## 7. Rough schedule

> [TODO: align the dates with the syllabus deadlines]

| Week | Dates | Milestone | Lead |
|------|-------|-----------|------|
| 1 | [TODO] | Data download, cleaning, EDA; random walk and ARIMA baselines | Data & baselines |
| 2 | [TODO] | LSTM/GRU training pipeline, first price-only and all-features results | Modeling |
| 3 | [TODO] | SHAP analysis and feature selection; full experiment grid on AWS | SHAP + Modeling |
| 4 | [TODO] | Ablations (architecture, lookback, horizon); significance tests; figures | All |
| 5 | [TODO] | Group report, slides, individual reports; clean up the GitHub repo | All |
