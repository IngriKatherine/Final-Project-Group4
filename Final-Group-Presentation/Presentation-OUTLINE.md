# Final Group Presentation — OUTLINE (12–15 minutes)

> **How to use this file:** build the slides in PowerPoint, export them as PDF and save the PDF in this
> folder (e.g. `Final-Group-Presentation/Group4-Presentation.pdf`). Aim for about 1 minute per slide.
> Each member presents the part they owned.

| # | Slide | Content | Visual | Time | Speaker |
|---|-------|---------|--------|------|---------|
| 1 | Title | Project title, Group 4, names | — | 0:20 | [TODO] |
| 2 | Why oil prices? | Why it matters; the random walk is hard to beat | WTI/Brent 2006–2025 chart | 0:45 | [TODO] |
| 3 | Two papers, two answers | Paper 1: extra features did not help. Paper 2: SHAP selection helped. Our research question | Two-column comparison | 1:00 | [TODO] |
| 4 | Data | Sources, series, sample, split | `prices_overview.png` + table | 1:00 | Data owner |
| 5 | Preprocessing | Log returns, the negative WTI price, leakage rules | Simple pipeline diagram | 0:45 | Data owner |
| 6 | Baselines | Random walk, ARIMA, and their test errors | Small table | 0:45 | Data owner |
| 7 | LSTM and GRU | Gates, equations, our architecture | Cell diagrams | 1:15 | Modeling owner |
| 8 | Training setup | Windows, horizons, loss, Adam, early stopping, seeds, AWS GPU | Learning curves | 1:00 | Modeling owner |
| 9 | SHAP in one slide | Shapley values, GradientExplainer, aggregation over lags | Formula + diagram | 1:00 | SHAP owner |
| 10 | What drives oil forecasts? | Feature ranking per horizon, selected features | `shap_bar_h*.png`, heatmap | 1:15 | SHAP owner |
| 11 | Main result | Price only vs. all vs. SHAP-selected, LSTM vs. GRU, per horizon | `rmse_by_featureset.png` | 1:30 | [TODO] |
| 12 | Is it significant? | Diebold–Mariano vs. random walk / ARIMA | Table | 0:45 | [TODO] |
| 13 | Ablations | Depth, width, lookback, dropout | `ablation_*.png` | 1:00 | Modeling owner |
| 14 | Forecasts in the test period | Actual vs. predicted, 2022–2025 | `forecast_vs_actual_h*.png` | 0:45 | [TODO] |
| 15 | Conclusions and future work | Answer to the research question, limitations, next steps | Bullets | 1:00 | [TODO] |
| 16 | Questions | Repo link | QR code | — | All |

**Total: about 14 minutes.** Rehearse with a timer at least once.
