# Final-Project-Group4 — Forecasting Crude Oil Prices: from Price History to Ships Seen from Space

Deep Learning and Fundamentals of AI · Final Project · Group 4
**Members:** Ingri Katherine Quevedo Rocha · Aditi Shukla · Aidan Carslisle · Constantine Zhang

**Goal: oil price forecasting.** We forecast the WTI crude oil spot price (Brent as a robustness
check) 1, 5 and 20 business days ahead, and test whether information beyond the price history makes
the forecasts better: market drivers, images of the price chart, and oil supply measured from
satellite images of tanker traffic in the Strait of Hormuz, including through the 2026 Hormuz shock.

---

## Four ways to forecast one price

| Family | Models | Reads | Predicts | Where |
|--------|--------|-------|----------|-------|
| **Statistical models** | Random walk, ARIMA, ARIMAX | Price history (ARIMAX adds lagged market drivers) | Price *h* days ahead | `Code/03_baselines.py` |
| **Recurrent networks** | LSTM, GRU (PyTorch) | 30-day windows of daily returns: price only, or price + market drivers | Log return *h* days ahead | `Code/04_train_rnn.py`, `06`, `07` |
| **Image CNN** | CNN on price-chart images (following Ren et al., 2024) | The recent price path drawn as a black-and-white picture | Direction of the next move (up / down) | in progress; writes `results/predictions/imgcnn_*.csv` |
| **Satellite CNN** | Faster R-CNN ship detector (pretrained), then LSTM / GRU | Sentinel-2 images of the Strait of Hormuz; ship capacity calibrated into barrels/day with IMF PortWatch transits | An oil-supply series, then the 2026 oil price | `Code/satellite/` |

### Hypotheses

| # | Hypothesis | Tested on |
|---|------------|-----------|
| H1 | At a 1-day horizon no model beats the random walk (oil prices are close to a random walk) | Test period 2022–2025, Diebold–Mariano tests |
| H2 | ARIMAX (with market drivers) is no better than ARIMA | Test period 2022–2025 |
| H3 | The image CNN predicts the direction of the next move better than a coin flip | Binomial test on non-overlapping test days |
| H4 | Adding satellite oil-supply measures lowers 2026 forecast error compared with price only (and with AIS-based supply) | 2026, held out, Diebold–Mariano tests |

## Approach at a glance

| Item | Choice |
|------|--------|
| Target | WTI spot price (`DCOILWTICO`); Brent (`DCOILBRENTEU`) as a robustness check |
| Market drivers | US dollar (DXY, broad index), gold, S&P 500, VIX, natural gas, energy ETF (XLE), 2y/10y Treasury yields, fed funds rate, 10y breakeven inflation |
| Market data sources | [FRED](https://fred.stlouisfed.org/) and Yahoo Finance (`yfinance`), all free |
| Satellite data | Sentinel-2 L2A, 10 m ([Earth Search on AWS](https://earth-search.aws.element84.com/v1)); ship labels from the [Ai2 Sentinel-2 vessel dataset](https://github.com/allenai/rslearn_projects/blob/master/docs/sentinel2_vessels.md) plus our hand-checked Hormuz chips; [IMF PortWatch](https://portwatch.imf.org/) daily chokepoint transits as calibration labels |
| Frequency and sample | Business-daily, 2006–2025 for the main study; satellite series 2023–2026 |
| Split (chronological) | Main study: train ≤ 2018 · validation 2019–2021 · test 2022–2025. 2026 study: train 2023-01 → 2025-06 · validation 2025-07 → 2025-12 · test 2026 |
| Horizons | 1, 5 and 20 business days ahead (one model per horizon) |
| Ablations | LSTM vs. GRU, depth, width, dropout, lookback window; detector pretraining (satellite) |
| Metrics | RMSE, MAE, MAPE (on prices, USD/bbl), directional accuracy, skill vs. random walk; Diebold–Mariano tests; mean ± std over seeds |

All settings live in [`Code/config.yaml`](Code/config.yaml) and
[`Code/satellite/config_satellite.yaml`](Code/satellite/config_satellite.yaml).

## Repository structure

```
Final-Project-Group4/
├── README.md                          ← this file
├── .gitignore
├── docs/
├── Group-Proposal/
├── Final-Group-Project-Report/        ← group report (Word → PDF)
├── Final-Group-Presentation/          ← slides PDF; the live version is Code/presentation
├── Code/                              ← all group code (see Code/README.md)
│   ├── README.md                      ← run order and description of every script
│   ├── config.yaml                    ← every setting in one place
│   ├── requirements.txt
│   ├── 01_fetch_data.py               ← download FRED + Yahoo data
│   ├── 02_preprocess.py               ← clean, align, descriptive stats
│   ├── 03_baselines.py                ← random walk, ARIMA, ARIMAX
│   ├── 04_train_rnn.py                ← train one LSTM/GRU run
│   ├── 05_shap_analysis.py            ← ranks drivers for the reduced input set
│   ├── 06_run_experiments.py          ← main grid: models × input sets × horizons × seeds
│   ├── 07_ablations.py                ← architecture / lookback / dropout ablations
│   ├── 08_make_figures_tables.py      ← final tables, DM tests, figures
│   ├── sample_start_check.py          ← does starting the sample in 2003 change accuracy?
│   ├── oilfc/                         ← shared package (data, models, training, metrics)
│   ├── satellite/                     ← Strait of Hormuz track (see satellite/README.md)
│   │   ├── s00_check_setup.py … s07_calibrate_and_export.py
│   │   ├── hormuz/                    ← shared package (raster, detector, measurement)
│   │   └── tests/
│   ├── presentation/                  ← interactive Streamlit presentation
│   ├── data/{raw,processed}/
│   ├── results/{tables,figures,predictions,predictions_2026,models}/
│   └── notebooks/                     ← exploration only
└── firstname-lastname-individual-project/   ← one folder per member
```

## How to run

```bash
git clone https://github.com/<owner>/Final-Project-Group4.git
cd Final-Project-Group4/Code
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build; omit on CPU
pip install -r requirements.txt

# Price models (2006–2025)
python 01_fetch_data.py            # 1. download data
python 02_preprocess.py            # 2. clean and align
python 03_baselines.py             # 3. random walk, ARIMA, ARIMAX
python 05_shap_analysis.py         # 4. reduced input set (needed while shap_selected is in feature_sets)
python 06_run_experiments.py       # 5. LSTM / GRU grid (resumable)
python 07_ablations.py             # 6. ablations
python 08_make_figures_tables.py   # 7. tables and figures for the report

# Satellite track (Strait of Hormuz, 2023–2026) — see Code/satellite/README.md
cd satellite
pip install -r requirements_satellite.txt
python s00_check_setup.py          # checks packages, data access and PortWatch before the long run
python s01_search_scenes.py && python s02_download_scenes.py
python s03_prepare_labels.py ai2 && python s04_train_detector.py --sources ai2 --tag ai2_coco
python s05_detect_scenes.py --ckpt runs/hormuz_ft/best.pt --quicklook
python s06_build_supply_series.py
python s07_calibrate_and_export.py # writes Code/data/processed/satellite_features.csv
cd ..

# Presentation
pip install -r presentation/requirements-presentation.txt
streamlit run presentation/app.py
```

Run every script from inside `Code/` (satellite scripts from inside `Code/satellite/`). Scripts 04–07
skip runs that already have results, so you can stop and restart them; use `--force` to retrain.
For the 2026 forecasts, set `data.end_date` in `Code/config.yaml` to a 2026 date and rerun 01 and 02
so `levels.csv` includes 2026 prices.

### Results files every model writes

Every forecast file has the columns `origin_date, target_date, base_price, true_price, pred_price`,
so all models are evaluated, compared and plotted the same way:

| Model | File |
|-------|------|
| Random walk, ARIMA, ARIMAX | `results/predictions/baseline_(rw\|arima\|arimax)_h<h>.csv` |
| LSTM, GRU | `results/predictions/(lstm\|gru)_<inputs>_h<h>_s<seed>.csv` |
| Image CNN | `results/predictions/imgcnn_price_only_h<h>_s<seed>.csv` (optional `prob_up` column) |
| 2026 forecasts | `results/predictions_2026/<model>_<inputs>_h<h>_s<seed>.csv`, inputs `price_only`, `ais`, `satellite`, `satellite_ais` |

### Design decisions (and why)

* **Target = h-step log return, metrics on prices.** A network trained on price levels mostly learns
  "tomorrow ≈ today". Predicting the return and converting back with `P_t · exp(ŷ)` makes the
  comparison with the random walk honest. Every metric is computed in USD/bbl.
* **Direct multi-horizon strategy.** One model per horizon (1, 5, 20 business days by default).
* **No look-ahead.** Samples are assigned to a split by their *target* date, and scalers are fit on
  training rows only. The satellite supply features are causal: on day *t* only images taken on or
  before *t* are used, and the PortWatch calibration uses 2023–2025 only, so 2026 stays unseen.
* **Same test origins for every model**, so Diebold–Mariano tests compare like with like.
* **Several seeds per configuration.** Results are reported as mean ± std.
* **Satellite counts vs. AIS.** In 2026 many tankers crossed Hormuz with AIS switched off. Comparing
  price + satellite supply against price + AIS supply shows whether the images carry information
  that ship tracking misses.

**On AWS:** run long scripts inside `tmux` so they survive a dropped SSH connection. Finished runs are
skipped on restart. Copy `Code/results/` and `Code/satellite/runs/` back before stopping the instance.

## Team and division of work

| Part | Owner | Code |
|------|-------|------|
| Data pipeline, ARIMA and ARIMAX | [TODO] | `01`, `02`, `03`, `sample_start_check.py`, `oilfc/data.py`, `oilfc/metrics.py` |
| LSTM / GRU forecasting (improving the LSTM) | [TODO] | `04`, `06`, `07`, `oilfc/models.py`, `oilfc/train.py` |
| Image CNN on price charts | [TODO] | image CNN script, `results/predictions/imgcnn_*.csv` |
| Satellite CNN and 2026 forecast | Ingri Katherine Quevedo Rocha, with [TODO] | `satellite/`, `results/predictions_2026/` |
| Presentation | Shared | `presentation/` |

## Deliverables checklist

- [ ] `Group-Proposal/` — proposal PDF
- [ ] `Final-Group-Project-Report/` — group report PDF (sections 1–8 of the guidelines)
- [ ] `Final-Group-Presentation/` — slides PDF (12–15 min)
- [ ] `Code/` — all code + README with run order
- [ ] One `firstname-lastname-individual-project/` folder per member, each with
      `Individual-Final-Project-Report/firstname-lastname-final-project.pdf` and `Code/`
- [ ] Template folder deleted; `[TODO]`s replaced
- [ ] Repo link uploaded to Blackboard (and checked that it opens)

## References

1. Ren, X., Jiang, W., Ji, Q., & Zhai, P. (2024). Seeing is believing: Forecasting crude oil price
   trend from the perspective of images. *Journal of Forecasting*, 43(7), 2809–2821.
   https://doi.org/10.1002/for.3149
2. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis:
   Forecasting and Control* (5th ed.). Wiley.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8).
4. Cho, K., et al. (2014). Learning phrase representations using RNN encoder–decoder for statistical
   machine translation. *EMNLP*.
5. Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection
   with region proposal networks. *NeurIPS*.
6. Beukema, P., et al. (2023). Satellite imagery and AI: A new era in ocean conservation, from
   research to deployment and impact. arXiv:2312.03207. Ai2 Sentinel-2 vessel dataset:
   https://github.com/allenai/rslearn_projects
7. IMF PortWatch (2026). Daily chokepoint transit calls and trade volume estimates. https://portwatch.imf.org
8. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business &
   Economic Statistics*, 13(3).
9. Data: Federal Reserve Bank of St. Louis, FRED; Yahoo Finance via `yfinance`; Copernicus
   Sentinel-2 (ESA), via Element 84 Earth Search on AWS Open Data.

*Code borrowed from websites or GitHub repos must be linked here and in the reports.*
