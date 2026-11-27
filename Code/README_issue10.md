# Issue #10 – Improving the LSTM with a pre-trained model (Chronos-2)

Scripts `15`–`21`, `cra_config.yaml`, `cra_followup.yaml`, `oilfc/cra_*.py`, `tests/test_cra*.py`.
Outputs live under `data/cra/` and `results/cra/`. The original pipeline (`01`–`08`, `config.yaml`) is untouched.

**Bottom line:** neither way of adding Chronos-2 beat the original LSTM or the random walk (RW).
No model significantly beats the RW in RMSE at any horizon in either period. For oil price
*levels* at 1–20 trading days, the RW remains very hard to beat.

## Methodology

**Data** (`15`): daily WTI spot price as the calendar, 2000 to Nov 2026. Inputs are Brent, heating oil (crack spread), OVX, VIX,
DXY, XLE, 10y Treasury, 10y breakeven and the daily geopolitical risk index (GPR, Caldara & Iacoviello).
There are 15 features (returns, momentum, realised volatility, spreads). Each row only uses information known at that day's close.

**Target:** the h-step log return, h ∈ {1, 5, 20}, scaled by the EWMA (λ = 0.94) ex-ante volatility.

**Protocol** (identical for every model):
- Expanding-window refits, one per evaluation year, each with a one-year validation set and early stopping.
- 5 seeds per configuration; forecasts are averaged over the seeds.
- Every model sees the same samples, scalers, loss mask and forecast origins (997 test origins per horizon).
- **Test** = 2022–2025. **Holdout** = 2026, which starts after Chronos-2's Oct 2025 release. That matters because Chronos-2's
  pre-training data may include 2022–2025, so test-period Chronos results could be optimistic.

**Models compared**

| Model | What it is |
|---|---|
| Random walk | forecast = last price |
| Original LSTM | the team's LSTM from `config.yaml`, refit under this protocol |
| Chronos-2 (zero-shot) | `amazon/chronos-2`, context 512, past-only covariates Brent / heating oil / OVX; q10 / q50 / q90 cached per origin (`16`) |
| **Design 1 – Gated residual LSTM** (`17`, `oilfc/cra_model.py`) | Starts from the Chronos median and learns a gated correction `q50 = chronos + g·δ`. Also has a variable-selection network, attention pooling and non-crossing quantile heads. The gate starts almost closed (sigmoid(−3) ≈ 0.05). Loss = pinball + Huber + a gate penalty |
| **Design 2 – Chronos-feature LSTM** (`20`, `oilfc/cra_followup_model.py`) | Chronos quantiles go in as input features through a learned gate, and the head starts at the RW. During training the whole Chronos block is dropped with p = 0.2, so the same network can also be scored with Chronos masked. Uses point-in-time GPR vintages (`19`, `oilfc/cra_gpr.py`) |

**Evaluation** (`18`, `21`):
- Point accuracy: RMSE, MAE, directional accuracy, out-of-sample R² vs the RW.
- Interval quality: 80% interval coverage.
- Forecast tests: Diebold–Mariano (HLN, Bartlett), with Holm correction across horizons for the primary test against the
  original LSTM; Clark–West vs the RW; Pesaran–Timmermann on non-overlapping origins.

## Results (seed-averaged RMSE, USD/bbl)

| Period | h | RW | Original LSTM | Chronos-2 | Gated residual (D1) | Chronos-feature (D2) | D2, Chronos masked |
|---|---|---|---|---|---|---|---|
| Test 2022–25 | 1 | 1.969 | **1.964** | 2.081 | 2.029 | 1.969 | 1.973 |
| | 5 | 4.303 | **4.278** | 4.427 | 4.409 | 4.321 | 4.338 |
| | 20 | **7.015** | 7.278 | 7.285 | 7.575 | 7.558 | 7.732 |
| 2026 | 1 | 3.725 | **3.724** | 3.844 | 3.742 | 3.727 | 3.727 |
| | 5 | **7.599** | 7.647 | 7.738 | 7.723 | 7.648 | 7.664 |
| | 20 | 14.276 | 14.534 | 14.644 | 15.046 | **14.162**\* | 14.192\* |

\* Design 2 was built after the 2026 numbers had been seen, so 2026 is not a clean holdout for it.
Its single "win" there is not significant: DM vs the RW p = 0.69, Clark–West p = 0.21.

**Significance**
- **Design 1 vs the original LSTM** (the primary test): worse at every horizon, though not significantly after the Holm correction
  (test h1: DM = 2.13, raw p = 0.034, Holm p = 0.10).
- **Design 2 vs the original LSTM:** no significant difference (Holm p ≥ 0.23).
- **Original LSTM vs the RW:** the only positive signal. Clark–West is significant on the test period at h = 1 (p = 0.017) and
  h = 5 (p = 0.032), and Pesaran–Timmermann direction is significant at h = 1 (p = 0.005). Neither holds in 2026.
- **Chronos-2 zero-shot vs the RW:** worse at every horizon (test h1: DM p = 0.016).
- **Intervals:** Chronos-based 80% intervals are roughly calibrated on the test period (0.75–0.81) but too narrow in 2026 (0.64–0.78).

All tables: `results/cra/tables/summary.md` and `results/cra/followup/tables/summary.md`.
Figures: `results/cra/figures/`, `results/cra/followup/figures/`.

## Why performance did not improve

1. **Little predictable signal in oil price levels.** The original LSTM's prediction std is about 5–20% of the actual moves
   (`pred_std_ratio`), so it is effectively the RW plus a tiny drift. That is why it is hard to beat, and also hard to improve.
2. **Zero-shot Chronos-2 is worse than the RW** on this series at every horizon. Anchoring on its median (Design 1) adds its
   error. The model learned to keep the gate mostly closed (mean 0.10–0.24), which still left RMSE above the LSTM's.
3. **The model barely uses Chronos.** Masking the Chronos block in Design 2 changes RMSE by < 0.2 (h20 test: 7.56 vs 7.73;
   h1: 1.969 vs 1.973). The gate settles around 0.44–0.49, i.e. it neither trusts nor ignores the embedding.
4. **The extra inputs are not used either.** The learned variable-selection weights are close to uniform
   (0.055–0.078 against 1/15 = 0.067; `variable_weights.csv`). The network found no input clearly more useful than another.
5. **2026 is a regime change.** RMSE roughly doubles in 2026 (h1: 3.7 vs 2.0), and no model trained on earlier years adapts.
6. **GPR revisions were a hidden issue in Design 1.** The archived vintages show the GPR values available at the time
   correlate only 0.29–0.50 with today's revised file (`gpr_alignment.csv`). Design 1 used the revised series with a 1-day lag,
   so it trained on information that was not available in real time. Design 2 uses the vintages from 2022-02-24 on; earlier dates still rely on the revised file shifted to its weekly release schedule.

## Proposed next steps

- **Fine-tune Chronos-2 on oil** (LoRA or a short full fine-tune on pre-2022 data) instead of using it zero-shot.
- **Combine forecasts instead of correcting them:** shrink toward the RW, e.g. a validation-fitted weight on the LSTM or Chronos
  forecast, which can only match or beat the RW on validation.
- **Score probabilistic forecasts with CRPS or pinball loss**, where Chronos's roughly calibrated intervals may add value even
  though its median does not.
- **Add fundamentals with real forecasting power:** the futures curve (contango/backwardation slope), EIA weekly inventories,
  and OPEC+ announcements.
- **Model regimes or recalibrate volatility** (e.g. GARCH or online recalibration of σ) to handle shifts like 2026.
- **Keep a clean holdout:** freeze the designs now and evaluate once on 2027 data.

## Reproduction

```bash
cd Code
pip install -r requirements.txt -r requirements-cra.txt  # tested: Python 3.13, torch 2.14, chronos-forecasting 2.3.2, pandas 3.1
pytest -q tests/test_cra.py tests/test_cra_followup.py   # 20 tests

python 15_cra_fetch_data.py            # FRED, Yahoo, GPR -> data/cra/  (--skip-download rebuilds levels.csv only)
python 16_cra_chronos_prior.py         # Chronos-2 quantiles per origin (~0.5 GB weights; ~20 s on a GPU)
python 17_cra_train.py                 # original LSTM + Design 1: 2 models x 3 h x 5 years x 5 seeds = 150 runs
python 18_cra_evaluate.py              # results/cra/tables, figures
python 19_cra_fetch_gpr_vintages.py    # 298 archived weekly GPR vintages
python 20_cra_followup_train.py        # Design 2: 75 runs
python 21_cra_followup_evaluate.py     # results/cra/followup/tables, figures
```

- **Rebuilding tables without retraining:** the per-seed predictions and run logs (`results/cra/**/predictions`, `**/runs`) are
  committed, so `18` and `21` rebuild every table directly from them. Re-running both against the committed files gave identical
  tables.
- **Checkpoints:** model files (`results/cra/**/models/*.pt`) are git-ignored, and `17`/`20` regenerate them.
- **Data vintage:** downloaded data runs to 2026-11-24. A fresh download adds later days, so 2026 numbers will shift slightly.
  Use `--skip-download` with the committed raw files to reproduce exactly.
