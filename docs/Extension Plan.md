# Group 4 — Extension Plan: CNNs, Optimizers & Augmentation for WTI Forecasting

Nov 22, 2026 · @Ingri

## 1. What the project currently does

The project asks whether SHAP-guided feature selection improves LSTM and GRU forecasts of daily WTI crude oil prices. So far, nothing beats the random walk.

- **Data.** 13 daily series from FRED and Yahoo Finance, 5,217 observations from 2006 to 2025. Prices enter as log returns and rates as first differences.
- **Splits.** Train 2006–2018, validation 2019–2021, test 2022–2025; scalers are fit on training data only.
- **Models.** LSTM and GRU (lookback 30, hidden size 64, 2 layers, dropout 0.2), trained with Adam at 1e-3, `ReduceLROnPlateau` and early stopping. Each is run on three feature sets (price only, all, SHAP-selected), at horizons h = 1, 5 and 20, with 3 seeds, plus one-at-a-time ablations at h = 5.
- **Benchmarks.** Random walk and rolling ARIMA, compared with Diebold–Mariano (DM) tests using the HLN correction.
- **Results.** The price-only GRU is the best network but ties the random walk (h = 1 RMSE 1.920 vs 1.925; every DM p > 0.35). Adding all features makes forecasts significantly worse at h = 1 and h = 20. SHAP selection recovers part of that loss but never beats price-only input. Directional accuracy is 0.47–0.55.

## 2. Key strengths

- **No leakage.** Samples are assigned to splits by target date, scalers are fit on training data, and SHAP uses only training and validation data.
- **An honest benchmark.** Every model faces the random walk, not just ARIMA, with DM-HLN significance tests. ARIMA alone would have made every RNN look good at h = 1.
- **SHAP on the network itself.** Importance comes from expected gradients on the RNN, not from a tree surrogate.
- **A controlled grid.** 2 models × 3 feature sets × 3 horizons × 3 seeds, plus ablations of hidden size, depth, lookback and dropout.
- **A cheap, reproducible pipeline.** Config-driven scripts 01–08 run end to end in about 30 minutes on one A10G GPU, so new experiments cost little.

## 3. Main weakness

The project is mostly a replication, and its central result, "nothing beats the random walk," is not yet explained.

- **No contribution of its own.** The design follows Firouzjaee & Khalilian and Okasha & Zaher (2026). The course's main topics (CNNs, optimization, augmentation) don't appear.
- **An uninterpretable null.** Nothing shows whether the networks learned a weak signal or simply predict "no change", or whether the pipeline could learn anything if a signal existed.
- **An unrepresentative validation window.** 2019–2021 contains COVID and the negative WTI price; validation MAPE is about 4.5% against 1.7% on test. Early stopping, the LR schedule and SHAP selection are all tuned on it.
- **Seed instability, measured with too few seeds.** Variance grows with more features (GRU, all features, h = 20: test RMSE std 0.63 against 0.06 for price only), and 3 seeds cannot estimate it reliably.
- **Untested training choices.** Adam at 1e-3, the plateau schedule and batch size 64 were never compared with alternatives.

## 4. Improvements

Reframe the study from "does SHAP help?" to "signal or variance?". The evidence says the bottleneck is weak signal plus variance, not architecture. Each course topic becomes a tool for the variance side, and a positive control tests the signal side, so even a null result explains something.

**Proposed title:** *Signal or Variance? SHAP Selection, Convolutional Encodings, Optimizers and Augmentation for Daily WTI Forecasts*

**Revised research question:** Daily WTI forecasts from LSTM and GRU tie the random walk, and extra inputs mainly add variance. Do convolutional inductive bias, the choice of optimizer and LR schedule, and train-only augmentation reduce that variance, and does that buy accuracy or directional skill beyond the random walk?

H1–H3 stay as they are. Everything that already works stays too: splits, train-only scalers, random walk and ARIMA, DM-HLN tests and seeds 0–2.

Tested code for every new part (image encoding, CNNs, augmentation, the optimizer loop, the checks and the Pesaran–Timmermann formula) is in Code sketches.

| ID | New hypothesis | Expected |
| --- | --- | --- |
| H4 | With price-only input, neither a 1D CNN nor a GAF image CNN beats the random walk, and the image model doesn't beat the 1D CNN | Supported (null) |
| H5 | As direction classification, no model shows directional skill (Pesaran–Timmermann) or beats the training base rate on Brier score; soft labels lower seed dispersion but add no skill | Supported (null) |
| H6 | With each optimizer's LR tuned, RMSE differences between optimizers and schedules are smaller than the seed spread; convergence speed, LR sensitivity and stability differ; weight averaging (EMA) lowers seed dispersion | Partly supported |
| H7 | Valid augmentation lowers seed dispersion for multi-feature inputs without collapsing predictions, but multi-feature input still doesn't beat price-only; time reversal hurts | Uncertain: the most interesting one |
| H8 | On realized volatility (positive control), at least one network beats the naive volatility benchmark | Likely; if not, suspect the pipeline |
| H9 | Different architectures make less correlated errors than seeds of one architecture, but their combination still doesn't beat the random walk, because most large errors are shared | Supported |

### 4.1 CNNs

The CNN work is justified for oil through second moments: image pixels built from returns expose volatility structure, which is predictable, rather than direction, which is not.

| Experiment | What | Why it fits this problem | New runs |
| --- | --- | --- | --- |
| E1: 1D CNN | Causal dilated convolutions (kernel 5, dilations 1, 2, 4) cover 29 of the 30 lookback days; about 12k parameters against 38.7k for the GRU | Fewer parameters with the same window, and the clean control for the image model | 27 |
| E1: GAF image CNN | Each 30-day window becomes Gramian Angular Summation and Difference Field images, fed to a small 2D CNN | GASF pixels are products of returns, so the CNN sees volatility clustering laid out in space | 27 |
| E2: Direction classification | The same three models trained with BCE on 1\[return > 0\]. Baselines: training base rate, momentum sign, logistic regression. Tests: Pesaran–Timmermann, and DM on Brier score | Turns the image model into literal image classification and tests directional skill properly | 27 |
| E2s: Soft labels | Targets Φ(r / 0.5σ) instead of 0/1, with σ from training data; evaluated on hard labels | A +0.01% day is a doubtful "up"; softening doubtful labels reduces memorization | 27 |
| E5: Positive control | Target = log realized variance over the next 5 or 20 days. Benchmarks: last 22-day volatility and HAR-RV | If a network wins here but not on price, the price null is about the data, not the method | 18 |
| E6: Error diversity | Error correlation between all models; share of large errors shared by every model; equal-weight combination of GRU, 1D CNN and GAF-CNN against a 3-seed GRU ensemble | A model that errs differently adds more to a combination than a stronger copy | 0 |

```latex
\tilde u_i = \operatorname{clip}(x_i/3,\,-1,\,1),\quad \mathrm{GASF}_{ij} = \tilde u_i\tilde u_j - \sqrt{1-\tilde u_i^2}\sqrt{1-\tilde u_j^2},\quad \mathrm{GADF}_{ij} = \sqrt{1-\tilde u_i^2}\,\tilde u_j - \tilde u_i\sqrt{1-\tilde u_j^2}
```

**Design rules for the image model**

- Scale with fixed bounds from training data (±3 training standard deviations). Per-window min–max scaling would erase the volatility level.
- Keep track of which corner is "today": add coordinate channels and pool to a 4 × 4 grid, not a global average.
- Build the images inside `forward()` from the existing windows, so `data.py`, the splits and the scalers stay untouched.
- Skip wavelet scalograms (30 points are too coarse, and a transform over the full series leaks the future), and skip the features × lags matrix as an image (its row order is arbitrary).

**Pesaran–Timmermann caveats.** With h = 5 and h = 20 the targets overlap, so test on every h-th origin. A classifier that always says "up" makes the test undefined; report it as degenerate, which is itself a finding.

### 4.2 Optimizers

The design is the tuning protocol: give every optimizer the same LR grid density and the same budget, because rankings often flip once each optimizer is tuned fairly (Choi et al. 2019). The setting is GRU and 1D CNN, SHAP-selected inputs, h = 5, the cell your ablations already use.

**Controls**

- **Same start.** For each seed, every optimizer gets the same initial weights and minibatch order, with separate random generators for initialization, batch order and augmentation.
- **Fixed budget.** 60 epochs with no early stopping. Report both the last-epoch and the best-validation checkpoint, because early stopping on the COVID-heavy validation window adds noise to the comparison.
- **Same clipping.** Clip gradients at 1.0 for every optimizer and log the share of clipped steps.
- **Selection.** Pick each LR on validation and touch the test set once.

| Stage | Varies | Held fixed | Runs | Tier |
| --- | --- | --- | --- | --- |
| E3a | 6 optimizers (SGD, momentum, Nesterov, RMSprop, Adam, AdamW) × 5 learning rates | Constant LR | 180 | 1 |
| E3b | Schedule: cosine with 5% warm-up, one-cycle, plateau (current default) | Each optimizer's best LR | 108 | 1 |
| E3e | Weight averaging (EMA, decay 0.995) on vs off | Best LR and schedule | 36 | 1 |
| E3c | Batch size 16 and 256 (64 reused) | Adam and AdamW, best schedule | 24 | 2 |
| E3d | The winning recipe applied to the full GRU grid | All feature sets and horizons | 27 | 2 |

**LR grids.** Plain SGD 1e-2 to 1; momentum and Nesterov 1e-3 to 1e-1, because momentum 0.9 multiplies the effective step by 10; RMSprop, Adam and AdamW 1e-4 to 1e-2. Use 5 log-spaced values per grid, and extend it if the best value lands on an edge.

**What to measure besides test RMSE**

- **Convergence speed:** updates until training loss is within 5% of its own minimum.
- **LR sensitivity:** how many grid LRs land within 5% of the best validation loss.
- **Seed dispersion:** RMSE standard deviation across seeds, and the mean pairwise RMSE between seed forecasts.
- **Stability:** divergences, clipped share and gradient-norm percentiles.
- **Prediction std ratio:** std(predicted return) / std(actual). A value near 0 means the network has collapsed to the random walk. It is the most useful new diagnostic, and you can compute it today from the existing prediction files as log(pred\_price / base\_price).

### 4.3 Augmentation

Augment the series, never the picture. Apply augmentations to the 30-day return window, on training windows only, and let the encoder turn the result into an image; every image is then a valid encoding of a plausible window.

| Augmentation | Verdict | Reason |
| --- | --- | --- |
| Jitter: Gaussian noise, σ = 0.05–0.1 standardized units | Use | Acts like measurement noise; equivalent to Tikhonov regularization |
| Joint magnitude scaling of window and target | Use | Mimics volatility regimes; the target must be scaled too |
| Block bootstrap of training rows (about 20-day blocks) | Use | Keeps cross-asset correlation and short-range dependence inside blocks |
| Mixup rescaled by 1/√(λ² + (1 − λ)²) | Test | Plain mixup makes windows calmer than real ones; the rescaling keeps their volatility |
| Time reversal | Negative control only | Moves "today" to the oldest slot; returns are not time-reversible |
| Sign flip, image flips, crops, shifts, time warping, rotation, colour jitter | Never | Each destroys the meaning of position, sign or lag |
| Smoothing or wavelet denoising over the full series | Never | Leaks future values into every window |

**Leakage rules**

1. Split first, then augment only training batches, on the fly. Never augment validation or test windows.
2. Estimate every augmentation parameter from training rows only. Bootstrap blocks come only from rows dated on or before 2018-12-31, with targets recomputed inside the synthetic series.
3. Fit scalers on the original training data, not on augmented copies.
4. Tune augmentation strength on validation, and touch the test set once.

**Testing whether it reduces seed instability**

- **Enough seeds.** Use seeds 0–9 for this block (0–2 stay comparable with the existing tables), and report a bootstrap confidence interval of the ratio of standard deviations (augmented / none).
- **The collapse trap.** A network can look stable by predicting zero everywhere, so report the prediction std ratio next to seed dispersion.
- **A fair comparison.** Compare a 10-member bagged ensemble with a 10-seed ensemble, never with one model.
- **Preview first.** Before training, compare real and augmented data on standard deviation, excess kurtosis and autocorrelation of absolute returns, so you know whether an augmentation keeps fat tails and volatility clustering.
- **End on real data.** Fade augmentation strength to zero over the last 10% of epochs.

Runs: E4 = {GRU, 1D CNN} × {all, SHAP-selected} × 6 augmentations (jitter, scaling, both, block bootstrap, mixup, reversal) × h {1, 20} × 10 seeds = 480. E4n adds 56 no-augmentation runs (seeds 3–9) as the reference. h = 1 and h = 20 are where all-features input is already significantly worse than the random walk, so that is where augmentation has damage to repair.

### 4.4 Possible pre-trained models

Pretraining helps when the inputs contain patterns the network already knows; it cannot create signal that daily returns lack. Both options are therefore optional, and each is tested where it has a fair chance.

| Option | How | Expected | Main risk | Priority |
| --- | --- | --- | --- | --- |
| E1c: ImageNet ResNet-18 fine-tuned on GAF images | Upsample images to 64 × 64 and adapt the first convolution to the channel count. Train with AdamW, layer-wise LR decay, warm-up + cosine and EMA. Price and volatility targets at h = 5, 3 seeds each (6 runs) | No gain on price; a possible gain on volatility, where texture filters have something to find | About 11M parameters on 3,359 training windows invite exactly the seed variance under study; GAF textures are not natural images | Tier 2 |
| Zero-shot time-series foundation model (for example Chronos or TimesFM) | Forecast price-only WTI with no training and compare with the random walk by DM. Inference only | About the random walk | Its pretraining data may include 2022–2025 market series, which would contaminate the test period. Check the documented training data and the licence, and report it as a benchmark, not a contribution | Optional |

### 4.5 Run budget

The core tier is 986 new runs, about 7 GPU-hours, or about 3 hours with three worker processes on the g5.xlarge.

| Block | Runs | GPU-hours (estimate) | Tier |
| --- | --- | --- | --- |
| E0: Day-1 audit and diagnostics | 1 check run | minutes | 1 |
| E1, E2, E2s, E5: CNNs, direction, positive control | 126 | about 0.7 | 1 |
| E3a, E3b, E3e: optimizers, schedules, EMA | 324 | about 3.6 | 1 |
| E4, E4n: augmentation and its reference | 536 | about 2.7 | 1 |
| E6: error diversity and combination | 0 | none | 1 |
| E1b encoding ablation (GASF, GADF, MTF, recurrence plot), E1c ResNet-18, E3c batch size, E3d carry-over | 72 | about 0.8 | 2 |

The estimates scale from today's 15 s per early-stopped run, with fixed 60-epoch runs taking 2–3× longer. Time one run of each new model type on Day 1 and rescale. If a block runs more than 50% over its estimate, cut it back right away. If time runs short, cut tier 2 first, then E4 to seeds 0–4, then E3a to 4 LRs per optimizer. Never cut E0, E5 or E6: they are cheap, and they are what makes the null results interpretable.

## 5. Other lessons to consider

Most gains in a project like this come from process rather than architecture: diagnose the last run, make one change aimed at what the diagnosis showed, and check it on held-out data.

1. **Audit the pipeline before tuning (E0, Day 1).** Perturb one input series and confirm that only its image channels change; the `reshape`-instead-of-`permute` bug passes silently otherwise. Apply the sigmoid before the 0.5 threshold. Rerun one old configuration and compare its predictions row by row.
2. **Read the predictions before deciding.** On the existing 54 runs, compute the prediction std ratio, each model's error correlation with the random walk, and errors split into calm and volatile days. Let these numbers decide where compute goes.
3. **Check that tuning beats noise.** Before keeping any tuned choice (LR, schedule, EMA, augmentation strength, SHAP top-k), pick it on a random half of the 20-day validation blocks and score it on the other half, 200 times. If it doesn't beat the default on the held-out halves, keep the default.
4. **Compare by rank, not level.** Validation errors are inflated by COVID, so compare configurations by their validation rank. After the single test evaluation, report how well validation and test rankings agree; poor agreement weakens every validation-based choice, early stopping and SHAP included.
5. **Diversity beats a stronger copy.** A weaker model whose errors differ adds more to a combination than a stronger copy of an existing one. Use equal weights; tuned combination weights rarely beat them out of sample.
6. **Report where the errors are.** Give the share of squared error from the worst 5% of days, and the share of large errors shared by every model, the random walk included.
7. **A tie is a tie.** Report a block-bootstrap win rate next to each DM test. When two models tie, prefer the one with fewer parameters.
8. **Select by the metric you report.** Choose classifier epochs by Brier score or log loss, not accuracy, and say in the report which metric drove each choice. Apply Holm correction within each block of tests.
9. **Make the process safe.** The training script currently skips any run whose results file exists, so changed settings can silently reuse stale numbers. Store a config hash and a "complete" status in each run file, and rerun on any mismatch.
10. **Write the report as a decision log.** In Results, give one row per block: what changed, why, the result and the decision. Add a "What did not work" table with the evidence for each rejected idea, a numbered Lessons list in Conclusions, a source line under every figure, and an AI-assistance note in References.

## 6. Timeline to December 9

The plan fits 17 days by freezing results on Thursday, December 3, leaving six days to write, rehearse and submit by Dec 9, 2026.

&#91;embedded content: Timeline to Dec 9 · 3 members × 4 phases, 2 gates\]

No long sweep starts before the Nov 25 gate: the audit must pass, the pilot timings must be in, and the diagnostics on the existing runs must have been read together. The core runs (about 3 hours of wall-clock time) go on the AWS instance in the background over Thanksgiving, which leaves room for reruns. No new runs start after Dec 3. Member 3 owns the shared statistics and the "What did not work" log, so they are computed and recorded one way everywhere.

**Submission checklist for December 9**

- [ ] Group report (Word, saved as PDF) in `Final-Group-Project-Report/`
- [ ] 12–15-minute presentation (PowerPoint, saved as PDF) in `Final-Group-Presentation/`
- [ ] `Code/` with a README giving the run order of scripts 01–08 and the new ones
- [ ] GitHub repo `Final-Project-Group4` with the four folders and a top-level `README.md`
- [ ] Each member's individual report (`firstname-lastname-final-project.pdf`, including the share of code taken from the internet) and code, in `firstname-lastname-individual-project/`; the guidelines don't give its due date, so check the syllabus
- [ ] One member uploads the repo link to Blackboard, after checking that the link works
