# Group 4 — Satellite Tanker CNN Plan

Nov 23, 2026 · @Ingri

## The idea

Train a CNN to find and measure oil tankers in Sentinel-1 radar images of about 12 OPEC crude export terminals. Turn its detections into a weekly "tankers loading" index, and test whether that index helps the existing GRU forecast Brent. This is the signal side of our *Signal or Variance?* question: every current input is a market price, while tankers at berth are physical supply that markets see only through reports.

The trick that makes labels feasible: at a crude export terminal (single-point moorings, oil jetties), nearly every vessel longer than about 200 m is a tanker. The CNN only has to detect vessels and estimate their length well; location plus length then gives the tanker size class (VLCC, Suezmax, Aframax) and a barrels-capacity proxy.

Deep learning appears three times: a detector trained on public labeled SAR data, fine-tuned on our own hand-labeled OPEC chips; a tanker-versus-other classifier; and, as the stretch goal, an end-to-end model where the CNN backbone feeds the GRU directly.

## Labeled datasets for the CNN

Use the Sentinel-1 radar sets as the core, because the model will run on Sentinel-1. The optical sets are for a Sentinel-2 variant and for the tanker-shape classifier.

| Dataset | Sensor, resolution | Size | Tanker label? | Use in this project | License |
| --- | --- | --- | --- | --- | --- |
| [xView3-SAR](https://iuu.xview.us/) | Sentinel-1, \~10 m | 991 full scenes, \~29,400 × 22,400 px each | No: vessel vs. non-vessel, length | Pretrain the detector and length head | Free with registration |
| [OpenSARShip](https://opensar.sjtu.edu.cn/DataAndCodes.html) | Sentinel-1 | 34,528 ship chips with AIS info | Yes: Tanker among 16 classes; 61% are Cargo | Train the tanker-vs-other classifier | Free, cite the papers |
| Own OPEC terminal chips | Sentinel-1 (and Sentinel-2) | Target \~400 chips, hand-labeled | Yes, by location + length | Fine-tune and test in-domain | Ours |
| [HRSID](https://github.com/guxinyi-hitsz/HRSID) | High-res SAR | Detection boxes | No | Optional extra detector pretraining | Academic |
| [ShipRSImageNet](https://www.github.com/zzndream/ShipRSImageNet) | Optical, up to 0.3 m | 3,435 images, 17,573 ships, oriented boxes | Yes: Oil Tanker class | Optical tanker classifier; Planet or very-high-res variant | Academic only |
| [FAIR1M](https://huggingface.co/datasets/blanchon/FAIR1M) | Optical, 0.3–0.8 m | 15,000 images planned, 1,732 released | Yes: Liquid Cargo Ship | Same as ShipRSImageNet | Check terms |
| [CLS Sentinel-2 ships](https://zenodo.org/records/10418786) | Sentinel-2, 10 m | 60 scenes, 1,147 ships with length and heading | No | Sentinel-2 detector variant | Open (Zenodo) |
| [Finnish Sentinel-2 vessels](https://zenodo.org/records/10046342) | Sentinel-2, 10 m | 15 products, boxes incl. wakes | No | Extra Sentinel-2 boxes | Open (Zenodo) |

Global Fishing Watch radar detections are not a tanker label source: its vessel-identity classes have no "tanker". Use them only to check vessel counts and positions.

## Imagery to run the CNN on

Export Sentinel-1 GRD chips (VV + VH, 10 m) from Google Earth Engine: one 512 × 512 chip (about 5 × 5 km) per terminal per pass, 2019 to today. That is roughly 12 sites × \~1–2 passes a week × 7 years, several thousand chips, small enough for one disk.

| Source | What | Link |
| --- | --- | --- |
| Sentinel-1 GRD in Earth Engine | Radar chips, all weather, day and night | [COPERNICUS/S1\_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) |
| Sentinel-2 in Earth Engine | Optical chips for the Sentinel-2 variant (Gulf is mostly cloud-free) | [COPERNICUS/S2\_SR\_HARMONIZED](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| Copernicus Data Space | Full scenes if Earth Engine quotas bite | [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) |
| ASF Vertex | Sentinel-1 archive search and download | [search.asf.alaska.edu](https://search.asf.alaska.edu/) |
| IMF PortWatch | Daily tanker port calls per port, from AIS: validates our index | [portwatch.imf.org](https://portwatch.imf.org/) |
| Global Fishing Watch API | Radar vessel detections matched to AIS: checks our counts | [GFW API docs](https://globalfishingwatch.org/our-apis/documentation/docs/v3/general-api-doc/key-concepts) |

**Terminals (draw each AOI in [geojson.io](https://geojson.io/) around the berths and moorings):** Ras Tanura and Ju'aymah (Saudi Arabia, Gulf), Yanbu (Saudi Arabia, Red Sea), Basra Oil Terminal (Iraq), Mina al-Ahmadi (Kuwait), Kharg Island (Iran), Es Sider (Libya), Arzew (Algeria), Bonny and Forcados (Nigeria), Jose (Venezuela), Djeno (Congo). Optional: Fujairah (UAE, left OPEC on 1 May 2026, but the main export route that bypasses Hormuz).

Revisit is uneven: Sentinel-1B failed in December 2021, and Sentinel-1C only joined in 2025. Record the number of passes per site per week and normalize by it.

## Pipeline: from labeled images to a price forecast

Six scripts, numbered to follow the existing `08_make_figures_tables.py`.

&#91;embedded content: pipeline · train, run, index, forecast\]

The detector is trained once on labeled data, then run on every Sentinel-1 pass; its counts become the index the GRU reads.

1. **`09_export_s1_chips.py` — build the image set.** Earth Engine export of VV/VH chips per terminal per pass; convert to dB, clip to \[-25, 0\] dB, stack VV, VH and VV−VH as three channels so ImageNet-style backbones accept them. Save a `chips.csv` index (site, date, orbit, pass id).
2. **`10_label_chips.py` + CVAT/Roboflow — our own labels.** Sample \~400 chips stratified by site and year. Draw oriented boxes on every vessel; tag each as tanker / other from berth location and length. Two people label 50 chips in common to measure agreement. Freeze 100 chips from 2 held-out terminals as the in-domain test set.
3. **`11_train_detector.py` — the CNN.**
   1. Model: YOLOv8-s with oriented boxes (or Faster R-CNN, ResNet-50 FPN, via torchvision), plus a regression head for length in metres.
   2. Stage A, pretrain on xView3 vessel boxes and lengths.
   3. Stage B, fine-tune on our OPEC chips; add a tanker-vs-other head trained on OpenSARShip tanker / cargo / other chips.
   4. Augmentation that respects SAR: 90° rotations and flips (ships have no preferred heading), speckle noise, ±2 dB gain. No colour jitter.
   5. Metrics: mAP@0.5 per site, length MAE (m), tanker F1, all on the held-out terminals.
4. **`12_build_tanker_index.py` — detections to a time series.** Per pass: tanker count, total tanker length, count per size class (VLCC > 300 m, Suezmax 250–300 m, Aframax 220–250 m) and a capacity proxy (VLCC ≈ 2.0 M bbl, Suezmax ≈ 1.0 M, Aframax ≈ 0.7 M). Aggregate to weekly (`W-FRI`), divide by passes that week, and mask weeks with no pass. Check the index against PortWatch weekly tanker calls for the same ports.
5. **`13_price_with_satellite.py` — the forecast.** Reuse `04_train_rnn.py` with weekly data and new feature sets: `price_only`, `satellite` (price + per-site and total tanker indices), `satellite+market`. SHAP (`05`) then shows which terminals drive the forecast.
6. **`14_end_to_end.py` — stretch.** Freeze the detector backbone; each week's chip per site → 512-d embedding → attention across sites → GRU → next-week log return. Train with an auxiliary loss on tanker count so the embeddings stay physical:

```latex
L = \mathrm{MSE}(\hat r_{t+h}, r_{t+h}) + \lambda \sum_{s} \mathrm{MSE}(\hat n_{s,t}, n_{s,t})
```

Here r is the h-week log return, n the detector's tanker count at site s, and λ is tuned on validation.

## Experiments and hypotheses

The hypotheses continue the Extension Plan's numbering (H4–H9).

| ID | Hypothesis | Test | Metric |
| --- | --- | --- | --- |
| H10 | Pretraining on xView3 then fine-tuning on OPEC chips beats training on OPEC chips alone | Same detector, with and without Stage A | mAP@0.5 and length MAE on the 2 held-out terminals |
| H11 | The tanker index tracks real loadings | Weekly correlation with PortWatch tanker calls; event check on the Hormuz closure (from 28 Feb 2026) | Pearson r per site; drop in index after 28 Feb |
| H12 | Adding the tanker index improves weekly Brent forecasts over price only | GRU: `price_only` vs `satellite` vs `satellite+market`, h = 1 and 4 weeks, 10 seeds | RMSE vs random walk, DM-HLN test, Pesaran–Timmermann for direction |
| H13 | The index predicts volatility better than returns | Same models, target = next-4-week realized volatility | RMSE vs HAR-RV |
| H14 | The end-to-end CNN → GRU beats the two-stage pipeline | Script 14 vs script 13 | Same as H12 |

**Splits.** Detector: split by site, never by random chip, so neighbouring passes can't leak. Price model: train 2019–2022, validate 2023, test 2024–2025, and treat 2026 as a separate stress test because of the Hormuz shock. That leaves about 200 weekly training points, so keep the GRU small (hidden 16–32, 1 layer) and lean on the per-site panel.

**Baselines.** Random walk; price-only GRU; PortWatch tanker calls as an AIS-based, non-image feature. If our CNN index beats the PortWatch feature, the images added information beyond AIS.

## Risks and fallbacks

| Risk | Why it matters | Fallback |
| --- | --- | --- |
| Domain gap: xView3 is mostly European waters | Gulf terminals have different clutter (jetties, platforms, SPM buoys) | Fine-tune on our \~400 OPEC chips; report per-site mAP |
| Tanker class is imbalanced in OpenSARShip (61% Cargo) | The classifier may call every tanker "cargo" | Weighted loss or 3-class subset; at terminals, fall back to the location + length rule |
| Uneven Sentinel-1 revisit (one satellite 2022–early 2025) | Some weeks have no pass, which looks like "zero tankers" | Mask no-pass weeks, normalize by pass count, or aggregate to 2 weeks |
| Moored vs. anchored ships look alike | Ships waiting at anchor are not loading | Separate berth polygons from anchorage polygons; count each |
| Small weekly sample (\~200 training weeks) | GRU overfits; seed variance dominates | Tiny GRU, 10 seeds, per-site panel; report seed spread as in the Extension Plan |
| Earth Engine export quotas or delays | Chips arrive late | Start exports on day 1; fall back to Copernicus Data Space for missing sites |
| Labeling takes longer than planned | Delays everything after step 2 | Pre-label with the xView3-pretrained detector, then correct the boxes |

## Timeline

&#91;embedded content: work plan · Nov 24 to Dec 9, 2026\]

The Nov 25 gate needs two terminals' chips exported and the xView3-pretrained detector running on them; if not, drop the end-to-end stretch and the Sentinel-2 variant. Thanksgiving (Nov 26) falls inside labeling, so split the 400 chips before the break.

## Sources

- [xView3-SAR paper (NeurIPS 2022)](https://arxiv.org/pdf/2206.00897) and [data portal](https://iuu.xview.us/)
- [OpenSAR data download (OpenSARShip)](https://opensar.sjtu.edu.cn/DataAndCodes.html); [SAR ship classification survey, class balance](https://arxiv.org/pdf/2503.11906)
- [ShipRSImageNet repository](https://www.github.com/zzndream/ShipRSImageNet)
- [FAIR1M on Hugging Face](https://huggingface.co/datasets/blanchon/FAIR1M); [FAIR1M paper](https://arxiv.org/pdf/2103.05569)
- [HRSID repository](https://github.com/guxinyi-hitsz/HRSID)
- [CLS Sentinel-2 ship dataset](https://zenodo.org/records/10418786); [Finnish Sentinel-2 vessel dataset](https://zenodo.org/records/10046342)
- [Global Fishing Watch API key concepts](https://globalfishingwatch.org/our-apis/documentation/docs/v3/general-api-doc/key-concepts)
- [IMF PortWatch data access (FAQ)](https://portwatch.imf.org/pages/faqs)
- [Sentinel-1 mission status (ASF)](https://hyp3-docs.asf.alaska.edu/sentinel1/)
- [Kayrros tracks oil storage with Sentinel-1 (ESA)](https://sentinels.copernicus.eu/web/success-stories/-/copernicus-sentinel-1-helps-monitor-oil-industry-activity)
- [IEA on the Strait of Hormuz shock](https://www.iea.org/commentaries/how-global-oil-supplies-have-readjusted-to-help-fill-the-huge-gap-left-by-the-strait-of-hormuz-shock)
