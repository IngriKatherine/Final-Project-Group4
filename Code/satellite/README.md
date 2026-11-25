# Satellite track: ships in the Strait of Hormuz → "supply of oil" → 2026 oil prices

A CNN (Faster R-CNN, pretrained) finds ships in free Sentinel-2 images of the Strait of Hormuz.
Each ship is measured, converted into tanker capacity, and summed into a weekly
**supply-of-oil series** that the group's LSTM/GRU uses to forecast prices in 2026.

Why images and not only ship-tracking (AIS) data: in 2026 many tankers crossed Hormuz with
their AIS transponders switched off ("dark"). AIS-based counts such as IMF PortWatch miss
them; a satellite image does not. The key result to look for is the satellite estimate
sitting **above** the AIS line after 28 Feb 2026.

## 1. Data: what we use and why

| Source | What | Access | Role |
|---|---|---|---|
| **Sentinel-2 L2A, 10 m** (ESA, on AWS Open Data) | Optical images every 2–5 days; tankers are 15–33 px long | [Earth Search STAC](https://earth-search.aws.element84.com/v1), collection `sentinel-2-c1-l2a` (Collection 1), no account, fastest from AWS | **Main input** (s01, s02) |
| **Ai2 Sentinel-2 vessel dataset** | 43,443 Sentinel-2 patches with 37,145 hand-labelled ships | `wget https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar` ([docs](https://github.com/allenai/rslearn_projects/blob/master/docs/sentinel2_vessels.md)) | **Training labels** (s03) |
| **Our Hormuz chips** | ~300 chips from our own scenes, hand-corrected | s03b export → Label Studio / CVAT / Roboflow → s03b import | **In-domain labels + test set** |
| **IMF PortWatch** | Daily AIS transits through Hormuz: tanker count + tonnage | [ArcGIS REST API](https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Chokepoints_Data/FeatureServer), no key | Calibration (≤2025) and baseline (s07) |
| **Sentinel-1 radar** via **NASA ASF DAAC** | All-weather radar, full archive | Free [Earthdata Login](https://urs.earthdata.nasa.gov); `asf_search` + NASA HyP3 RTC | Optional (s01b): needs a radar detector |
| GFW Sentinel-2 detections | Vessel detections with length, CC0 | [Zenodo](https://zenodo.org/records/18346615) / GFW data portal | Optional extra labels (s03 gfw) |

**"Does NASA have this data?"** Partly. NASA's ASF DAAC distributes the complete Sentinel-1
radar archive (s01b downloads it), and NASA's HLS product harmonises Landsat + Sentinel-2 at
30 m. But 30 m is too coarse here: a VLCC is ~11 px and an Aframax ~8 px. NASA's VIIRS
night-boat detections target lit fishing boats. So the main input is the original 10 m
Sentinel-2 imagery (ESA data, hosted free on AWS), with NASA's radar as the all-weather
extension.

## 2. Method

```
s01 search ─► s02 download ─► s05 detect (CNN) ─► s06 supply series ─► s07 calibrate + export ─► price model
                                   ▲
s03 Ai2 labels ─► s04 train ─┐     │
s03b Hormuz labels ──────────┴─► s04 fine-tune
```

1. **Images (s01, s02).** For every date 2023-01 → 2026-11, read only the AOI window
   (56.0–57.0°E, 26.15–27.0°N) of bands B04/B03/B02/B08 + the SCL cloud mask, put it on
   one fixed 10 m UTM grid, and save one 5-band GeoTIFF per date.
2. **Labels (s03, s03b).** Convert Ai2's point labels to 512 px chips with 20 px boxes.
   Export ~300 Hormuz chips with suggested boxes, half before and half after 28 Feb 2026.
   Correct them by hand, then import, split by date into train/val/test.
3. **CNN (s04).** torchvision Faster R-CNN ResNet-50 FPN v2, initialised from COCO.
   Alternatives are ImageNet, SSL4EO-S12 (self-supervised on Sentinel-2), or nothing.
   Anchors are 16–256 px and chips are upsampled 512 → 800 px for small ships. Augmentation
   uses flips, 90° rotations and brightness/contrast/gamma jitter. Training uses SGD with
   warm-up + cosine decay and mixed precision.
   It is trained on Ai2, then fine-tuned on Hormuz. Metrics are AP, precision, recall, F1
   and per-chip **count error**, with a ship counted as found within 100 m of its label.
4. **Detection (s05).** Sliding windows over each date. Detections on land or at a cloud
   edge are rejected. Hull length and heading come from the near-infrared band (tested:
   ±5% on synthetic hulls). Each ship is assigned to a zone: Omani traffic lanes, the
   2026 Iranian route, or the Larak/Qeshm anchorage.
5. **Supply series (s06).** Per date, ships ≥150 m are counted per zone. Deadweight is
   estimated with DWT ≈ 300,000 t × (L / 330 m)^3.3 and converted at 7.33 bbl per tonne.
   Counts are corrected for cloud using the visible share of each zone; dates where a
   zone is more than 40% hidden are dropped. Outputs: transit capacity (oil moving) and
   anchorage capacity (oil waiting).
6. **Calibration + export (s07).** One constant turns the snapshot into a flow in
   million barrels/day. It is fitted against PortWatch **on 2023–2025 only**, so 2026 is
   untouched. The export writes causal business-day features to
   `Code/data/processed/satellite_features.csv`.

## 3. Run order (AWS GPU machine)

```bash
cd Code/satellite
pip install -r requirements_satellite.txt          # + torchgeo for --backbone ssl4eo

# 0. checks before the long runs (all three should pass)
python tests/test_units.py                         # seconds: metrics, boxes, offsets, hull length...
python s00_check_setup.py --weights                # ~1 min: GPU, disk, Earth Search read, PortWatch, Ai2 link
python tests/run_smoke_test.py                     # ~5 min on GPU: whole pipeline on a synthetic strait,
                                                   # including checking that the CNN really learns

python s01_search_scenes.py                        # ~1 min; prints dates per year and the newest image
python s02_download_scenes.py --start 2025-06-01   # 2026 focus first (~100 dates)...
python s02_download_scenes.py                      # ...then the rest of 2023-2025

wget https://storage.googleapis.com/ai2-rslearn-projects-data/sentinel2_vessels/sentinel2_vessels.tar
mkdir -p data/ai2_sentinel2_vessels && tar xf sentinel2_vessels.tar -C data/ai2_sentinel2_vessels
python s03_prepare_labels.py ai2 --inspect         # check the layout matches before converting
python s03_prepare_labels.py ai2 --workers 8
python s03_prepare_labels.py preview               # LOOK at figures/labels_preview_ai2.png
python s03_prepare_labels.py stats

python s04_train_detector.py --sources ai2 --backbone coco --max-train 15000 --epochs 6 --tag ai2_coco
python s03b_hormuz_chips.py export --n 300 --ckpt runs/ai2_coco/best.pt
#   ... correct data/hormuz_to_label/labels/*.txt in Label Studio / CVAT / Roboflow (YOLO format) ...
python s03b_hormuz_chips.py import
python s03_prepare_labels.py preview               # LOOK at figures/labels_preview_hormuz.png
python s04_train_detector.py --sources hormuz --init runs/ai2_coco/best.pt --epochs 15 --lr 0.003 --tag hormuz_ft

python s05_detect_scenes.py --ckpt runs/hormuz_ft/best.pt --quicklook
python s06_build_supply_series.py
python s07_calibrate_and_export.py --ais-lag 3
```

Time one epoch before committing to a long run. Disk: scenes are 40–120 MB per date (10–30 GB
for 2023–2026), plus the Ai2 tar; plan on a ≥150 GB volume. Every long script resumes where
it stopped.

**Zones.** The polygons in `config_satellite.yaml` are approximate. After s02, open a clear
2025 scene and a 2026 scene in QGIS: check where traffic actually runs and where tankers
anchor, then edit the polygons before s05.

## 4. Using the series in the price model

`satellite_features.csv` holds HZ_SAT_FLOW_MBD, HZ_SAT_ANCHOR_MBBL, HZ_SAT_LARGE_N,
HZ_AIS_FLOW_MBD and HZ_OBS_AGE_D on business days, plus WTI/BRENT from `levels.csv`.

* **Get 2026 prices.** In `Code/config.yaml`, set `data.end_date` to `"2026-11-24"` and
  rerun `01_fetch_data.py` and `02_preprocess.py`.
* **Split.** The satellite series starts in 2023: train 2023-01 → 2025-06,
  validate 2025-07 → 2025-12, test 2026.
* **Transforms.** The supply features can be zero, so the log-return transform doesn't
  apply. Treat them like rates (first differences) without editing the shared file:
  ```python
  import oilfc.data as D
  D.RATE_COLUMNS |= {"HZ_SAT_FLOW_MBD", "HZ_SAT_ANCHOR_MBBL", "HZ_SAT_LARGE_N", "HZ_AIS_FLOW_MBD", "HZ_OBS_AGE_D"}
  ```
* **Fair comparison.** Run price only, price + AIS (HZ_AIS_FLOW_MBD) and price + satellite
  on the same 2026 test weeks. If satellite beats AIS, the images carried information that
  ship tracking did not.

## 5. Experiments for the report

| ID | Question | How |
|---|---|---|
| S1 | Does pretraining help, and which kind? | s04 with `--backbone scratch / imagenet / coco / ssl4eo`, same Hormuz data, Hormuz test AP and count MAE |
| S2 | Do in-domain labels matter? | Ai2-only model vs. + Hormuz fine-tune, on the Hormuz test chips |
| S3 | Does the series track real flows? | 2023–2025 fit R² vs PortWatch (`figures/calibration_scatter.png`) |
| S4 | Does satellite see what AIS misses? | Satellite − AIS gap after 28 Feb 2026 (`figures/satellite_vs_ais_flow.png`) |
| S5 | Does it help predict 2026 prices? | Price model: price only vs + AIS vs + satellite, 2026 test, DM test vs random walk |

## 6. Known limits

* Sentinel-2 is optical and daytime only. Cloudy dates are dropped per zone, and dust
  storms in summer may remove more.
* At 10 m a tanker cannot be told from a container ship or LNG carrier by shape. "Large
  ship ≥150 m" is the proxy, and the calibration absorbs the average tanker share. Hull
  length (±5%) gives the size class, not the cargo; laden and empty ships look alike.
* An image is a snapshot. The snapshot-to-flow constant assumes ships keep a similar speed
  through the strait; convoys or escorts in 2026 can break that.
* Ai2's labels are points, so their boxes are a fixed 20 px. Use the Hormuz chips for any
  size-sensitive check.

## 7. Tests

| Command | Time | What it proves |
|---|---|---|
| `python tests/test_units.py` | seconds | Metrics give known answers; augmentation keeps boxes on ships; hull length ±8% and heading ±5° on known hulls; old/new Sentinel-2 offsets give the same brightness; band names resolve (never B8A); zones are non-empty and don't overlap; PortWatch dates parse as numbers or text; calibration falls back when data are short |
| `python s00_check_setup.py --weights` | ~1 min | This machine can read real Sentinel-2 files anonymously, the collection has recent images, PortWatch answers, the Ai2 tar is reachable, COCO weights load, GPU and disk are adequate |
| `python tests/run_smoke_test.py` | ~5 min GPU / ~15 min CPU | Every script s02–s07 runs in order on a synthetic strait in a temporary copy (your data are untouched). On GPU it also checks that the CNN learns (val AP ≥ 0.5) and that the supply series drops after 28 Feb 2026 |

If the s01 log warns that the newest image is old, switch `stac.collection` to
`sentinel-2-l2a`. If PortWatch is down, rerun s07 from the saved copy:
`--portwatch-csv data/series/portwatch_hormuz_daily.csv`.
