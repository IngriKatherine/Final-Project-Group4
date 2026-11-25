"""Fast unit tests (seconds, no network, no GPU). Run from Code/satellite:
    python tests/test_units.py          # or: python -m pytest tests/test_units.py
Each test guards a place where a silent mistake would corrupt the results without crashing.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hormuz.common import aoi_grid, load_cfg, lonlat_to_pixel, pixel_to_lonlat, zone_masks  # noqa: E402
from hormuz.detector import augment, detection_metrics, read_yolo, write_yolo  # noqa: E402
from hormuz.measure import dwt_from_length, hull_size, size_class  # noqa: E402
from hormuz.raster import _dn_to_u8, _scale_offset, asset_key, gdal_href, valid_water  # noqa: E402

CFG = load_cfg()


def test_metrics_known_answers():
    gt = np.array([[10, 10, 30, 30], [100, 100, 120, 120]], float)
    m = detection_metrics([dict(boxes=gt, scores=np.array([.9, .8]), gt=gt)], 10, .5)
    assert m["ap"] == 1 and m["f1"] == 1 and m["count_mae"] == 0
    m = detection_metrics([dict(boxes=np.array([[10, 10, 30, 30], [300, 300, 320, 320]], float),
                                scores=np.array([.9, .8]), gt=gt)], 10, .5)
    assert abs(m["ap"] - 0.5) < 1e-9 and abs(m["precision"] - 0.5) < 1e-9
    # a false positive on an empty chip counts +1 in the count error
    m = detection_metrics([dict(boxes=np.array([[0, 0, 5, 5.]]), scores=np.array([.7]), gt=np.zeros((0, 4)))], 10, .5)
    assert m["count_bias"] == 1


def test_augment_keeps_boxes_on_the_ship():
    img = np.zeros((512, 512, 3), np.uint8)
    img[40:50, 300:340] = 255
    box = np.array([[300, 40, 340, 50]], np.float32)
    rng = random.Random(0)
    for _ in range(100):
        a, b = augment(img, box, rng)
        x0, y0, x1, y1 = b[0].round().astype(int)
        assert a[y0:y1, x0:x1].mean() > 0.5, "box no longer covers the ship"
        assert abs((a > 0.5).sum() / 3 - 400) < 40, "ship pixels lost"


def test_yolo_round_trip():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.txt"
        boxes = np.array([[10, 20, 30, 50], [500, 500, 530, 520]], float)
        write_yolo(f, boxes, 512, 512)
        back = read_yolo(f, 512, 512)
        assert np.allclose(back, [[10, 20, 30, 50], [500, 500, 512, 512]], atol=0.01)   # clipped to the chip
        write_yolo(f, np.zeros((0, 4)), 512, 512)
        assert read_yolo(f, 512, 512).shape == (0, 4)


def test_hull_length_and_heading():
    rng = np.random.default_rng(1)
    for L in (15, 25, 33):                       # pixels = 150, 250, 330 m
        for hdg in (0, 45, 100):
            img = (10 + rng.normal(0, 1.5, (128, 128))).clip(1, 255)
            yy, xx = np.mgrid[0:128, 0:128]
            a = np.radians(hdg)
            dx, dy = np.sin(a), -np.cos(a)
            u = (xx - 64) * dx + (yy - 64) * dy
            v = -(xx - 64) * dy + (yy - 64) * dx
            img[(abs(u) <= L / 2) & (abs(v) <= max(1, L / 12))] = 200
            m = hull_size(img.astype(np.uint8), 64, 64)
            assert abs(m["length_m"] - L * 10) / (L * 10) < 0.08, (L, hdg, m)
            assert min(abs(m["heading_deg"] - hdg), 180 - abs(m["heading_deg"] - hdg)) < 5, (L, hdg, m)


def test_size_classes_and_dwt():
    assert size_class(40, CFG) == "small" and size_class(100, CFG) == "medium" and size_class(330, CFG) == "large"
    assert abs(dwt_from_length(330, CFG) - 300000) < 1
    assert 80000 < dwt_from_length(245, CFG) < 140000           # Aframax ~ 80-120 kt


def test_reflectance_offset_by_baseline():
    def item(baseline, bands=None):
        extra = {"raster:bands": bands} if bands else {}
        return SimpleNamespace(properties={"s2:processing_baseline": baseline},
                               assets={"red": SimpleNamespace(extra_fields=extra)})
    assert _scale_offset(item("03.01"), "red") == (1e-4, 0.0)
    assert _scale_offset(item("05.10"), "red") == (1e-4, -0.1)          # +1000 DN offset since 2022
    assert _scale_offset(item("05.10", [{"scale": 1e-4, "offset": -0.1}]), "red") == (1e-4, -0.1)
    # the same water (reflectance 0.03) must give the same uint8 before and after the offset change
    old = _dn_to_u8(np.array([[300]], np.uint16), 1e-4, 0.0, 0.25)
    new = _dn_to_u8(np.array([[1300]], np.uint16), 1e-4, -0.1, 0.25)
    assert old[0, 0] == new[0, 0]
    assert _dn_to_u8(np.array([[0]], np.uint16), 1e-4, -0.1, 0.25)[0, 0] == 0          # no data stays 0


def test_asset_names_and_hrefs():
    it = SimpleNamespace(id="x", assets={"B04": 1, "B03": 1, "B02": 1, "B08": 1, "nir08": 1, "SCL": 1})
    assert asset_key(it, "red", CFG) == "B04" and asset_key(it, "nir", CFG) == "B08"   # never nir08 (B8A)
    assert asset_key(it, "scl", CFG) == "SCL"
    assert gdal_href("s3://bucket/a/b.tif") == "/vsis3/bucket/a/b.tif"
    assert gdal_href("https://x/y.tif") == "https://x/y.tif"


def test_grid_zones_and_coordinates():
    g = aoi_grid(CFG)
    assert 9000 < g.width < 11000 and 9000 < g.height < 10500          # ~100 x 95 km at 10 m
    masks = zone_masks(CFG, g)
    for name, m in masks.items():
        assert m.sum() > 10000, f"zone {name} is (almost) empty on the grid"
    names = list(masks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert (masks[names[i]] & masks[names[j]]).sum() < 0.01 * masks[names[i]].sum(), "zones overlap"
    lon, lat = pixel_to_lonlat(g, [1234, 5000], [4321, 100])
    c, r = lonlat_to_pixel(g, lon, lat)
    assert np.array_equal(c, [1234, 5000]) and np.array_equal(r, [4321, 100])


def test_valid_water_mask():
    rgbn = np.full((4, 4, 4), 30, np.uint8)
    rgbn[3] = 8                                           # water: NIR dark -> NDWI > 0
    rgbn[:, 0, 0] = 0                                     # no data
    rgbn[3, 1, 1] = 120                                   # land/ship: NIR bright
    scl = np.full((4, 4), 6, np.uint8)
    scl[2, 2] = 9                                         # cloud
    scl[1, 1] = 5
    v = valid_water(rgbn, scl)
    assert not v[0, 0] and not v[1, 1] and not v[2, 2] and v[3, 3]


def test_portwatch_date_parsing():
    """PortWatch may return dates as epoch milliseconds or as text; both must work (pandas 2 and 3)."""
    import s07_calibrate_and_export as s07

    for dates in ([1767225600000, 1767312000000], ["2026-01-01", "2026-01-02"], ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]):
        feats = [{"attributes": {"date": d, "portname": "Strait of Hormuz", "n_tanker": 5,
                                 "capacity_tanker": 1e5, "n_total": 9}} for d in dates]

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {"features": feats}
        orig = s07.requests.get
        s07.requests.get = lambda *a, **k: R()
        try:
            df = s07.fetch_portwatch(CFG)
        finally:
            s07.requests.get = orig
        assert list(df["date"]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")], dates


def test_calibration_fallback():
    import s07_calibrate_and_export as s07
    fit = s07.fit_k(np.array([1.0, 2.0]), np.array([2.0, 4.0]), fallback=6.6)
    assert fit["fallback"] and fit["k"] == 6.6
    x = np.arange(1, 21, dtype=float)
    fit = s07.fit_k(x, 3 * x, fallback=6.6)
    assert not fit["fallback"] and abs(fit["k"] - 3) < 1e-9 and fit["r2"] > 0.999


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
