"""Read Sentinel-2 L2A bands for the AOI straight from Cloud-Optimized GeoTIFFs.

Each date's image is stored as one GeoTIFF on the fixed AOI grid with 5 uint8 bands:
    1 red (B04)  2 green (B03)  3 blue (B02)  4 nir (B08)  5 SCL (scene classification)
Bands 1-4 hold reflectance mapped linearly to 1..255 (0 = no data).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import Grid, gdal_env

BANDS = ["red", "green", "blue", "nir"]          # order of bands 1-4 in our files
SCL_CLOUD = {3, 8, 9, 10}                         # cloud shadow, cloud medium, cloud high, cirrus
SCL_NODATA = {0, 1}                               # no data, saturated/defective
SCL_WATER = 6


# Asset names differ between STAC catalogues/collections; try these in order.
# (Never "nir08": that is B8A at 20 m, not B08 at 10 m.)
ASSET_CANDIDATES = {
    "red": ["red", "B04", "b04"],
    "green": ["green", "B03", "b03"],
    "blue": ["blue", "B02", "b02"],
    "nir": ["nir", "B08", "b08", "nir08_10m"],
    "scl": ["scl", "SCL"],
}


def asset_key(item, band: str, cfg: dict | None = None) -> str:
    """The item's asset name for one of our bands (config first, then known aliases)."""
    tried = []
    if cfg is not None:
        tried.append(cfg["stac"]["assets"].get(band))
    tried += ASSET_CANDIDATES[band]
    for k in tried:
        if k and k in item.assets:
            return k
    raise KeyError(f"no asset for band {band!r} in item {item.id}; assets are {sorted(item.assets)}")


def gdal_href(href: str) -> str:
    """s3://bucket/key -> /vsis3/bucket/key (anonymous read); https and local paths unchanged."""
    if href.startswith("s3://"):
        return "/vsis3/" + href[len("s3://"):]
    return href


def _scale_offset(item, asset_key: str) -> tuple[float, float]:
    """Digital number -> reflectance. Uses STAC raster:bands metadata when present.

    Sentinel-2 processing baseline 04.00 (from 25 Jan 2022) adds +1000 to every L2A value,
    so old and new scenes only line up if that offset is removed.
    """
    asset = item.assets[asset_key]
    bands = asset.extra_fields.get("raster:bands") or []
    if bands and ("scale" in bands[0] or "offset" in bands[0]):
        return float(bands[0].get("scale", 1e-4)), float(bands[0].get("offset", 0.0))
    baseline = str(item.properties.get("s2:processing_baseline", "00.00"))
    try:
        new = float(baseline) >= 4.0
    except ValueError:
        new = False
    return 1e-4, (-0.1 if new else 0.0)


def read_band(href: str, grid: Grid, nearest: bool = False) -> np.ndarray:
    """Read one band, reprojected/resampled onto the AOI grid. Pixels outside the tile = 0."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    with rasterio.Env(**gdal_env()):
        with rasterio.open(gdal_href(href)) as src:
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                           height=grid.height, nodata=0,
                           resampling=Resampling.nearest if nearest else Resampling.bilinear) as vrt:
                return vrt.read(1)


def _dn_to_u8(dn: np.ndarray, scale: float, offset: float, max_refl: float) -> np.ndarray:
    """Digital numbers -> reflectance -> 1..255 uint8 (0 = no data), in row blocks to save memory."""
    out = np.zeros(dn.shape, np.uint8)
    for r in range(0, dn.shape[0], 1024):
        blk = dn[r:r + 1024].astype(np.float32)
        refl = blk * scale + offset
        u8 = np.clip(refl / max_refl, 0, 1) * 254 + 1
        out[r:r + 1024] = np.where(blk > 0, u8.round(), 0).astype(np.uint8)
    return out


def mosaic_date(items: list, cfg: dict, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Mosaic every tile acquired on one date onto the AOI grid (first valid pixel wins).

    Returns rgbn uint8 (4, H, W) with reflectance scaled to 1..255 (0 = no data) and the
    SCL cloud classification uint8 (H, W). Works band by band to keep memory ~1 GB.
    """
    max_refl = cfg["image"]["scale_max_reflectance"]
    rgbn = np.zeros((4, grid.height, grid.width), np.uint8)
    scl = np.zeros((grid.height, grid.width), np.uint8)
    filled = np.zeros((grid.height, grid.width), bool)
    for item in items:
        take = None
        for i, b in enumerate(BANDS):
            key = asset_key(item, b, cfg)
            dn = read_band(item.assets[key].href, grid)
            if take is None:                       # decide the pixels from this tile once
                take = (dn > 0) & ~filled
                if not take.any():
                    break
            scale, offset = _scale_offset(item, key)
            u8 = _dn_to_u8(dn, scale, offset, max_refl)
            rgbn[i][take] = u8[take]
            del dn, u8
        if take is None or not take.any():
            continue
        s = read_band(item.assets[asset_key(item, "scl", cfg)].href, grid, nearest=True).astype(np.uint8)
        scl[take] = s[take]
        filled |= take
    return rgbn, scl


def to_uint8(refl: np.ndarray, max_refl: float) -> np.ndarray:
    """Reflectance -> 1..255 (0 reserved for no data)."""
    out = np.clip(refl / max_refl, 0, 1) * 254 + 1
    out = np.where(np.isnan(refl), 0, out)
    return out.round().astype(np.uint8)


def write_scene(file: Path, rgbn_u8: np.ndarray, scl: np.ndarray, grid: Grid, tags: dict) -> None:
    import rasterio

    file.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(driver="GTiff", dtype="uint8", count=5, width=grid.width, height=grid.height,
                   crs=grid.crs, transform=grid.transform, nodata=0, compress="deflate",
                   predictor=2, tiled=True, blockxsize=512, blockysize=512)
    tmp = file.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(rgbn_u8, [1, 2, 3, 4])
        dst.write(scl, 5)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
        for i, name in enumerate(BANDS + ["scl"], start=1):
            dst.set_band_description(i, name)
    tmp.replace(file)


def read_scene(file: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """-> (rgbn uint8 (4,H,W), scl uint8 (H,W), tags)."""
    import rasterio

    with rasterio.open(file) as src:
        data = src.read()
        tags = src.tags()
    return data[:4], data[4], tags


def ndwi(green_u8: np.ndarray, nir_u8: np.ndarray) -> np.ndarray:
    """Normalised difference water index from the uint8 bands (same linear scale, so valid)."""
    g = green_u8.astype(np.float32)
    n = nir_u8.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (g - n) / (g + n)


def valid_water(rgbn: np.ndarray, scl: np.ndarray, ndwi_min: float = 0.0) -> np.ndarray:
    """Pixels that are visible open water: has data, not cloud/shadow, NDWI above threshold."""
    has_data = (rgbn[0] > 0) & ~np.isin(scl, list(SCL_NODATA))
    clear = ~np.isin(scl, list(SCL_CLOUD))
    water = (ndwi(rgbn[1], rgbn[3]) > ndwi_min) | (scl == SCL_WATER)
    return has_data & clear & water
