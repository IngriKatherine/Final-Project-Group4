"""Config, paths, area-of-interest grid and zones shared by every script."""
from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

SAT_DIR = Path(__file__).resolve().parents[1]          # Code/satellite/
CONFIG_FILE = SAT_DIR / "config_satellite.yaml"


def load_cfg(file: str | Path = CONFIG_FILE) -> dict:
    with open(file) as f:
        return yaml.safe_load(f)


def P(cfg: dict, key: str, mkdir: bool = True) -> Path:
    """Absolute path for cfg['paths'][key]. Creates the folder (or the file's parent)."""
    p = (SAT_DIR / cfg["paths"][key]).resolve()
    target = p if p.suffix == "" else p.parent
    if mkdir:
        target.mkdir(parents=True, exist_ok=True)
    return p


def resolve(rel: str) -> Path:
    """Resolve a path written relative to Code/satellite/."""
    return (SAT_DIR / rel).resolve()


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj, file: str | Path) -> None:
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def gdal_env() -> dict:
    """GDAL settings for fast, anonymous reads of public Cloud-Optimized GeoTIFFs."""
    return dict(
        AWS_NO_SIGN_REQUEST="YES",
        AWS_REGION="us-west-2",                  # Earth Search buckets live in us-west-2
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.jp2",
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
        VSI_CACHE="TRUE",
    )


# -----------------------------------------------------------------------------
# Area-of-interest grid (UTM, 10 m) and zones
# -----------------------------------------------------------------------------
@dataclass
class Grid:
    crs: str
    transform: object     # rasterio Affine
    width: int
    height: int


def aoi_grid(cfg: dict) -> Grid:
    """Snap the lon/lat bounding box to a fixed UTM grid so every date lines up pixel for pixel."""
    from pyproj import Transformer
    from rasterio.transform import from_origin

    lon0, lat0, lon1, lat1 = cfg["aoi"]["bbox"]
    res = float(cfg["aoi"]["resolution"])
    tr = Transformer.from_crs("EPSG:4326", cfg["aoi"]["crs"], always_xy=True)
    xs, ys = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
    xmin, xmax = np.floor(min(xs) / res) * res, np.ceil(max(xs) / res) * res
    ymin, ymax = np.floor(min(ys) / res) * res, np.ceil(max(ys) / res) * res
    width, height = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    return Grid(cfg["aoi"]["crs"], from_origin(xmin, ymax, res, res), width, height)


def zones_lonlat(cfg: dict) -> dict:
    """{name: (type, shapely Polygon in lon/lat)}."""
    from shapely.geometry import Polygon
    return {name: (z["type"], Polygon(z["polygon"])) for name, z in cfg["zones"].items()}


def zone_masks(cfg: dict, grid: Grid) -> dict[str, np.ndarray]:
    """Boolean raster mask per zone on the AOI grid."""
    from pyproj import Transformer
    from rasterio.features import rasterize
    from shapely.geometry import Polygon

    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    out = {}
    for name, (_, poly) in zones_lonlat(cfg).items():
        x, y = tr.transform(*zip(*poly.exterior.coords))
        out[name] = rasterize([(Polygon(zip(x, y)), 1)], out_shape=(grid.height, grid.width),
                              transform=grid.transform, fill=0, dtype="uint8").astype(bool)
    return out


def pixel_to_lonlat(grid: Grid, cols, rows):
    """Pixel centres (col, row) on the AOI grid -> lon, lat arrays."""
    from pyproj import Transformer
    from rasterio.transform import xy

    xs, ys = xy(grid.transform, np.asarray(rows), np.asarray(cols), offset="center")
    tr = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.asarray(xs), np.asarray(ys))
    return np.asarray(lon), np.asarray(lat)


def lonlat_to_pixel(grid: Grid, lon, lat):
    from pyproj import Transformer
    from rasterio.transform import rowcol

    tr = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    x, y = tr.transform(np.asarray(lon), np.asarray(lat))
    rows, cols = rowcol(grid.transform, np.atleast_1d(x), np.atleast_1d(y))
    return np.asarray(cols), np.asarray(rows)
