"""Rasterize ESA WorldCover onto the same grid as Sentinel-2 tabular features."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import rasterio
import rasterio.transform
from pyproj import Transformer
from rasterio.warp import Resampling, reproject

from src.config import (
    BUILT_CLASSES,
    VEGETATION_CLASSES,
    WATER_CLASSES,
    WORLDCOVER_CLASS_NAMES,
)
from src.geo.grid import reference_grid


def worldcover_classes_on_grid(
    worldcover_path: str | Path,
    dst_crs: str = "EPSG:32632",
) -> Tuple[np.ndarray, object]:
    """
    Nearest-neighbour warp of WorldCover (EPSG:4326 clip) onto the assignment grid.

    Returns
    -------
    classes : int32 array (H, W)
        WorldCover class codes; nodata where source has no coverage.
    transform : Affine
    """
    meta = reference_grid(crs=dst_crs)
    height, width = meta["height"], meta["width"]
    dst_transform = meta["transform"]

    dst = np.full((height, width), -1, dtype=np.int32)

    with rasterio.open(worldcover_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=-1,
        )

    return dst, dst_transform


def composition_proportions(classes: np.ndarray) -> Dict[str, np.ndarray]:
    """
    For each pixel (grid cell), compute coarse-group proportions in {0,1}.

    WorldCover stores a single dominant class per 10 m pixel; after warping to 300 m,
    each cell still holds one class code — proportions are 0/1 indicators at cell level.
    For modeling *composition within a cell*, aggregate at finer resolution inside the cell
    (see ``composition_from_fine_worldcover``).
    """
    built = np.isin(classes, list(BUILT_CLASSES)).astype(np.float32)
    veg = np.isin(classes, list(VEGETATION_CLASSES)).astype(np.float32)
    water = np.isin(classes, list(WATER_CLASSES)).astype(np.float32)
    valid = classes >= 0
    other = (valid.astype(np.float32) - built - veg - water).clip(0, 1)
    return {
        "prop_built": built,
        "prop_vegetation": veg,
        "prop_water": water,
        "prop_other": other,
    }


def composition_from_fine_worldcover(
    worldcover_path: str | Path,
    dst_crs: str = "EPSG:32632",
) -> Dict[str, np.ndarray]:
    """
    True area-fraction composition per grid cell by counting 10 m WorldCover pixels
    that fall inside each 300 m cell polygon.
    """
    meta = reference_grid(crs=dst_crs)
    height, width = meta["height"], meta["width"]
    dst_transform = meta["transform"]
    res = float(meta["resolution_m"])

    built = np.zeros((height, width), dtype=np.float32)
    veg = np.zeros((height, width), dtype=np.float32)
    water = np.zeros((height, width), dtype=np.float32)
    other = np.zeros((height, width), dtype=np.float32)
    valid_count = np.zeros((height, width), dtype=np.float32)

    with rasterio.open(worldcover_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        transform = src.transform
        src_crs = src.crs

    rows, cols = np.indices(arr.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    xm, ym = transformer.transform(xs.ravel(), ys.ravel())
    row_idx, col_idx = rasterio.transform.rowcol(dst_transform, xm, ym)
    row_idx = np.asarray(row_idx, dtype=np.int64).reshape(arr.shape)
    col_idx = np.asarray(col_idx, dtype=np.int64).reshape(arr.shape)

    flat_cls = arr.ravel()
    flat_row = row_idx.ravel()
    flat_col = col_idx.ravel()

    if nodata is not None:
        mask = flat_cls != nodata
    else:
        mask = np.ones_like(flat_cls, dtype=bool)

    in_grid = (
        mask
        & (flat_row >= 0)
        & (flat_row < height)
        & (flat_col >= 0)
        & (flat_col < width)
    )

    # Optional: distance guard so pixels far outside the fishnet are ignored
    # (should not occur for a clip that matches the bbox)

    for r, c, cls in zip(flat_row[in_grid], flat_col[in_grid], flat_cls[in_grid]):
        if cls in BUILT_CLASSES:
            built[r, c] += 1
        elif cls in VEGETATION_CLASSES:
            veg[r, c] += 1
        elif cls in WATER_CLASSES:
            water[r, c] += 1
        else:
            other[r, c] += 1
        valid_count[r, c] += 1

    total = np.maximum(valid_count, 1e-6)
    out = {
        "prop_built": built / total,
        "prop_vegetation": veg / total,
        "prop_water": water / total,
        "prop_other": other / total,
    }
    # Keep raw counts for diagnostics (label noise, mixed pixels)
    out["wc_valid_pixels"] = valid_count
    out["wc_fine_resolution_m"] = np.float32(abs(transform.a))
    out["grid_resolution_m"] = np.float32(res)
    return out


def describe_worldcover_codes() -> Dict[int, str]:
    """Legend helper for reports / notebooks."""
    return dict(WORLDCOVER_CLASS_NAMES)
