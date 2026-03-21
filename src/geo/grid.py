"""Fishnet grid over the study area in a projected CRS."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio.transform
from rasterio.transform import from_bounds
from shapely.geometry import box

from src.config import GRID_RESOLUTION_M, NUREMBERG_BBOX_WGS84, WORKING_CRS


def study_area_gdf() -> gpd.GeoDataFrame:
    min_lon, min_lat, max_lon, max_lat = NUREMBERG_BBOX_WGS84
    gdf = gpd.GeoDataFrame(
        {"name": ["nuremberg_study_area"]},
        geometry=[box(min_lon, min_lat, max_lon, max_lat)],
        crs="EPSG:4326",
    )
    return gdf.to_crs(WORKING_CRS)


def study_area_bounds_utm(dst_crs: Any) -> Tuple[float, float, float, float]:
    """Return (west, south, east, north) in ``dst_crs`` for the study bbox."""
    area = study_area_gdf().to_crs(dst_crs)
    west, south, east, north = area.total_bounds
    return float(west), float(south), float(east), float(north)


def reference_grid(crs: Any = WORKING_CRS, resolution_m: float = GRID_RESOLUTION_M) -> Dict[str, Any]:
    """Pixel grid shared by Sentinel-2 features, WorldCover labels, and the fishnet polygons."""
    west, south, east, north = study_area_bounds_utm(crs)
    width = max(1, int(np.ceil((east - west) / resolution_m)))
    height = max(1, int(np.ceil((north - south) / resolution_m)))
    transform = from_bounds(west, south, east, north, width, height)
    return {
        "crs": crs,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
        "width": width,
        "height": height,
        "transform": transform,
        "resolution_m": resolution_m,
    }


def build_fishnet(meta: Dict[str, Any] | None = None) -> gpd.GeoDataFrame:
    """One square polygon per raster pixel (``cell_id = row * width + col``)."""
    meta = meta or reference_grid()
    transform = meta["transform"]
    height = int(meta["height"])
    width = int(meta["width"])

    polys = []
    cell_ids = []
    for row in range(height):
        for col in range(width):
            ul = rasterio.transform.xy(transform, row, col, offset="ul")
            lr = rasterio.transform.xy(transform, row, col, offset="lr")
            xmin, xmax = sorted((ul[0], lr[0]))
            ymin, ymax = sorted((ul[1], lr[1]))
            polys.append(box(xmin, ymin, xmax, ymax))
            cell_ids.append(row * width + col)

    return gpd.GeoDataFrame({"cell_id": cell_ids, "geometry": polys}, crs=meta["crs"])
