"""Build cloud-screened Sentinel-2 L2A median composites via Microsoft Planetary Computer STAC.

Imagery is Copernicus Sentinel-2; the STAC catalog is only a distribution endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import stackstac
import xarray as xr
from rasterio.transform import from_bounds
from rioxarray import rioxarray  # noqa: F401 — registers .rio accessor

from src.config import (
    GRID_RESOLUTION_M,
    NUREMBERG_BBOX_WGS84,
    PATHS,
    S2_ASSETS,
    S2_MAX_CLOUD,
    S2_MONTH,
    S2_YEARS,
    WORKING_CRS,
)
from src.geo.grid import study_area_bounds_utm


def _month_range(year: int, month: int) -> str:
    # STAC datetime interval (inclusive semantics depend on server; month window is enough here)
    return f"{year}-{month:02d}-01/{year}-{month:02d}-28"


def _search_items(year: int, max_items: int = 80) -> List:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    min_lon, min_lat, max_lon, max_lat = NUREMBERG_BBOX_WGS84
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=(min_lon, min_lat, max_lon, max_lat),
        datetime=_month_range(year, S2_MONTH),
        query={"eo:cloud_cover": {"lt": S2_MAX_CLOUD}},
        limit=max_items,
        sortby=[{"field": "properties.datetime", "direction": "desc"}],
    )
    return list(search.items())


def _stack_items(items: Iterable, epsg: int) -> xr.DataArray:
    data = stackstac.stack(
        items,
        assets=list(S2_ASSETS),
        epsg=epsg,
        resolution=10,
        bounds_latlon=NUREMBERG_BBOX_WGS84,
        dtype=np.float64,
        rescale=False,
        fill_value=np.nan,
        properties=True,
    )
    return data


def composite_year(
    year: int,
    out_path: str | Path | None = None,
    epsg: int = int(WORKING_CRS.split(":")[-1]),
) -> Path:
    """Median composite over all qualifying scenes in July of ``year``; save GeoTIFF (UTM)."""
    out_path = Path(out_path or (Path(PATHS.raw) / f"sentinel2_median_{year}_july.tif"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = _search_items(year)
    if not items:
        raise RuntimeError(
            f"No Sentinel-2 L2A items returned for {year}-{S2_MONTH:02d}. "
            "Try widening the month, raising the cloud threshold, or another year."
        )

    cube = _stack_items(items, epsg=epsg)
    # stackstac uses axis order (time, band, y, x)
    median = cube.median(dim="time", skipna=True).compute()

    if "band" in median.coords:
        median = median.assign_coords(band=[str(b) for b in median.coords["band"].values])

    median.rio.write_crs(f"EPSG:{epsg}", inplace=True)
    median.rio.to_raster(out_path, driver="GTiff", compress="deflate", tiled=True)

    with rasterio.open(out_path, "r+") as dst:
        for i, name in enumerate(S2_ASSETS, start=1):
            if i <= dst.count:
                dst.set_band_description(i, name)

    return out_path


def composite_all_years(out_dir: str | Path | None = None) -> dict[int, Path]:
    out_dir = Path(out_dir or PATHS.raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {y: composite_year(y, out_path=out_dir / f"sentinel2_median_{y}_july.tif") for y in S2_YEARS}


def downsample_to_grid(
    raster_path: str | Path,
    dst_path: str | Path,
    resolution_m: float = GRID_RESOLUTION_M,
) -> Path:
    """Average-resample a UTM GeoTIFF to ``resolution_m`` pixels matching the study-area grid."""
    raster_path = Path(raster_path)
    dst_path = Path(dst_path)

    with rasterio.open(raster_path) as src:
        if not src.crs.is_projected:
            raise RuntimeError("Expected projected CRS on Sentinel-2 composite for grid resampling.")
        from rasterio.warp import reproject, Resampling

        dst_crs = src.crs
        west, south, east, north = study_area_bounds_utm(dst_crs)
        width = max(1, int(np.ceil((east - west) / resolution_m)))
        height = max(1, int(np.ceil((north - south) / resolution_m)))
        dst_transform = from_bounds(west, south, east, north, width, height)

        profile = src.profile.copy()
        profile.update(
            {
                "crs": dst_crs,
                "transform": dst_transform,
                "width": width,
                "height": height,
                "compress": "deflate",
                "tiled": True,
            }
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            for b in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, b),
                    destination=rasterio.band(dst, b),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.average,
                )

    return dst_path


if __name__ == "__main__":
    for y, p in composite_all_years().items():
        print("S2", y, p)
