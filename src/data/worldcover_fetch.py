"""Download / clip ESA WorldCover tiles (AWS Open Data) to the Nuremberg bbox."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import rasterio
from rasterio.windows import from_bounds

from src.config import NUREMBERG_BBOX_WGS84, PATHS, S2_YEARS, WORLDCOVER_BUCKET, WORLDCOVER_TILES


def _vsicurl_url(https_url: str) -> str:
    return f"/vsicurl/{https_url}"


def clip_worldcover_year(year: int, out_dir: str | Path | None = None) -> Path:
    """
    Read the public WorldCover GeoTIFF for the N48E009 tile and write a COG clipped
    to ``NUREMBERG_BBOX_WGS84`` (EPSG:4326 bounds).
    """
    if year not in WORLDCOVER_TILES:
        raise KeyError(f"No WorldCover path configured for year={year}. Keys: {sorted(WORLDCOVER_TILES)}")
    out_dir = Path(out_dir or PATHS.raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"worldcover_{year}_nuremberg.tif"
    if out_path.exists():
        return out_path

    rel = WORLDCOVER_TILES[year]
    url = f"{WORLDCOVER_BUCKET}/{rel}"
    vsicurl = _vsicurl_url(url)

    min_lon, min_lat, max_lon, max_lat = NUREMBERG_BBOX_WGS84
    left, bottom, right, top = min_lon, min_lat, max_lon, max_lat

    with rasterio.open(vsicurl) as src:
        if src.crs.to_string() != "EPSG:4326":
            raise RuntimeError(f"Unexpected WorldCover CRS: {src.crs}")
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        profile = src.profile.copy()
        profile.update(
            {
                "height": int(window.height),
                "width": int(window.width),
                "transform": rasterio.windows.transform(window, src.transform),
                "compress": "deflate",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
            }
        )
        data = src.read(1, window=window)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
            if src.nodata is not None:
                dst.set_band_description(1, "WorldCover class code")
            dst.update_tags(
                source=url,
                citation="ESA WorldCover 2020/2021 (CC-BY-4.0); see assignment report.",
            )

    return out_path


def fetch_all_worldcover(out_dir: str | Path | None = None) -> Dict[int, Path]:
    missing = [y for y in S2_YEARS if y not in WORLDCOVER_TILES]
    if missing:
        raise KeyError(
            f"No WorldCover tile URL configured for years {missing}. "
            f"Either extend WORLDCOVER_TILES or change S2_YEARS."
        )
    return {y: clip_worldcover_year(y, out_dir=out_dir) for y in S2_YEARS}


if __name__ == "__main__":
    for y, p in fetch_all_worldcover().items():
        print(y, p, os.path.getsize(p) // (1024 * 1024), "MB")
