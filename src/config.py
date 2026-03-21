"""Project-wide constants: study area, grid, and data sources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Nuremberg city centre + buffer (~25 km) in WGS84 (min_lon, min_lat, max_lon, max_lat)
NUREMBERG_BBOX_WGS84: Tuple[float, float, float, float] = (
    10.85,
    49.30,
    11.35,
    49.60,
)

# Working projected CRS for Sentinel-2 / grid (UTM 32N covers Bavaria)
WORKING_CRS = "EPSG:32632"

# Grid cell size in metres (assignment: choose and justify; 300 m balances speed vs. spatial detail)
GRID_RESOLUTION_M = 300

# Two time periods for change analysis (L2A products).
# Chosen to match public ESA WorldCover annual maps (2020 v100, 2021 v200).
S2_YEARS = (2020, 2021)
# Target months (reduce seasonal swing); summer composites
S2_MONTH = 7
S2_MAX_CLOUD = 40  # percent — used in STAC query text filter where supported

# Sentinel-2 L2A surface reflectance bands (Planetary Computer asset names)
S2_ASSETS = ("B02", "B03", "B04", "B08", "B11", "B12")

# ESA WorldCover on AWS Open Data (same bucket for v100 2020 and v200 2021)
WORLDCOVER_BUCKET = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"
WORLDCOVER_TILES = {
    2020: "v100/2020/map/ESA_WorldCover_10m_2020_v100_N48E009_Map.tif",
    2021: "v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif",
}

# Class groupings for composition labels (WorldCover v200 legend — adjust if using v100 docs)
# https://esa-worldcover.org/en/data-access
WORLDCOVER_CLASS_NAMES = {
    10: "tree",
    20: "shrub",
    30: "grassland",
    40: "cropland",
    50: "built_up",
    60: "bare_sparse",
    70: "snow_ice",
    80: "water",
    90: "herbaceous_wetland",
    95: "mangroves",
    100: "moss_lichen",
}

# Assignment-focused coarse groups (built / vegetation / water / other)
BUILT_CLASSES = {50}
VEGETATION_CLASSES = {10, 20, 30, 40, 90, 95, 100}
WATER_CLASSES = {80}


@dataclass(frozen=True)
class Paths:
    raw: str = "data/raw"
    processed: str = "data/processed"
    models: str = "artifacts/models"


PATHS = Paths()

# Modeling: predict 2021 coarse composition from 2020 tabular state + summer imagery
PREDICTION_BASE_YEAR = S2_YEARS[0]
PREDICTION_TARGET_YEAR = S2_YEARS[-1]

FEATURE_COLUMNS_T0 = [
    "s2_b02",
    "s2_b03",
    "s2_b04",
    "s2_b08",
    "s2_b11",
    "s2_b12",
    "ndvi",
    "ndbi",
    "mndwi",
    "ndwi",
    "label_prop_built",
    "label_prop_vegetation",
    "label_prop_water",
    "label_prop_other",
]

TARGET_COLUMNS_COMPOSITION_T1 = [
    "label_prop_built",
    "label_prop_vegetation",
    "label_prop_water",
    "label_prop_other",
]
