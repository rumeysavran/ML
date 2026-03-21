"""Turn resampled Sentinel-2 stacks into per-cell tabular features + spectral indices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import rasterio

from src.config import PATHS, S2_ASSETS, S2_YEARS
from src.data.sentinel2_fetch import downsample_to_grid
from src.data.worldcover_fetch import fetch_all_worldcover
from src.features.worldcover_labels import composition_from_fine_worldcover
from src.geo.grid import build_fishnet, reference_grid


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return num / (den + eps)


def spectral_indices_from_stack(stack: np.ndarray, band_names: Iterable[str]) -> Dict[str, np.ndarray]:
    """
    ``stack`` shape (n_bands, H, W) in the same order as ``band_names``.
    Reflectance scales from stackstac are typically 0–1 after scaling; ratios still work.
    """
    order = {b: i for i, b in enumerate(band_names)}
    b = lambda name: stack[order[name]]

    blue, green, red = b("B02"), b("B03"), b("B04")
    nir, swir1, swir2 = b("B08"), b("B11"), b("B12")

    ndvi = _safe_ratio(nir - red, nir + red)
    ndbi = _safe_ratio(swir1 - nir, swir1 + nir)
    mndwi = _safe_ratio(green - swir1, green + swir1)
    ndwi = _safe_ratio(green - nir, green + nir)

    return {
        "ndvi": ndvi.astype(np.float32),
        "ndbi": ndbi.astype(np.float32),
        "mndwi": mndwi.astype(np.float32),
        "ndwi": ndwi.astype(np.float32),
        "mean_blue": blue.astype(np.float32),
        "mean_green": green.astype(np.float32),
        "mean_red": red.astype(np.float32),
        "mean_nir": nir.astype(np.float32),
        "mean_swir1": swir1.astype(np.float32),
        "mean_swir2": swir2.astype(np.float32),
    }


def build_feature_table(
    s2_paths: Dict[int, Path],
    worldcover_paths: Dict[int, Path],
    processed_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Assemble one row per grid cell with:
    - Sentinel-2 reflectance + indices for each year
    - ESA WorldCover coarse composition for each year
    - Simple temporal deltas (2021 − 2020) for labels / change modeling
    """
    processed_dir = Path(processed_dir or PATHS.processed)
    processed_dir.mkdir(parents=True, exist_ok=True)

    meta = reference_grid()
    height, width = int(meta["height"]), int(meta["width"])
    fishnet = build_fishnet(meta)
    cell_ids = np.arange(height * width, dtype=np.int64)

    frames: List[pd.DataFrame] = []
    spec_year: Dict[int, Dict[str, np.ndarray]] = {}

    for year in S2_YEARS:
        p = Path(s2_paths[year])
        low = processed_dir / f"sentinel2_grid_{year}.tif"
        if not low.exists():
            downsample_to_grid(p, low)
        with rasterio.open(low) as src:
            stack = src.read()
            names = list(S2_ASSETS)
            if stack.shape[0] != len(names):
                raise ValueError(f"Expected {len(names)} bands, got {stack.shape[0]} for {low}")

        spec_year[year] = spectral_indices_from_stack(stack, names)
        comp = composition_from_fine_worldcover(worldcover_paths[year])

        d = {"cell_id": cell_ids, "year": year}
        for i, nm in enumerate(names[: stack.shape[0]]):
            d[f"s2_{nm.lower()}"] = stack[i].reshape(-1)

        for k, v in spec_year[year].items():
            d[k] = v.reshape(-1)

        for k in ("prop_built", "prop_vegetation", "prop_water", "prop_other"):
            d[f"label_{k}"] = comp[k].reshape(-1)

        d["label_wc_valid_pixels"] = comp["wc_valid_pixels"].reshape(-1)

        frames.append(pd.DataFrame(d))

    long = pd.concat(frames, ignore_index=True)

    years = sorted(S2_YEARS)
    if len(years) == 2:
        y0, y1 = years
        base = long[long["year"] == y0].set_index("cell_id")
        nxt = long[long["year"] == y1].set_index("cell_id")

        delta_cols: Dict[str, pd.Series] = {}
        for col in spec_year[y1].keys():
            delta_cols[f"delta_{col}"] = nxt[col] - base[col]
        for lab in ("prop_built", "prop_vegetation", "prop_water", "prop_other"):
            c = f"label_{lab}"
            delta_cols[f"delta_label_{lab}"] = nxt[c] - base[c]

        delta_df = pd.DataFrame(delta_cols).reset_index(names="cell_id")
        long = long.merge(delta_df, on="cell_id", how="left")

    out_path = processed_dir / "tabular_features.parquet"
    long.to_parquet(out_path, index=False)
    fishnet_path = processed_dir / "grid_cells.gpkg"
    fishnet.to_file(fishnet_path, driver="GPKG")

    return long


def run_default_pipeline(raw_dir: str | Path | None = None) -> pd.DataFrame:
    """Fetch WorldCover clips, expect Sentinel-2 composites present or fetch separately."""
    raw_dir = Path(raw_dir or PATHS.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    wc = fetch_all_worldcover(raw_dir)

    s2_paths = {y: raw_dir / f"sentinel2_median_{y}_july.tif" for y in S2_YEARS}
    missing = [p for p in s2_paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Sentinel-2 composites missing. Run `python scripts/prepare_data.py --fetch-s2` first. "
            f"Missing: {missing}"
        )

    return build_feature_table(s2_paths, wc)
