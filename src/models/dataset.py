"""Build supervised matrices from ``tabular_features.parquet`` with a clear temporal split."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS_T0,
    PATHS,
    PREDICTION_BASE_YEAR,
    PREDICTION_TARGET_YEAR,
    TARGET_COLUMNS_COMPOSITION_T1,
)
from src.geo.grid import reference_grid


def load_long_table(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path or Path(PATHS.processed) / "tabular_features.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/prepare_data.py --features` (after raw rasters exist)."
        )
    return pd.read_parquet(path)


def composition_forecasting_xy(
    df: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Predict year ``PREDICTION_TARGET_YEAR`` coarse composition per cell from
    features measured at ``PREDICTION_BASE_YEAR`` only.

    Returns
    -------
    X : features indexed by cell_id (one row per cell)
    y : same order, multi-target composition at t1
    cell_ids : ndarray of cell_id for merging geometries / maps
    """
    df = df if df is not None else load_long_table()
    base = df[df["year"] == PREDICTION_BASE_YEAR].set_index("cell_id")
    tgt = df[df["year"] == PREDICTION_TARGET_YEAR].set_index("cell_id")

    missing_x = set(FEATURE_COLUMNS_T0) - set(base.columns)
    missing_y = set(TARGET_COLUMNS_COMPOSITION_T1) - set(tgt.columns)
    if missing_x or missing_y:
        raise KeyError(f"Missing columns. X: {missing_x}  y: {missing_y}")

    common = base.index.intersection(tgt.index)
    base = base.loc[common]
    tgt = tgt.loc[common]

    X = base[FEATURE_COLUMNS_T0].copy()
    y = tgt[TARGET_COLUMNS_COMPOSITION_T1].copy()
    cell_ids = common.to_numpy()

    # Drop rows with invalid inputs or labels
    valid = (
        X.notna().all(axis=1)
        & y.notna().all(axis=1)
        & (base["label_wc_valid_pixels"] > 0)
        & (tgt["label_wc_valid_pixels"] > 0)
    )
    X = X.loc[valid]
    y = y.loc[valid]
    cell_ids = cell_ids[valid.to_numpy()]

    return X, y, cell_ids


def spatial_train_test_mask(cell_ids: np.ndarray, holdout: str = "east") -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial hold-out: assign test cells by raster column (vertical fence) or row (horizontal).

    This is a simple blocked split that reduces neighbour leakage compared to i.i.d. sampling.
    """
    meta = reference_grid()
    width = int(meta["width"])
    col = cell_ids % width
    row = cell_ids // width

    if holdout == "east":
        split_val = np.median(col)
        test_mask = col > split_val
    elif holdout == "north":
        split_val = np.median(row)
        test_mask = row > split_val
    else:
        raise ValueError("holdout must be 'east' or 'north'")

    train_mask = ~test_mask
    return train_mask, test_mask
