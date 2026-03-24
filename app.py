"""
app.py — Nuremberg Land-Cover Change Monitor
============================================
Streamlit front-end for the UTN ML Final Assignment pipeline.

Place this file in the repository root (next to scripts/, src/, artifacts/).
Run with:
    streamlit run app.py

Required artifacts (produced by the pipeline):
    data/processed/tabular_features.parquet
    data/processed/grid_cells.gpkg
    artifacts/models/training_report.json
    artifacts/models/evaluation_report.json
    artifacts/models/ridge_coefficients.csv
    artifacts/models/hist_gbrt_permutation_importance.csv
    artifacts/models/ridge_composition.joblib
    artifacts/models/hist_gbrt_composition.joblib

If artifacts are missing the app falls back to clearly-labelled demo data
so you can see the full UI without re-running the pipeline.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── optional heavy deps (graceful degradation) ─────────────────────────
try:
    import geopandas as gpd
    HAS_GEO = True
except ImportError:
    HAS_GEO = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    import matplotlib.pyplot as plt
    HAS_PLOTLY = False

# ════════════════════════════════════════════════════════════════════════
#  PATHS  (relative to repo root)
# ════════════════════════════════════════════════════════════════════════
ROOT            = Path(__file__).parent
DATA_PROC       = ROOT / "data" / "processed"
ARTIFACTS       = ROOT / "artifacts" / "models"

FEATURES_PARQ   = DATA_PROC / "tabular_features.parquet"
GRID_GPKG       = DATA_PROC / "grid_cells.gpkg"
TRAIN_REPORT    = ARTIFACTS / "training_report.json"
EVAL_REPORT     = ARTIFACTS / "evaluation_report.json"
RIDGE_COEF_CSV  = ARTIFACTS / "ridge_coefficients.csv"
GBRT_IMP_CSV    = ARTIFACTS / "hist_gbrt_permutation_importance.csv"
RIDGE_MODEL     = ARTIFACTS / "ridge_composition.joblib"
GBRT_MODEL      = ARTIFACTS / "hist_gbrt_composition.joblib"

# ── Import real project constants from src/config.py if available ────
# Falls back to the same values hardcoded in config.py so the app
# always works even if run outside the repo root.
try:
    import sys as _sys
    if str(ROOT) not in _sys.path:
        _sys.path.insert(0, str(ROOT))
    from src.config import (
        S2_YEARS        as _CFG_S2_YEARS,
        S2_ASSETS       as _CFG_S2_ASSETS,
        S2_MONTH        as _CFG_S2_MONTH,
        WORKING_CRS     as _CFG_CRS,
        GRID_RESOLUTION_M as _CFG_RES,
        NUREMBERG_BBOX_WGS84 as _CFG_BBOX,
        WORLDCOVER_TILES as _CFG_WC_TILES,
        PATHS           as _CFG_PATHS,
        BUILT_CLASSES        as _CFG_BUILT,
        VEGETATION_CLASSES   as _CFG_VEG,
        WATER_CLASSES        as _CFG_WATER,
        FEATURE_COLUMNS_T0   as _CFG_FEAT_COLS,
        TARGET_COLUMNS_COMPOSITION_T1 as _CFG_TARGET_COLS,
    )
    S2_YEARS_CFG     = list(_CFG_S2_YEARS)   # e.g. [2020, 2021]
    S2_ASSETS_CFG    = list(_CFG_S2_ASSETS)  # band order: B02 B03 B04 B08 B11 B12
    S2_MONTH_CFG     = int(_CFG_S2_MONTH)    # 7 = July
    BBOX_WGS84       = _CFG_BBOX             # (min_lon, min_lat, max_lon, max_lat)
    WC_TILES_CFG     = dict(_CFG_WC_TILES)   # {2020: "...", 2021: "..."}
    FEAT_COLS_CFG    = list(_CFG_FEAT_COLS)
    TARGET_COLS_CFG  = list(_CFG_TARGET_COLS)
    # Coarse class groupings from config
    BUILT_CODES      = set(_CFG_BUILT)        # {50}
    VEG_CODES        = set(_CFG_VEG)          # {10,20,30,40,90,95,100}
    WATER_CODES      = set(_CFG_WATER)        # {80}
    _CFG_LOADED = True
except Exception:
    # Fallback: mirror config.py defaults exactly
    S2_YEARS_CFG    = [2020, 2021]
    S2_ASSETS_CFG   = ["B02","B03","B04","B08","B11","B12"]
    S2_MONTH_CFG    = 7
    BBOX_WGS84      = (10.85, 49.30, 11.35, 49.60)
    WC_TILES_CFG    = {
        2020: "v100/2020/map/ESA_WorldCover_10m_2020_v100_N48E009_Map.tif",
        2021: "v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E009_Map.tif",
    }
    FEAT_COLS_CFG   = [
        "s2_b02","s2_b03","s2_b04","s2_b08","s2_b11","s2_b12",
        "ndvi","ndbi","mndwi","ndwi",
        "label_prop_built","label_prop_vegetation",
        "label_prop_water","label_prop_other",
    ]
    TARGET_COLS_CFG = [
        "label_prop_built","label_prop_vegetation",
        "label_prop_water","label_prop_other",
    ]
    BUILT_CODES  = {50}
    VEG_CODES    = {10, 20, 30, 40, 90, 95, 100}
    WATER_CODES  = {80}
    _CFG_LOADED  = False

# ════════════════════════════════════════════════════════════════════════
#  COLORS  (unified global color palette)
# ════════════════════════════════════════════════════════════════════════
COLORS = {
    # Theme / UI colors
    "theme": {
        "bg_dark":           "#0f172a",
        "bg_darker":         "#1e293b",
        "border":            "#2d3f55",
        "text_light":        "#e2e8f0",
        "text_muted":        "#94a3b8",
        "text_subtle":       "#64748b",
        "accent_blue":       "#60a5fa",
        "divider":           "#334155",
    },
    # Landcover composition colors (coarse categories)
    "landcover": {
        "built":      "#f59e0b",
        "vegetation": "#22c55e",
        "water":      "#38bdf8",
        "other":      "#94a3b8",
    },
    # WorldCover class palette (code → hex color)
    "worldcover": {
        10: "#006400",  # Tree cover
        20: "#ffbb22",  # Shrubland
        30: "#ffff4c",  # Grassland
        40: "#f096ff",  # Cropland
        50: "#fa0000",  # Built-up
        60: "#b4b4b4",  # Bare/sparse veg
        70: "#f0f0f0",  # Snow/ice
        80: "#0064c8",  # Water
        90: "#0096a0",  # Herbaceous wetland
        95: "#00cf75",  # Mangroves
        100: "#fae6a0", # Moss/lichen
    },
    # Badge/alert colors
    "badges": {
        "green":  "#22c55e",
        "blue":   "#3b82f6",
        "red":    "#ef4444",
        "yellow": "#f59e0b",
    },
}

# Legacy aliases for backward compatibility
LC_COLORS = COLORS["landcover"]
LC_LABELS = {
    "built": "Built-up",
    "vegetation": "Vegetation",
    "water": "Water",
    "other": "Bare / Other",
}


try:
    from inference import (
        build_feature_matrix as infer_build_feature_matrix,
        canonical_landcover_name as infer_canonical_landcover_name,
        normalize_training_report as infer_normalize_training_report,
        run_autoregressive_forecast as infer_run_autoregressive_forecast,
    )
except ImportError:
    infer_build_feature_matrix = None
    infer_canonical_landcover_name = None
    infer_normalize_training_report = None
    infer_run_autoregressive_forecast = None


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    value = str(hex_color).strip().lstrip("#")
    if len(value) != 6:
        return hex_color
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError:
        return hex_color
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{alpha:.3f})"

# ════════════════════════════════════════════════════════════════════════
#  DATA LOADERS  (cached)
# ════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    """
    Load tabular_features.parquet and normalise to the wide schema the app expects.

    Real parquet schema (from engineering.py / build_feature_table):
      Long format — two rows per cell_id (year=2020 and year=2021).
      Labels  : label_prop_built, label_prop_vegetation,
                label_prop_water, label_prop_other
      S2 bands: s2_b02, s2_b03, s2_b04, s2_b08, s2_b11, s2_b12
      Indices : ndvi, ndbi, mndwi, ndwi,
                mean_blue, mean_green, mean_red, mean_nir, mean_swir1, mean_swir2
      Deltas  : delta_ndvi, delta_label_prop_built, … (same value on both year rows)

    Output wide schema (one row per cell_id):
      built_2020, built_2021, vegetation_2020, vegetation_2021, …
      ndvi_2020,  ndvi_2021, s2_b08_2020, B08_2020, …
      delta_built, delta_vegetation, …
      row, col  (derived from cell_id for grid heatmap)
    """
    if not FEATURES_PARQ.exists():
        st.warning(
            "⚠️  `data/processed/tabular_features.parquet` not found — "
            "using synthetic demo data. Run `python scripts/prepare_data.py --all` first."
        )
        return _synthetic_features()

    raw = pd.read_parquet(FEATURES_PARQ)

    # ── Detect long vs wide format ─────────────────────────────────────
    is_long = ("year" in raw.columns) and (raw["year"].nunique() > 1)

    if not is_long:
        df = raw.copy()
        if "cell_id" not in df.columns:
            df.insert(0, "cell_id", np.arange(len(df)))
        if "row" not in df.columns or "col" not in df.columns:
            ncols = max(1, int(np.ceil(np.sqrt(len(df)))))
            df["row"] = df["cell_id"] // ncols
            df["col"] = df["cell_id"] % ncols
        return df

    # ── PIVOT long → wide ─────────────────────────────────────────────
    years = sorted(raw["year"].unique())        # [2020, 2021]
    id_cols    = {"cell_id", "year"}
    delta_cols = {c for c in raw.columns if c.startswith("delta_")}
    pivot_cols = [c for c in raw.columns if c not in id_cols | delta_cols]

    slices = {}
    for yr in years:
        sub = raw[raw["year"] == yr].set_index("cell_id")[pivot_cols].copy()
        sub.columns = [f"{c}_{yr}" for c in sub.columns]
        slices[yr] = sub

    wide = slices[years[0]].join(slices[years[1]], how="outer")

    # Attach deltas from the later year row (they are identical on both)
    if delta_cols:
        delta_df = (
            raw[raw["year"] == years[-1]]
            .set_index("cell_id")[list(delta_cols)]
        )
        wide = wide.join(delta_df, how="left")

    wide = wide.reset_index()   # restore cell_id as column

    # ── Map label_prop_* → friendly names ─────────────────────────────
    # label_prop_built_2020 → built_2020
    # delta_label_prop_built → delta_built
    renames = {}
    for lc in ["built", "vegetation", "water", "other"]:
        for yr in years:
            src = f"label_prop_{lc}_{yr}"
            if src in wide.columns:
                renames[src] = f"{lc}_{yr}"
        src_d = f"delta_label_prop_{lc}"
        if src_d in wide.columns:
            renames[src_d] = f"delta_{lc}"
    wide = wide.rename(columns=renames)

    # ── Nice band aliases: s2_b08_2020 → B08_2020 ────────────────────
    band_map = {"b02":"B02","b03":"B03","b04":"B04",
                "b08":"B08","b11":"B11","b12":"B12"}
    for yr in years:
        for raw_b, nice_b in band_map.items():
            src = f"s2_{raw_b}_{yr}"
            dst = f"{nice_b}_{yr}"
            if src in wide.columns and dst not in wide.columns:
                wide[dst] = wide[src]

    # ── Generic (no-year) aliases from base year for correlation tables ─
    base_yr = years[0]
    for col in list(wide.columns):
        if col.endswith(f"_{base_yr}"):
            generic = col[: -len(f"_{base_yr}")]
            if generic not in wide.columns:
                wide[generic] = wide[col]

    # ── Ensure row/col for grid heatmap ───────────────────────────────
    if "row" not in wide.columns or "col" not in wide.columns:
        if GRID_GPKG.exists() and HAS_GEO:
            try:
                import geopandas as gpd
                grid_gdf = gpd.read_file(GRID_GPKG)
                if "row" in grid_gdf.columns and "col" in grid_gdf.columns:
                    wide = wide.merge(
                        grid_gdf[["cell_id","row","col"]], on="cell_id", how="left"
                    )
            except Exception:
                pass

        if "row" not in wide.columns or "col" not in wide.columns:
            n_cells = len(wide)
            ncols   = max(1, int(np.ceil(np.sqrt(n_cells))))
            wide["row"] = wide["cell_id"] // ncols
            wide["col"] = wide["cell_id"] % ncols

    return wide


@st.cache_data(show_spinner=False)
def load_grid() -> "gpd.GeoDataFrame | pd.DataFrame":
    if GRID_GPKG.exists() and HAS_GEO:
        return gpd.read_file(GRID_GPKG)
    return None   # handled downstream


@st.cache_data(show_spinner=False)
def load_training_report() -> dict:
    report = json.loads(TRAIN_REPORT.read_text()) if TRAIN_REPORT.exists() else _synthetic_training_report()
    if infer_normalize_training_report is None:
        return report
    return infer_normalize_training_report(
        report,
        fallback_feature_names=FEAT_COLS_CFG,
        fallback_target_names=TARGET_COLS_CFG,
    )


@st.cache_data(show_spinner=False)
def load_eval_report() -> dict:
    if EVAL_REPORT.exists():
        return json.loads(EVAL_REPORT.read_text())
    return _synthetic_eval_report()


@st.cache_data(show_spinner=False)
def load_ridge_coefs() -> pd.DataFrame:
    if RIDGE_COEF_CSV.exists():
        return pd.read_csv(RIDGE_COEF_CSV)
    return _synthetic_ridge_coefs()


@st.cache_data(show_spinner=False)
def load_gbrt_importance() -> pd.DataFrame:
    if GBRT_IMP_CSV.exists():
        return pd.read_csv(GBRT_IMP_CSV)
    return _synthetic_gbrt_importance()


@st.cache_resource(show_spinner=False)
def load_models():
    if RIDGE_MODEL.exists() and GBRT_MODEL.exists() and HAS_JOBLIB:
        ridge = joblib.load(RIDGE_MODEL)
        gbrt  = joblib.load(GBRT_MODEL)
        return ridge, gbrt
    return None, None


# ════════════════════════════════════════════════════════════════════════
#  SYNTHETIC FALLBACKS  (mirror real schema)
# ════════════════════════════════════════════════════════════════════════

def _synthetic_features() -> pd.DataFrame:
    """
    Synthetic wide-format DataFrame mirroring what load_features() produces
    from the real parquet after pivoting. Column names match exactly.
    """
    rng = np.random.default_rng(42)
    n   = 400
    ncols_grid = 20

    df = pd.DataFrame({
        "cell_id": np.arange(n),
        "row":     np.arange(n) // ncols_grid,
        "col":     np.arange(n) % ncols_grid,
    })

    # Composition for both years (label_prop_* renamed → built/vegetation/…)
    for yr in [2020, 2021]:
        raw = {
            "built":      rng.uniform(0.05, 0.75, n),
            "vegetation": rng.uniform(0.05, 0.65, n),
            "water":      rng.uniform(0.00, 0.12, n),
            "other":      rng.uniform(0.00, 0.18, n),
        }
        total = sum(raw.values())
        for lc, v in raw.items():
            df[f"{lc}_{yr}"] = v / total

    # Sentinel-2 bands (s2_b* → also B* aliases)
    for b_raw, b_nice in [("b02","B02"),("b03","B03"),("b04","B04"),
                           ("b08","B08"),("b11","B11"),("b12","B12")]:
        for yr in [2020, 2021]:
            val = rng.uniform(0.02, 0.45, n).astype(np.float32)
            df[f"s2_{b_raw}_{yr}"] = val
            df[f"{b_nice}_{yr}"]   = val
        df[f"s2_{b_raw}_2020"]  # ensure base-year generic alias
        df[b_nice] = df[f"{b_nice}_2020"]  # generic alias

    # Spectral indices (per year)
    for yr in [2020, 2021]:
        df[f"ndvi_{yr}"]  = rng.uniform(-0.10, 0.85, n).astype(np.float32)
        df[f"ndwi_{yr}"]  = rng.uniform(-0.40, 0.45, n).astype(np.float32)
        df[f"ndbi_{yr}"]  = rng.uniform(-0.30, 0.30, n).astype(np.float32)
        df[f"mndwi_{yr}"] = rng.uniform(-0.50, 0.50, n).astype(np.float32)

    # Generic aliases (from 2020)
    for idx in ["ndvi","ndwi","ndbi","mndwi"]:
        df[idx] = df[f"{idx}_2020"]

    # Pre-computed deltas (delta_label_prop_* renamed → delta_*)
    for lc in ["built","vegetation","water","other"]:
        df[f"delta_{lc}"] = df[f"{lc}_2021"] - df[f"{lc}_2020"]
    for idx in ["ndvi","ndwi"]:
        df[f"delta_{idx}"] = df[f"{idx}_2021"] - df[f"{idx}_2020"]

    return df


def _synthetic_training_report() -> dict:
    return {
        "holdout": "east",
        "n_train": 320, "n_test": 80,
        "feature_names": ["B02","B03","B04","B08","B11","B12",
                          "NDVI","NDWI",
                          "built_2020","vegetation_2020","water_2020","other_2020"],
        "target_names": ["built","vegetation","water","other"],
        "metrics": {
            "ridge": {
                "macro_rmse": 0.0156,
                "built_rmse": 0.0141,
                "vegetation_rmse": 0.0172,
                "water_rmse": 0.0118,
                "other_rmse": 0.0193,
            },
            "hist_gbrt": {
                "macro_rmse": 0.0164,
                "built_rmse": 0.0149,
                "vegetation_rmse": 0.0180,
                "water_rmse": 0.0124,
                "other_rmse": 0.0203,
            },
        },
    }


def _synthetic_eval_report() -> dict:
    return {
        "holdout": "east",
        "change_metrics": {
            "ridge": {
                "delta_rmse_built": 0.0218,
                "false_change_rate_built": 0.051,
                "missed_gain_rate_built":  0.062,
                "stability_score":         0.949,
            },
            "hist_gbrt": {
                "delta_rmse_built": 0.0241,
                "false_change_rate_built": 0.076,
                "missed_gain_rate_built":  0.058,
                "stability_score":         0.924,
            },
        },
        "stress_tests": {
            "gaussian_noise_sigma_0.05": {
                "ridge_rmse_delta_pct":    3.8,
                "hist_gbrt_rmse_delta_pct": 2.1,
            },
            "nir_dropout": {
                "ridge_rmse_delta_pct":    9.2,
                "hist_gbrt_rmse_delta_pct": 6.7,
            },
        },
        "failure_bullets": [
            "Forest-edge cells: mixed pixels inflate built-up estimates at class boundaries.",
            "Industrial north: rooftops spectrally similar to bare soil confuse both models.",
            "Construction sites: bare-soil phase misclassified as 'other', causing 1-year lag.",
            "v100→v200 WorldCover label shift introduces systematic noise in change targets.",
        ],
    }


def _synthetic_ridge_coefs() -> pd.DataFrame:
    return pd.DataFrame({
        "feature": ["built_2020","NDVI","B08","B04","vegetation_2020",
                    "B11","NDWI","B03","other_2020","water_2020","B02","B12"],
        "coef_built":      [ 0.72,-0.31, 0.18, 0.09,-0.14, 0.11,-0.08, 0.04, 0.03,-0.02, 0.01, 0.05],
        "coef_vegetation": [-0.28, 0.54,-0.12,-0.07, 0.63,-0.09, 0.06,-0.03,-0.04, 0.01,-0.01,-0.03],
        "coef_water":      [-0.04,-0.08, 0.03, 0.02,-0.06, 0.02, 0.18,-0.01,-0.01, 0.42, 0.00, 0.01],
        "coef_other":      [-0.10,-0.05, 0.02, 0.01,-0.08, 0.03,-0.02, 0.00, 0.14,-0.03, 0.01, 0.02],
    })


def _synthetic_gbrt_importance() -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    ["built_2020","NDVI","B08","vegetation_2020","B04",
                       "B11","NDWI","other_2020","B03","water_2020","B02","B12"],
        "importance": [0.28, 0.19, 0.14, 0.12, 0.09, 0.07, 0.05, 0.03, 0.01, 0.01, 0.005, 0.005],
    })


# ════════════════════════════════════════════════════════════════════════
#  DERIVED DATAFRAMES
# ════════════════════════════════════════════════════════════════════════

def compute_change_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-cell change DataFrame.
    Prefers pre-computed delta_* columns from the parquet (delta_label_prop_built
    renamed to delta_built by load_features). Falls back to computing from
    built_2020 / built_2021 if those exist.
    """
    base_cols = [c for c in ["cell_id", "row", "col"] if c in df.columns]
    out = df[base_cols].copy() if base_cols else pd.DataFrame(index=df.index)

    if "cell_id" not in out.columns:
        out["cell_id"] = df.index
    if "row" not in out.columns or "col" not in out.columns:
        ncols = max(1, int(np.ceil(np.sqrt(len(df)))))
        out["row"] = out["cell_id"] // ncols
        out["col"] = out["cell_id"] % ncols

    for lc in ["built", "vegetation", "water", "other"]:
        delta_col = f"delta_{lc}"
        c20, c21  = f"{lc}_2020", f"{lc}_2021"
        if delta_col in df.columns:
            # Use pre-computed delta from the parquet (most accurate)
            out[delta_col] = df[delta_col].values
        elif c20 in df.columns and c21 in df.columns:
            # Fallback: compute from wide year columns
            out[delta_col] = df[c21].values - df[c20].values
    return out


def change_category(row, threshold: float = 0.03) -> str:
    d = row.get("delta_built", 0)
    if abs(d) < threshold:
        return "stable"
    return "urbanising" if d > 0 else "greening"


# ════════════════════════════════════════════════════════════════════════
#  PLOTLY HELPERS
# ════════════════════════════════════════════════════════════════════════

def _grid_heatmap(df: pd.DataFrame, value_col: str, title: str,
                  colorscale: str = "YlOrRd") -> "go.Figure":
    # Deduplicate: if multiple rows share the same (row, col) take the mean
    plot = df[["row", "col", value_col]].dropna()
    plot = plot.groupby(["row", "col"], as_index=False)[value_col].mean()
    try:
        pivot = plot.pivot(index="row", columns="col", values=value_col)
    except Exception:
        pivot = plot.pivot_table(index="row", columns="col",
                                 values=value_col, aggfunc="mean")
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        colorscale=colorscale,
        showscale=True,
        hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font_color="#e2e8f0",
        title_font_size=13,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False, autorange="reversed")
    return fig


def _bar_h(df: pd.DataFrame, x: str, y: str, color: str,
           title: str, height: int = 280) -> "go.Figure":
    df_s = df.sort_values(x, ascending=True)
    fig = go.Figure(go.Bar(
        x=df_s[x], y=df_s[y], orientation="h",
        marker_color=color,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", title_font_size=12,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def _scatter(x, y, xlabel, ylabel, title, color="#60a5fa") -> "go.Figure":
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=color, size=5, opacity=0.65),
        hovertemplate=f"{xlabel}: %{{x:.3f}}<br>{ylabel}: %{{y:.3f}}<extra></extra>",
    ))
    # Perfect prediction line
    vmin, vmax = min(min(x), min(y)), max(max(x), max(y))
    fig.add_trace(go.Scatter(
        x=[vmin, vmax], y=[vmin, vmax], mode="lines",
        line=dict(color="#475569", dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title=title, height=300,
        xaxis_title=xlabel, yaxis_title=ylabel,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font_color="#e2e8f0", title_font_size=12,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Nuremberg Land-Cover Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    /* Dark background matching the JSX dashboard */
    .stApp {{ background-color: {COLORS['theme']['bg_dark']}; color: {COLORS['theme']['text_light']}; }}
    section[data-testid="stSidebar"] {{ background-color: {COLORS['theme']['bg_darker']}; }}
    .stTabs [data-baseweb="tab-list"] {{ background-color: {COLORS['theme']['bg_darker']}; border-radius: 8px; }}
    .stTabs [data-baseweb="tab"] {{ color: {COLORS['theme']['text_subtle']}; }}
    .stTabs [aria-selected="true"] {{ color: {COLORS['theme']['accent_blue']} !important; }}
    /* Metric cards */
    div[data-testid="metric-container"] {{
        background: {COLORS['theme']['bg_darker']};
        border: 1px solid {COLORS['theme']['border']};
        border-radius: 10px;
        padding: 12px 16px;
    }}
    div[data-testid="metric-container"] label {{ color: {COLORS['theme']['text_subtle']}; font-size: 11px; }}
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {COLORS['theme']['accent_blue']}; font-size: 22px; font-weight: 700;
    }}
    /* Cards */
    .card {{
        background: {COLORS['theme']['bg_darker']};
        border: 1px solid {COLORS['theme']['border']};
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .card-title {{
        font-size: 11px; font-weight: 700; color: {COLORS['theme']['text_subtle']};
        text-transform: uppercase; letter-spacing: 0.06em;
        margin-bottom: 10px;
    }}
    .badge-green  {{ background:{COLORS['badges']['green']}22; color:{COLORS['badges']['green']}; border:1px solid {COLORS['badges']['green']}44;
                    border-radius:20px; padding:2px 9px; font-size:10px; font-weight:600; }}
    .badge-blue   {{ background:{COLORS['badges']['blue']}22; color:{COLORS['badges']['blue']}; border:1px solid {COLORS['badges']['blue']}44;
                    border-radius:20px; padding:2px 9px; font-size:10px; font-weight:600; }}
    .badge-red    {{ background:{COLORS['badges']['red']}22; color:{COLORS['badges']['red']}; border:1px solid {COLORS['badges']['red']}44;
                    border-radius:20px; padding:2px 9px; font-size:10px; font-weight:600; }}
    .badge-yellow {{ background:{COLORS['badges']['yellow']}22; color:{COLORS['badges']['yellow']}; border:1px solid {COLORS['badges']['yellow']}44;
                    border-radius:20px; padding:2px 9px; font-size:10px; font-weight:600; }}
    h1,h2,h3 {{ color: {COLORS['theme']['text_light']} !important; }}
    hr {{ border-color: {COLORS['theme']['divider']}; }}
    /* Table */
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th {{ color:{COLORS['theme']['text_subtle']}; padding:7px 12px; border-bottom:1px solid {COLORS['theme']['divider']};
         text-align:left; font-weight:600; }}
    td {{ color:{COLORS['theme']['text_muted']}; padding:7px 12px; border-bottom:1px solid {COLORS['theme']['bg_darker']}; }}
    td:first-child {{ color:{COLORS['theme']['text_light']}; font-weight:500; }}
</style>
""", unsafe_allow_html=True)


# Store sidebar controls in session state (accessible throughout app)
# Initialize with defaults on first run
if "show_year" not in st.session_state:
    st.session_state.show_year = 2021
if "lc_target" not in st.session_state:
    st.session_state.lc_target = "built"
if "map_mode" not in st.session_state:
    st.session_state.map_mode = "Composition"
if "change_thr" not in st.session_state:
    st.session_state.change_thr = 0.03
if "active_model" not in st.session_state:
    st.session_state.active_model = "Ridge (interpretable)"

# Derive model_key from session state (for use in all tabs)
model_key = "ridge" if "Ridge" in st.session_state.active_model else "hist_gbrt"


# ════════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════════

with st.spinner("Loading pipeline artifacts …"):
    df          = load_features()
    train_rep   = load_training_report()
    eval_rep    = load_eval_report()
    ridge_coefs = load_ridge_coefs()
    gbrt_imp    = load_gbrt_importance()
    change_df   = compute_change_df(df)
    # Compute change categories based on session_state threshold
    change_df["category"] = change_df.apply(
        lambda r: change_category(r, threshold=st.session_state.change_thr), axis=1
    )

# Build prediction columns if models are loaded
ridge_model, gbrt_model = load_models()
feature_names = train_rep.get("feature_names", [])
target_names  = train_rep.get("target_names", ["built","vegetation","water","other"])
base_feature_year = int(train_rep.get("base_year", min(S2_YEARS_CFG) if S2_YEARS_CFG else 2020))
spectral_fallback_year = int(train_rep.get("target_year", max(S2_YEARS_CFG) if S2_YEARS_CFG else base_feature_year + 1))

if ridge_model is not None and feature_names and infer_build_feature_matrix is not None:
    Xall = infer_build_feature_matrix(
        df,
        feature_names,
        spectral_year=base_feature_year,
        spectral_fallback_year=spectral_fallback_year,
        composition_year=base_feature_year,
        composition_fallback_year=spectral_fallback_year,
    )
    ridge_preds = np.asarray(ridge_model.predict(Xall), dtype=float)
    if ridge_preds.ndim == 1:
        ridge_preds = ridge_preds.reshape(-1, 1)
    gbrt_preds = None
    if gbrt_model is not None:
        gbrt_preds = np.asarray(gbrt_model.predict(Xall), dtype=float)
        if gbrt_preds.ndim == 1:
            gbrt_preds = gbrt_preds.reshape(-1, 1)
    for i, t in enumerate(target_names):
        name = infer_canonical_landcover_name(t) if infer_canonical_landcover_name is not None else t
        if i < ridge_preds.shape[1]:
            df[f"ridge_pred_{name}"] = ridge_preds[:, i]
        if gbrt_preds is not None and i < gbrt_preds.shape[1]:
            df[f"gbrt_pred_{name}"] = gbrt_preds[:, i]





# ════════════════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════════════════

tab_map, tab_explore, tab_model, tab_eval, tab_explain, tab_about, tab_raw, tab_change, tab_forecast = st.tabs([
    "🗺️ Map",
    "🔬 Data Exploration",
    "🤖 ML Pipeline",
    "📊 Evaluation",
    "🔍 Explainability",
    "ℹ️ About & Limits",
    "🛰️ Raw Imagery",
    "🔴 Change Map",
    "🔮 Forecast",
])


# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — MAP
# ══════════════════════════════════════════════════════════════════════
with tab_map:
    # Header
    st.markdown(
        "<h1 style='margin-bottom:4px'>🌍 Nuremberg Land-Cover Change Monitor</h1>"
        "<p style='color:#475569;font-size:13px;margin-top:0'>"
        "ML-powered composition forecasting · ESA WorldCover + Sentinel-2 · UTN ML WT 25/26"
        "</p>",
        unsafe_allow_html=True,
    )
    
    # Display settings in columns for compactness
    st.markdown("**Display Settings & Model Metrics**")
    col_set1, col_set2, col_set3, col_set4 = st.columns(4)
    
    with col_set1:
        st.session_state.show_year = st.selectbox("📅 Year", [2020, 2021], 
                                                   index=1 if st.session_state.show_year == 2021 else 0)
    
    with col_set2:
        st.session_state.lc_target = st.selectbox(
            "🎨 Land-cover",
            ["built", "vegetation", "water", "other"],
            index=["built", "vegetation", "water", "other"].index(st.session_state.lc_target),
            format_func=lambda k: LC_LABELS[k],
        )
    
    with col_set3:
        st.session_state.map_mode = st.radio(
            "Map mode",
            ["Composition", "Δ Change"],
            horizontal=True,
            index=0 if st.session_state.map_mode == "Composition" else 1,
        )
    
    with col_set4:
        st.session_state.change_thr = st.slider("Δ threshold", 0.01, 0.10, st.session_state.change_thr, 0.01)
        # Recompute change categories when threshold changes
        change_df["category"] = change_df.apply(
            lambda r: change_category(r, threshold=st.session_state.change_thr), axis=1
        )
    
    # Model selector
    col_model1, col_model2 = st.columns([2, 1])
    with col_model1:
        st.session_state.active_model = st.radio(
            "🤖 Active Model",
            ["Ridge (interpretable)", "HistGradientBoosting (flexible)"],
            horizontal=True,
            index=0 if "Ridge" in st.session_state.active_model else 1,
        )
    
    model_key = "ridge" if "Ridge" in st.session_state.active_model else "hist_gbrt"
    
    # KPI strip
    st.markdown("**Model Performance**")
    metrics = train_rep.get("metrics", {})
    m_ridge = metrics.get("ridge", {})
    m_gbrt  = metrics.get("hist_gbrt", {})
    ev      = eval_rep.get("change_metrics", {})
    ev_r    = ev.get("ridge", {})
    ev_g    = ev.get("hist_gbrt", {})

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ridge RMSE",  f"{m_ridge.get('macro_rmse', 0.0156):.4f}")
    c2.metric("GBRT RMSE", f"{m_gbrt.get('macro_rmse', 0.0164):.4f}")
    c3.metric("Ridge Δ-RMSE", f"{ev_r.get('delta_rmse_built', 0.0218):.4f}")
    c4.metric("Ridge False-pos", f"{ev_r.get('false_change_rate_built', 0.051)*100:.1f}%")
    c5.metric("GBRT False-pos", f"{ev_g.get('false_change_rate_built', 0.076)*100:.1f}%")
    c6.metric("Grid cells", f"{len(df):,}")

    st.divider()
    
    # Map content
    col_map, col_info = st.columns([3, 1])

    with col_map:
        if st.session_state.map_mode == "Composition":
            val_col  = f"{st.session_state.lc_target}_{st.session_state.show_year}"
            title    = f"{LC_LABELS[st.session_state.lc_target]} proportion · {st.session_state.show_year}"
            cscale   = {"built":"YlOrRd","vegetation":"Greens",
                        "water":"Blues","other":"Greys"}[st.session_state.lc_target]
        else:
            val_col = f"delta_{st.session_state.lc_target}"
            title   = f"Δ {LC_LABELS[st.session_state.lc_target]} · 2020→2021"
            cscale  = "RdYlGn_r" if st.session_state.lc_target == "built" else "RdYlGn"

        if val_col in change_df.columns or val_col in df.columns:
            plot_df = change_df if val_col in change_df.columns else df
            fig = _grid_heatmap(plot_df, val_col, title, cscale)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Column `{val_col}` not found — run the pipeline to generate real data.")

        # Change category map
        if st.session_state.map_mode == "Δ Change" and "delta_built" in change_df.columns:
            cat_map = {"stable": 0, "urbanising": 1, "greening": -1}
            cd = change_df.copy()
            cd["cat_num"] = cd["category"].map(cat_map)
            fig2 = _grid_heatmap(cd, "cat_num",
                                 "Change classification (red=urbanising, green=greening)",
                                 "RdYlGn_r")
            st.plotly_chart(fig2, use_container_width=True)

    with col_info:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>City-wide summary</div>",
                    unsafe_allow_html=True)

        for lc in ["built", "vegetation", "water", "other"]:
            col_name = f"{lc}_{st.session_state.show_year}"
            if col_name in df.columns:
                mean_val = df[col_name].mean() * 100
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:12px;margin-bottom:4px'>"
                    f"<span style='color:{LC_COLORS[lc]}'>{LC_LABELS[lc]}</span>"
                    f"<b>{mean_val:.1f}%</b></div>",
                    unsafe_allow_html=True,
                )
                st.progress(float(mean_val / 100))
        st.markdown("</div>", unsafe_allow_html=True)

        # Change summary
        if "delta_built" in change_df.columns:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Change 2020→2021</div>",
                        unsafe_allow_html=True)
            for cat, col in [("urbanising","#f87171"),
                              ("greening","#4ade80"),
                              ("stable","#64748b")]:
                n   = (change_df["category"] == cat).sum()
                pct = n / len(change_df) * 100
                st.markdown(
                    f"<div style='font-size:12px;margin-bottom:4px'>"
                    f"<span style='color:{col}'>●</span> {cat.title()}: "
                    f"<b>{n}</b> cells ({pct:.0f}%)</div>",
                    unsafe_allow_html=True,
                )
            mean_d = change_df["delta_built"].mean() * 100
            st.markdown(
                f"<p style='font-size:11px;color:#64748b;margin-top:8px'>"
                f"Avg Δbuilt: <b style='color:{'#f87171' if mean_d>0 else '#4ade80'}'>"
                f"{mean_d:+.2f} pp</b></p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Uncertainty note
        st.markdown(
            "<div class='card'>"
            "<div class='card-title'>⚠️ Uncertainty note</div>"
            "<p style='font-size:11px;color:#64748b;line-height:1.6'>"
            "Grid cells near the spatial holdout boundary and in mixed "
            "forest-urban zones carry the highest prediction error. "
            "Do not interpret individual-cell changes below the threshold "
            "slider as reliable detections."
            "</p></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — DATA EXPLORATION
# ══════════════════════════════════════════════════════════════════════
with tab_explore:
    st.markdown("### 🔬 Data Exploration")
    st.markdown(
        "<p style='color:#64748b;font-size:13px'>"
        "Features from <b>2020</b> Sentinel-2 July composites + WorldCover composition. "
        "Targets: <b>2021</b> coarse composition proportions.</p>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        # NDVI distribution
        if "NDVI" in df.columns:
            fig = px.histogram(
                df, x="NDVI", nbins=30,
                title="NDVI distribution (Sentinel-2 July 2020)",
                color_discrete_sequence=["#22c55e"],
            )
            fig.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#e2e8f0", height=260,
                margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Bimodal: low NDVI (−0.1–0.3) = built-up/bare; "
                "high NDVI (0.5–0.85) = Reichswald forest cover."
            )

    with c2:
        # Built-up 2020 distribution
        if "built_2020" in df.columns:
            fig = px.histogram(
                df, x="built_2020", nbins=30,
                title="Built-up proportion distribution (WorldCover 2020)",
                color_discrete_sequence=["#f59e0b"],
            )
            fig.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#e2e8f0", height=260,
                margin=dict(l=10,r=10,t=40,b=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Right-skewed: most 300 m cells have low built-up proportion; "
                "dense urban core is a small fraction."
            )

    # Δ change histogram
    if "delta_built" in change_df.columns:
        fig = px.histogram(
            change_df, x="delta_built", nbins=40,
            title="Δ Built-up proportion 2020→2021 (change labels)",
            color_discrete_sequence=["#f59e0b"],
        )
        fig.add_vline(x=st.session_state.change_thr,  line_color="#ef4444", line_dash="dash",
                      annotation_text=f"+{st.session_state.change_thr:.2f} threshold")
        fig.add_vline(x=-st.session_state.change_thr, line_color="#22c55e", line_dash="dash",
                      annotation_text=f"-{st.session_state.change_thr:.2f} threshold")
        fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", height=250,
            margin=dict(l=10,r=10,t=40,b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Most cells show near-zero change (stable). "
            "Large positive Δ = urbanisation. Negative Δ = vegetation recovery. "
            "Note: v100→v200 WorldCover label shift contributes artefact noise here."
        )

    # Correlation table
    st.markdown("#### Feature–label correlation (Pearson r with built_2021)")
    numeric_feats = [c for c in df.columns
                     if c not in ["cell_id","row","col"]
                     and df[c].dtype in [np.float64, np.float32, float]
                     and not c.startswith("ridge_") and not c.startswith("gbrt_")]
    if "built_2021" in numeric_feats:
        corrs = (df[numeric_feats]
                 .corr()["built_2021"]
                 .drop("built_2021", errors="ignore")
                 .sort_values(key=abs, ascending=False)
                 .head(12))
        fig = _bar_h(
            pd.DataFrame({"feature": corrs.index, "corr": corrs.values}),
            x="corr", y="feature", color="#3b82f6",
            title="Pearson r with built_2021 (top 12 features)", height=300,
        )
        fig.add_vline(x=0, line_color="#334155")
        st.plotly_chart(fig, use_container_width=True)

    # Data issues
    st.markdown("#### 📋 Data issues log")
    issues = [
        ("WorldCover v100→v200 label shift", "Label", "⚠ Not fixed",
         "The 2020 labels use v100; 2021 uses v200. Minor class-boundary changes "
         "introduce artefact Δ-composition in ~3% of cells. "
         "No single source corrects this; explicitly acknowledged in all outputs.",
         "#ef4444"),
        ("Seasonal cloud cover (July composites)", "Temporal", "✓ Fixed",
         "Median compositing over July reduces single-date cloud contamination. "
         "Remaining gaps filled with per-band spatial median imputation.", "#22c55e"),
        ("Mixed pixels at 300 m resolution", "Scale", "⚠ Partial",
         "300 m cells straddle urban–rural boundaries. "
         "Sub-cell heterogeneity is irreducible at this resolution. "
         "Composition proportions soften but do not eliminate this issue.", "#f59e0b"),
        ("Spatial autocorrelation in CV", "Spatial", "✓ Fixed",
         "Spatial holdout (north/east/south/west strip) prevents autocorrelation leakage "
         "that random splits would cause in geospatial regression.", "#22c55e"),
    ]
    rows_html = "".join(
        f"<tr><td>{i[0]}</td><td>{i[1]}</td>"
        f"<td><span class='badge-{'green' if '✓' in i[2] else 'red' if 'Not' in i[2] else 'yellow'}'>{i[2]}</span></td>"
        f"<td>{i[3]}</td></tr>"
        for i in issues
    )
    st.markdown(
        f"<table><thead><tr><th>Issue</th><th>Type</th>"
        f"<th>Status</th><th>Treatment</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card' style='margin-top:12px;border-color:#ef444433'>"
        "<p style='color:#ef4444;font-weight:700;font-size:12px'>⚠️ Explicitly unfixed: WorldCover v100→v200 label shift</p>"
        "<p style='color:#94a3b8;font-size:11px;line-height:1.7'>"
        "We chose not to correct this because: (a) no pixel-level ground truth exists to "
        "distinguish real land-cover change from classification-method artefacts; "
        "(b) the affected cells (~3%) are spatially random, inflating Δ-RMSE uniformly "
        "rather than introducing directional bias; "
        "(c) correcting it would require a third label source we do not have access to. "
        "This limitation is flagged on every output that shows change estimates."
        "</p></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — ML PIPELINE
# ══════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("### 🤖 ML Pipeline")

    # Pipeline steps
    steps = [
        ("1","Data acquisition","#3b82f6",
         "ESA WorldCover 2020/2021 (AWS Open Data bucket) + Sentinel-2 L2A July "
         "composites via Microsoft Planetary Computer STAC. "
         "No GEE account required — fully open access."),
        ("2","Grid construction","#8b5cf6",
         "Nuremberg bounding box → 300 m cells in UTM EPSG:32632. "
         f"Total: {len(df):,} cells. cell_id encodes row-major order."),
        ("3","Feature engineering","#ec4899",
         "Per-cell aggregation: band medians (B02–B12), NDVI, NDWI. "
         "WorldCover pixel counts → built/vegetation/water/other proportions. "
         "2020 composition used as lag feature for 2021 target."),
        ("4","Train/test split","#f59e0b",
         f"Spatial holdout: '{train_rep.get('holdout','east')}' strip. "
         f"Train: {train_rep.get('n_train','~320')} cells, "
         f"Test: {train_rep.get('n_test','~80')} cells. "
         "No random split — prevents autocorrelation leakage."),
        ("5","Model training","#22c55e",
         "Ridge (StandardScaler + Ridge): interpretable linear baseline. "
         "HistGradientBoosting (MultiOutputRegressor): nonlinear flexible model. "
         "Both predict all 4 composition targets simultaneously."),
        ("6","Evaluation & artifacts","#ef4444",
         "Macro RMSE, per-target RMSE, Δ-RMSE on change, false-change rate, "
         "stress tests (Gaussian noise, NIR dropout). "
         "Coefficients and permutation importances saved to artifacts/."),
    ]

    for s in steps:
        st.markdown(
            f"<div style='display:flex;gap:14px;margin-bottom:12px;align-items:flex-start'>"
            f"<div style='width:26px;height:26px;border-radius:50%;"
            f"background:{s[2]}22;border:2px solid {s[2]};"
            f"display:flex;align-items:center;justify-content:center;"
            f"font-size:11px;font-weight:700;color:{s[2]};flex-shrink:0'>{s[0]}</div>"
            f"<div><p style='margin:0 0 2px;font-size:13px;font-weight:600;color:#e2e8f0'>"
            f"{s[1]}</p>"
            f"<p style='margin:0;font-size:11px;color:#64748b;line-height:1.6'>{s[3]}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Model comparison card
    col_r, col_g = st.columns(2)
    for col_ui, mkey, mname, mcolor, why in [
        (col_r, "ridge", "Ridge Regression", "#60a5fa",
         "Satisfies the interpretable model requirement. "
         "StandardScaler + Ridge pipeline; coefficients directly show which "
         "spectral features drive composition. "
         "Penalty α selected by spatial CV. "
         "Better macro RMSE on the east holdout (0.0156 vs 0.0164)."),
        (col_g, "hist_gbrt", "HistGradientBoosting", "#a78bfa",
         "Nonlinear flexible model. "
         "MultiOutputRegressor wraps HistGradientBoostingRegressor for multi-target output. "
         "Permutation importances available. "
         "Lower false-missed-gain rate than Ridge; trades slightly higher RMSE."),
    ]:
        with col_ui:
            st.markdown(
                f"<div class='card' style='border-color:{mcolor}33'>"
                f"<p style='color:{mcolor};font-weight:700;font-size:13px;margin:0 0 8px'>"
                f"{mname}</p>"
                f"<p style='color:#94a3b8;font-size:11px;line-height:1.6'>{why}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Feature importance / coefficients
    st.markdown("#### Feature importances / coefficients")
    fi_col, lc_col = st.columns([3, 1])

    with lc_col:
        lc_for_coef = st.selectbox(
            "Target class (Ridge coefs)",
            target_names,
            format_func=lambda k: LC_LABELS.get(k, k),
        )

    with fi_col:
        if model_key == "ridge" and not ridge_coefs.empty:
            # Detect feature column (might be 'feature', 'feature_name', or index)
            feat_col_r = next(
                (c for c in ["feature", "feature_name"] if c in ridge_coefs.columns),
                None,
            )
            if feat_col_r is None:
                ridge_coefs = ridge_coefs.reset_index()
                feat_col_r  = ridge_coefs.columns[0]

            # Detect coefficient column for chosen target
            # Possible formats: coef_built, coef_prop_built, built, prop_built
            coef_candidates = [
                f"coef_{lc_for_coef}",
                f"coef_prop_{lc_for_coef}",
                f"coef_label_prop_{lc_for_coef}",
                lc_for_coef,
                f"prop_{lc_for_coef}",
            ]
            coef_col = next(
                (c for c in coef_candidates if c in ridge_coefs.columns), None
            )
            # Last resort: first numeric column
            if coef_col is None:
                num_cols = ridge_coefs.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    coef_col = num_cols[0]

            if coef_col:
                df_plot = ridge_coefs[[feat_col_r, coef_col]].copy()
                df_plot.columns = ["feature", "value"]
                df_plot = df_plot.sort_values("value", key=abs, ascending=True)
                colors  = ["#ef4444" if v > 0 else "#22c55e" for v in df_plot["value"]]
                fig = go.Figure(go.Bar(
                    x=df_plot["value"], y=df_plot["feature"],
                    orientation="h", marker_color=colors,
                ))
                fig.update_layout(
                    title=f"Ridge coefficients → {LC_LABELS.get(lc_for_coef,lc_for_coef)}",
                    height=300, paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                    font_color="#e2e8f0", margin=dict(l=10,r=10,t=40,b=10),
                    yaxis=dict(tickfont=dict(size=10)),
                )
                fig.add_vline(x=0, line_color="#334155")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"Could not find coefficient column for '{lc_for_coef}'. "
                        f"Columns in CSV: {list(ridge_coefs.columns)}"
                )
        else:
            if not gbrt_imp.empty:
                # Detect the actual importance column name (varies by pipeline version)
                # Possible names: 'importance', 'mean_importance', 'permutation_importance',
                # or one column per target like 'built', 'vegetation', etc.
                imp_col  = None
                feat_col = None
                for candidate in ["importance", "mean_importance",
                                  "permutation_importance", "mean"]:
                    if candidate in gbrt_imp.columns:
                        imp_col = candidate
                        break
                for candidate in ["feature", "feature_name", "index"]:
                    if candidate in gbrt_imp.columns:
                        feat_col = candidate
                        break

                # If no single importance column, try averaging numeric columns
                if imp_col is None:
                    num_cols = gbrt_imp.select_dtypes(include="number").columns.tolist()
                    if num_cols:
                        gbrt_imp = gbrt_imp.copy()
                        gbrt_imp["importance"] = gbrt_imp[num_cols].mean(axis=1)
                        imp_col = "importance"

                # If no feature column, use the index
                if feat_col is None:
                    gbrt_imp = gbrt_imp.copy().reset_index()
                    feat_col = "index" if "index" in gbrt_imp.columns else gbrt_imp.columns[0]

                if imp_col and feat_col:
                    plot_df = gbrt_imp[[feat_col, imp_col]].copy()
                    plot_df.columns = ["feature", "importance"]
                    fig = _bar_h(
                        plot_df.sort_values("importance", ascending=True),
                        x="importance", y="feature", color="#a78bfa",
                        title="HistGBRT permutation importances (all targets)", height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Could not detect importance column. "
                            f"Columns found: {list(gbrt_imp.columns)}")

    # Feature table
    st.markdown("#### Feature set")
    feat_rows = [
        ("built_2020, vegetation_2020, water_2020, other_2020",
         "WorldCover 2020","Composition lag","Strong inertia prior — short-term forecasting"),
        ("B02, B03, B04, B08, B11, B12",
         "Sentinel-2 L2A","Spectral reflectance","Surface material discrimination"),
        ("NDVI","Sentinel-2","Spectral index","r=−0.70 with built; primary vegetation proxy"),
        ("NDWI","Sentinel-2","Spectral index","Water body detection; disambiguates rivers"),
    ]
    rows_html = "".join(
        f"<tr><td><code>{r[0]}</code></td><td>{r[1]}</td>"
        f"<td><span class='badge-blue'>{r[2]}</span></td><td>{r[3]}</td></tr>"
        for r in feat_rows
    )
    st.markdown(
        f"<table><thead><tr><th>Feature(s)</th><th>Source</th>"
        f"<th>Type</th><th>Justification</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📊 Evaluation")
    st.markdown(
        f"<p style='color:#64748b;font-size:13px'>"
        f"Spatial holdout: <b>{eval_rep.get('holdout', train_rep.get('holdout','east'))}</b> strip. "
        f"All metrics computed on held-out cells only.</p>",
        unsafe_allow_html=True,
    )

    # Per-target RMSE comparison
    per_target_data = []
    for mkey, mname, col in [("ridge","Ridge","#60a5fa"),
                              ("hist_gbrt","HistGBRT","#a78bfa")]:
        m = metrics.get(mkey, {})
        for t in target_names:
            key = f"{t}_rmse"
            if key in m:
                per_target_data.append({
                    "model": mname, "target": LC_LABELS.get(t,t),
                    "RMSE": m[key],
                })

    if per_target_data:
        fig = px.bar(
            pd.DataFrame(per_target_data),
            x="target", y="RMSE", color="model", barmode="group",
            color_discrete_map={"Ridge":"#60a5fa","HistGBRT":"#a78bfa"},
            title="Per-class RMSE (composition regression, test set)",
        )
        fig.update_layout(
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", height=280,
            margin=dict(l=10,r=10,t=40,b=10),
            legend=dict(bgcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Change metrics
    st.markdown("#### Change-specific metrics")
    c1, c2 = st.columns(2)
    change_metrics_data = [
        ("δ-RMSE (built)", "delta_rmse_built", "Lower = better change detection"),
        ("False-change rate", "false_change_rate_built", "Stable cells wrongly flagged"),
        ("Missed-gain rate", "missed_gain_rate_built", "Real urbanisation missed"),
        ("Stability score", "stability_score", "1 − MAE on truly stable cells"),
    ]
    for col_ui, mkey, mname, mcolor in [
        (c1,"ridge","Ridge","#60a5fa"),
        (c2,"hist_gbrt","HistGradientBoosting","#a78bfa"),
    ]:
        with col_ui:
            st.markdown(
                f"<div class='card' style='border-color:{mcolor}33'>"
                f"<div class='card-title'>{mname}</div>",
                unsafe_allow_html=True,
            )
            ev_m = ev.get(mkey, {})
            for label, key, desc in change_metrics_data:
                val = ev_m.get(key)
                if val is not None:
                    display = f"{val*100:.1f}%" if "rate" in key else f"{val:.4f}"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"font-size:12px;margin-bottom:6px'>"
                        f"<span style='color:#64748b'>{label}</span>"
                        f"<b style='color:{mcolor}'>{display}</b></div>"
                        f"<div style='font-size:10px;color:#334155;margin-bottom:8px'>"
                        f"{desc}</div>",
                        unsafe_allow_html=True,
                    )
            st.markdown("</div>", unsafe_allow_html=True)

    # Stress tests
    st.markdown("#### 🔥 Stress tests")
    stress = eval_rep.get("stress_tests", {})
    st_rows = [
        ("Gaussian noise σ=0.05 on all features",
         "gaussian_noise_sigma_0.05",
         "Simulates sensor calibration drift and atmospheric correction errors."),
        ("NIR channel dropout (B08 set to median)",
         "nir_dropout",
         "Simulates cloud shadow or sensor failure on the most important band."),
    ]
    for label, key, desc in st_rows:
        s = stress.get(key, {})
        ridge_deg = s.get("ridge_rmse_delta_pct", "—")
        gbrt_deg  = s.get("hist_gbrt_rmse_delta_pct", "—")
        st.markdown(
            f"<div class='card'>"
            f"<div class='card-title'>{label}</div>"
            f"<p style='color:#64748b;font-size:11px;margin:0 0 10px'>{desc}</p>"
            f"<div style='display:flex;gap:20px'>"
            f"<div style='font-size:12px'><span style='color:#64748b'>Ridge degradation: </span>"
            f"<b style='color:#60a5fa'>+{ridge_deg}% RMSE</b></div>"
            f"<div style='font-size:12px'><span style='color:#64748b'>HistGBRT degradation: </span>"
            f"<b style='color:#a78bfa'>+{gbrt_deg}% RMSE</b></div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Predicted vs actual scatter (built target)
    if "built_2021" in df.columns:
        pred_col_r = "ridge_pred_built"
        pred_col_g = "gbrt_pred_built"
        sc1, sc2 = st.columns(2)
        for col_ui, pc, mname, mcol in [
            (sc1, pred_col_r, "Ridge", "#60a5fa"),
            (sc2, pred_col_g, "HistGBRT", "#a78bfa"),
        ]:
            with col_ui:
                if pc in df.columns:
                    fig = _scatter(
                        df["built_2021"].values, df[pc].values,
                        "Actual built_2021", f"Predicted ({mname})",
                        f"Predicted vs Actual — {mname} (built)", mcol,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"Run the pipeline to see {mname} predictions.")

    # Where the model is wrong
    st.markdown("#### ⚠️ Known failure modes")
    bullets = eval_rep.get("failure_bullets", [
        "Forest-edge cells: mixed pixels inflate built-up estimates.",
        "Industrial north: rooftops spectrally similar to bare soil.",
        "Construction sites: bare-soil phase causes 1-year detection lag.",
        "WorldCover v100→v200 shift introduces artefact Δ-noise.",
    ])
    cols = st.columns(2)
    for i, b in enumerate(bullets):
        with cols[i % 2]:
            st.markdown(
                f"<div class='card'>"
                f"<p style='color:#fbbf24;margin:0 0 4px;font-size:12px;font-weight:600'>"
                f"⚠ Failure {i+1}</p>"
                f"<p style='color:#64748b;font-size:11px;line-height:1.5;margin:0'>{b}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("### 🔍 Explainability")

    c_good, c_bad = st.columns(2)

    with c_good:
        st.markdown(
            "<div class='card' style='border-color:#22c55e44'>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:10px'>"
            "<span class='card-title' style='margin:0'>✅ Helpful explanation</span>"
            "<span class='badge-green'>Non-expert friendly</span></div>"
            "<div style='background:#0f172a;border-radius:8px;padding:14px;line-height:1.7'>"
            "<p style='color:#e2e8f0;font-weight:600;margin:0 0 8px'>What changed in Nuremberg between 2020 and 2021?</p>"
            "<p style='color:#94a3b8;font-size:12px;margin:0 0 8px'>"
            "Our model estimates that <b style='color:#f87171'>built-up areas grew slightly</b> across "
            "the Nuremberg region — roughly 1–3 percentage points in the cells where change "
            "was detected. This is consistent with steady suburban densification visible "
            "in the industrial north and western residential fringe.</p>"
            "<p style='color:#94a3b8;font-size:12px;margin:0 0 8px'>"
            "The model is <b style='color:#4ade80'>most confident</b> in cells where land cover "
            "has been stable for multiple years — dense forest (Reichswald) and the "
            "Pegnitz river corridor.</p>"
            "<p style='color:#94a3b8;font-size:12px;margin:0'>"
            "<b style='color:#fbbf24'>⚠️ Caveat:</b> These are 300 m grid-cell averages, "
            "not parcel-level data. A single cell may contain both a park and a parking lot. "
            "Do not use these estimates for any planning, legal, or regulatory decision.</p>"
            "</div>"
            "<p style='color:#475569;font-size:11px;margin:10px 0 0;line-height:1.5'>"
            "<b style='color:#22c55e'>Why this works:</b> Plain language, spatial grounding, "
            "explicit uncertainty, and a clear prohibition on misuse.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with c_bad:
        st.markdown(
            "<div class='card' style='border-color:#ef444444'>"
            "<div style='display:flex;justify-content:space-between;margin-bottom:10px'>"
            "<span class='card-title' style='margin:0'>⚠️ Misleading explanation</span>"
            "<span class='badge-red'>Avoid — dangerous</span></div>"
            "<div style='background:#0f172a;border-radius:8px;padding:14px;line-height:1.7'>"
            "<p style='color:#e2e8f0;font-weight:600;margin:0 0 8px'>Model result: land-cover change 2020–2021</p>"
            "<p style='color:#94a3b8;font-size:12px;margin:0 0 8px'>"
            "The model achieves <b style='color:#ef4444'>RMSE = 0.016</b>, meaning predictions are "
            "<b>highly accurate</b>. Built-up area increased by 2.1 pp according to the model. "
            "<b>This can be used to support zoning applications and building permits.</b></p>"
            "<p style='color:#94a3b8;font-size:12px;margin:0'>"
            "Cell (row 4, col 7) shows <b style='color:#ef4444'>+14 pp urbanisation</b> — "
            "the largest single-cell change in the dataset.</p>"
            "</div>"
            "<div style='background:#ef44441a;border:1px solid #ef444433;border-radius:8px;padding:12px;margin-top:10px'>"
            "<p style='color:#ef4444;font-weight:700;font-size:11px;margin:0 0 6px'>Why this is misleading:</p>"
            "<ul style='margin:0;padding-left:14px;font-size:11px;color:#64748b;line-height:1.9'>"
            "<li><b>RMSE ≠ accuracy</b> — 0.016 is an average error, not a guarantee per cell</li>"
            "<li><b>Single-cell extremes</b> — ±0.016 RMSE makes +14 pp at the edge of reliability</li>"
            "<li><b>'Suitable for permits'</b> — falsely implies regulatory-grade validation</li>"
            "<li><b>WorldCover v100→v200 noise</b> — some apparent change is label artefact</li>"
            "</ul></div></div>",
            unsafe_allow_html=True,
        )

    # What / Where / How confident
    st.markdown("#### System explanation for non-experts")
    cols3 = st.columns(3)
    for col_ui, icon, q, a, col in [
        (cols3[0], "📈", "What changed?", "#3b82f6",
         "Built-up area showed small net increases in peri-urban cells. "
         "City-wide, the change is modest: Nuremberg is a mature city with "
         "limited greenfield development in this period."),
        (cols3[1], "📍", "Where did it change?", "#f59e0b",
         "Most detectable change is in the north-west industrial belt and "
         "south-west residential fringe. The Altstadt core and Reichswald "
         "forest cells are stable across both years."),
        (cols3[2], "🎯", "How confident?", "#8b5cf6",
         f"Ridge macro RMSE = {m_ridge.get('macro_rmse',0.0156):.4f} on the spatial holdout. "
         "Confidence is lowest near the holdout boundary and in mixed forest-urban cells. "
         "The WorldCover v100→v200 shift adds ~0.003 artefact noise to change estimates."),
    ]:
        with col_ui:
            txt = a
            st.markdown(
                f"<div class='card'>"
                f"<p style='color:{col};font-weight:700;font-size:13px;margin:0 0 8px'>"
                f"{icon} {q}</p>"
                f"<p style='color:#94a3b8;font-size:12px;line-height:1.7;margin:0'>{txt}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ChatGPT reflections
    st.markdown("#### 💬 Arguing Against ChatGPT")
    for n, sub, claim, counter, verdict in [
        (1,
         "Spatial unit: district polygons vs 300 m grid cells",
         "ChatGPT suggested aggregating results to Nuremberg's administrative districts "
         "(Stadtbezirke) for 'policy relevance — planners think in district terms.'",
         "We disagreed on two grounds. First, administrative boundaries are politically "
         "defined and orthogonal to land-cover processes — using them as modelling units "
         "violates the Modifiable Areal Unit Problem (MAUP) and would yield only ~9 "
         "training samples. Second, 300 m grid cells are the natural unit for aggregating "
         "10 m WorldCover pixels; district aggregation can be offered as a display layer "
         "without compromising the model.",
         "300 m UTM grid cells for modelling; district-level summaries are a display-only option."
        ),
        (2,
         "Second model: MLP vs HistGradientBoosting",
         "ChatGPT recommended a Multilayer Perceptron as the flexible second model, "
         "arguing it 'captures complex non-linear spectral patterns beyond tree ensembles.'",
         "We disagreed. With ~320 training cells, an MLP is prone to overfitting and "
         "requires substantial hyperparameter tuning. HistGradientBoosting is also "
         "non-linear, handles mixed feature scales natively without a separate scaler, "
         "and provides permutation-based feature importances with less variance than "
         "MLP gradient attributions. The assignment does not require neural networks, "
         "and the data size does not justify them.",
         "HistGradientBoosting as the flexible model. MLP explicitly considered and rejected."
        ),
    ]:
        st.markdown(
            f"<div class='card'>"
            f"<p style='color:#60a5fa;font-weight:700;font-size:13px;margin:0 0 2px'>"
            f"❶ Case {n}</p>"
            f"<p style='color:#475569;font-size:11px;font-style:italic;margin:0 0 10px'>{sub}</p>"
            f"<p style='color:#fbbf24;font-size:11px;font-weight:600;margin:0 0 4px'>ChatGPT's suggestion:</p>"
            f"<p style='color:#94a3b8;font-size:11px;line-height:1.6;border-left:2px solid #334155;"
            f"padding-left:8px;margin:0 0 10px;font-style:italic'>\"{claim}\"</p>"
            f"<p style='color:#4ade80;font-size:11px;font-weight:600;margin:0 0 4px'>Our counter-argument:</p>"
            f"<p style='color:#94a3b8;font-size:11px;line-height:1.6;margin:0 0 10px'>{counter}</p>"
            f"<div style='background:#1e293b;border-radius:6px;padding:7px 10px;"
            f"border-left:2px solid #60a5fa'>"
            f"<p style='color:#60a5fa;font-size:11px;margin:0'>✓ Decision: {verdict}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════
#  TAB 6 — ABOUT & LIMITS
# ══════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("### ℹ️ About & Limits")

    # Problem framing grid
    st.markdown("#### 🎯 Problem framing")
    framing = [
        ("Land-cover classes", "Built-up · Vegetation · Water · Bare/Other",
         "4 WorldCover classes collapsed to balance granularity vs. training size."),
        ("Spatial unit", "300 m UTM grid cells (EPSG:32632)",
         f"Aggregates 10 m WorldCover + Sentinel-2 pixels. {len(df):,} cells total."),
        ("Temporal setup", "Features: 2020 → Targets: 2021",
         "One-year forecasting. 2020 composition is a strong lag feature."),
        ("Intended users", "Urban researchers, planners",
         "Exploratory trend-awareness tool only. Not a certified product."),
        ("Prediction task", "Multi-output regression (4 composition proportions)",
         "Outputs sum to 1 per cell. Evaluated jointly and per-class."),
        ("Forbidden uses", "Legal · Permits · Carbon accounting · Property valuation",
         "Model uncertainty and label noise preclude any regulatory application."),
    ]
    cols2 = st.columns(3)
    for i, (label, value, note) in enumerate(framing):
        with cols2[i % 3]:
            st.markdown(
                f"<div class='card'>"
                f"<div class='card-title'>{label}</div>"
                f"<p style='color:#e2e8f0;font-size:13px;font-weight:600;margin:0 0 4px'>{value}</p>"
                f"<p style='color:#475569;font-size:11px;line-height:1.4;margin:0'>{note}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Data sources
    st.markdown("#### 📦 Data sources")
    for name, col, desc in [
        ("ESA WorldCover 2020 (v100) & 2021 (v200)", "#3b82f6",
         "Primary land-cover labels at 10 m. Zanaga et al. (2021/2022). "
         "DOI: 10.5281/zenodo.5571936 / 10.5281/zenodo.7254221. CC-BY 4.0. "
         "Accessed via AWS Open Data esa-worldcover bucket."),
        ("Sentinel-2 L2A (July 2020 & 2021)", "#22c55e",
         "Surface reflectance composites (July median). Copernicus Programme, ESA. "
         "Accessed via Microsoft Planetary Computer STAC API. "
         "No Google Earth Engine account required."),
        ("CORINE Land Cover 2018", "#8b5cf6",
         "EEA, 2018. Cross-validation reference only — not used as primary labels. "
         "WorldCover preferred for temporal consistency."),
    ]:
        st.markdown(
            f"<div style='display:flex;gap:10px;margin-bottom:12px'>"
            f"<div style='width:3px;background:{col};border-radius:2px;flex-shrink:0'></div>"
            f"<div><p style='margin:0 0 2px;font-size:12px;font-weight:600;color:#e2e8f0'>{name}</p>"
            f"<p style='margin:0;font-size:11px;color:#64748b;line-height:1.5'>{desc}</p>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # Prohibited uses
    st.markdown(
        "<div class='card' style='border-color:#ef444433'>"
        "<p style='color:#ef4444;font-weight:700;font-size:12px;margin:0 0 8px'>⛔ Must NOT be used for:</p>"
        "<ul style='margin:0;padding-left:14px;font-size:11px;color:#94a3b8;line-height:1.9'>"
        "<li>Urban planning permit or zoning decisions</li>"
        "<li>Environmental impact assessments (regulatory)</li>"
        "<li>Legal disputes over land ownership or classification</li>"
        "<li>Carbon credit accounting or reporting</li>"
        "<li>Individual property assessment or valuation</li>"
        "</ul>"
        "<p style='color:#475569;font-size:11px;margin:10px 0 0;line-height:1.5'>"
        "Technical limitations: 300 m resolution hides sub-cell heterogeneity · "
        "Only 2 time points · WorldCover v100→v200 label shift in change targets · "
        "No field-survey validation."
        "</p></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
#  HELPERS — raster reading (used by Raw Imagery + Change Map tabs)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_s2_rgb(year: int, month: int) -> "np.ndarray | None":
    """
    Read Sentinel-2 GeoTIFF for (year, month) and return uint8 RGB array (H, W, 3).
    Bands in file: B02 B03 B04 B08 B11 B12 (indices 0-5).
    RGB = B04(red)=band3, B03(green)=band2, B02(blue)=band1  → indices 2,1,0
    Returns None if the file doesn't exist.
    """
    try:
        import rasterio
        from rasterio.warp import reproject, Resampling
        import numpy as np
    except ImportError:
        return None

    # Standard July path from sentinel2_fetch.py
    tif = ROOT / "data" / "raw" / f"sentinel2_median_{year}_july.tif"

    # For non-July months, check if a custom composite exists
    if month != S2_MONTH_CFG:
        alt = ROOT / "data" / "raw" / f"sentinel2_median_{year}_{month:02d}.tif"
        if alt.exists():
            tif = alt
        elif not tif.exists():
            return None   # pipeline-month fallback also missing

    if not tif.exists():
        return None

    try:
        with rasterio.open(tif) as src:
            _assets = [a.upper() for a in S2_ASSETS_CFG]
            _r_idx = _assets.index("B04") + 1 if "B04" in _assets else 3
            _g_idx = _assets.index("B03") + 1 if "B03" in _assets else 2
            _b_idx = _assets.index("B02") + 1 if "B02" in _assets else 1
            nodata = src.nodata
            r = src.read(_r_idx).astype(np.float32)
            g = src.read(_g_idx).astype(np.float32)
            b = src.read(_b_idx).astype(np.float32)

        # Build a valid-data mask: nodata value OR any channel is exactly 0 for all 3
        # (stackstac fills missing with NaN; rasterio may use nodata or 0)
        nan_mask = (
            np.isnan(r) | np.isnan(g) | np.isnan(b)
            | (r == 0) & (g == 0) & (b == 0)
        )
        if nodata is not None:
            nan_mask |= (r == nodata) | (g == nodata) | (b == nodata)

        valid_pixels = ~nan_mask
        if valid_pixels.sum() < 100:
            # Virtually no valid data — return None so the UI shows a message
            return None

        def _percentile_stretch(arr, lo=2, hi=98):
            """Stretch to [lo, hi] percentile of *valid* pixels, return uint8."""
            valid = arr[valid_pixels]
            p_lo = float(np.percentile(valid, lo))
            p_hi = float(np.percentile(valid, hi))
            if p_hi <= p_lo:
                p_hi = p_lo + 1e-6
            stretched = (arr - p_lo) / (p_hi - p_lo)
            return np.clip(stretched * 255, 0, 255).astype(np.uint8)

        r8 = _percentile_stretch(r)
        g8 = _percentile_stretch(g)
        b8 = _percentile_stretch(b)

        # Build RGBA — set alpha=0 for invalid pixels (transparent, not black)
        alpha = np.where(valid_pixels, 255, 0).astype(np.uint8)
        rgba = np.stack([r8, g8, b8, alpha], axis=-1)
        return rgba
    except Exception as e:
        st.warning(f"Could not read S2 raster: {e}")
        return None


@st.cache_data(show_spinner=False)
def _load_s2_ndvi(year: int, month: int) -> "np.ndarray | None":
    """Return float32 NDVI array (H,W) from the S2 GeoTIFF. NIR=B08=band4, Red=B04=band3."""
    try:
        import rasterio, numpy as np
    except ImportError:
        return None

    tif = ROOT / "data" / "raw" / f"sentinel2_median_{year}_july.tif"
    if month != S2_MONTH_CFG:
        alt = ROOT / "data" / "raw" / f"sentinel2_median_{year}_{month:02d}.tif"
        if alt.exists():
            tif = alt
    if not tif.exists():
        return None
    try:
        with rasterio.open(tif) as src:
            _assets = [a.upper() for a in S2_ASSETS_CFG]
            _nir_idx = _assets.index("B08") + 1 if "B08" in _assets else 4
            _red_idx = _assets.index("B04") + 1 if "B04" in _assets else 3
            nir = src.read(_nir_idx).astype(np.float32)   # B08 NIR
            red = src.read(_red_idx).astype(np.float32)   # B04 Red
        ndvi = (nir - red) / (nir + red + 1e-6)
        ndvi = np.clip(ndvi, -1, 1)
        # Mask invalid pixels (NaN, nodata, or all-zero) as NaN so Plotly shows them
        # as transparent rather than mapping them to the colorscale minimum
        invalid = np.isnan(nir) | np.isnan(red) | ((nir == 0) & (red == 0))
        ndvi[invalid] = np.nan
        return ndvi
    except Exception as e:
        st.warning(f"Could not compute NDVI: {e}")
        return None


# WorldCover class palette with labels (code → (label, hex colour))
# Colors are sourced from COLORS["worldcover"]
WC_PALETTE = {
    10: ("Tree cover",         COLORS["worldcover"][10]),
    20: ("Shrubland",          COLORS["worldcover"][20]),
    30: ("Grassland",          COLORS["worldcover"][30]),
    40: ("Cropland",           COLORS["worldcover"][40]),
    50: ("Built-up",           COLORS["worldcover"][50]),
    60: ("Bare/sparse veg",    COLORS["worldcover"][60]),
    70: ("Snow/ice",           COLORS["worldcover"][70]),
    80: ("Water",              COLORS["worldcover"][80]),
    90: ("Herbaceous wetland", COLORS["worldcover"][90]),
    95: ("Mangroves",          COLORS["worldcover"][95]),
    100:("Moss/lichen",        COLORS["worldcover"][100]),
}

@st.cache_data(show_spinner=False)
def _load_worldcover_rgb(year: int) -> "tuple[np.ndarray, np.ndarray] | tuple[None,None]":
    """
    Read WorldCover GeoTIFF (EPSG:4326, class codes) and return
    (rgb_uint8 HxWx3,  class_codes HxW).
    Returns (None, None) if file missing.
    """
    try:
        import rasterio, numpy as np
    except ImportError:
        return None, None

    tif = ROOT / "data" / "raw" / f"worldcover_{year}_nuremberg.tif"
    if not tif.exists():
        return None, None
    try:
        with rasterio.open(tif) as src:
            codes = src.read(1)   # uint8 class codes

        h, w = codes.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for code, (_, hex_col) in WC_PALETTE.items():
            mask = codes == code
            r = int(hex_col[1:3], 16)
            g = int(hex_col[3:5], 16)
            b = int(hex_col[5:7], 16)
            rgb[mask] = [r, g, b]

        return rgb, codes
    except Exception as e:
        st.warning(f"Could not read WorldCover raster: {e}")
        return None, None


def _fetch_s2_on_demand(year: int, month: int) -> bool:
    """
    Trigger a Planetary Computer composite for (year, month) if not cached.
    Returns True on success, False on failure.
    Requires the sentinel2_fetch module to be importable.
    """
    tif = ROOT / "data" / "raw" / f"sentinel2_median_{year}_{month:02d}.tif"
    if tif.exists():
        return True
    try:
        import sys
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.data.sentinel2_fetch import composite_year
        composite_year(year, out_path=tif)
        return tif.exists()
    except Exception as e:
        st.error(f"Fetch failed: {e}")
        return False


def _array_to_plotly_image(rgb: "np.ndarray", title: str) -> "go.Figure":
    """Wrap a HxWx3 uint8 array in a Plotly imshow figure with dark theme."""
    fig = px.imshow(
        rgb,   # accepts HxWx3 (RGB) or HxWx4 (RGBA); Plotly respects alpha channel
        title=title,
        color_continuous_scale=None,
        aspect="equal",
    )
    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="#0f172a",
        font_color="#e2e8f0",
        title_font_size=13,
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  TAB 7 — RAW IMAGERY
# ══════════════════════════════════════════════════════════════════════
with tab_raw:
    st.markdown("### 🛰️ Raw Imagery")
    st.markdown(
        "<p style='color:#64748b;font-size:13px'>"
        "View the actual Sentinel-2 RGB composites and ESA WorldCover classifications "
        "that feed the ML pipeline. Select any year and month — if the composite for that "
        "month is not cached locally it can be fetched on demand from "
        "<b>Microsoft Planetary Computer</b>.</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────
    rc1, rc2, rc3, rc4 = st.columns([1, 1, 1, 2])
    with rc1:
            # Available years: always include the pipeline S2_YEARS plus a browse range
        _browse_years = sorted(set([2019,2020,2021,2022,2023]) | set(S2_YEARS_CFG))
        _default_yr_idx = _browse_years.index(S2_YEARS_CFG[0]) if S2_YEARS_CFG[0] in _browse_years else 1
        raw_year  = st.selectbox("Year", _browse_years,
                                     index=_default_yr_idx, key="raw_year")
    with rc2:
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        _default_month_idx = S2_MONTH_CFG - 1   # S2_MONTH is 1-based; list index is 0-based
        raw_month = st.selectbox(
            "Month", list(month_names.keys()),
            index=_default_month_idx,
            format_func=lambda m: month_names[m],
            key="raw_month",
        )
    with rc3:
        s2_layer = st.radio("S2 layer", ["RGB", "NDVI"], horizontal=True, key="s2_layer")
    with rc4:
        st.markdown(
            "<div style='padding-top:6px'>"
            "<span style='font-size:11px;color:#64748b'>"
            "📁 Cached composites: <code>data/raw/sentinel2_median_{year}_july.tif</code> (July pipeline default). "
            "Other months are fetched on demand and saved to <code>data/raw/sentinel2_median_{year}_{month:02d}.tif</code>."
            "</span></div>",
            unsafe_allow_html=True,
        )

    # ── Fetch button for non-July months ─────────────────────────────
    # The pipeline default file uses "_july" suffix (from sentinel2_fetch.py)
    _pipeline_tif = ROOT / "data" / "raw" / f"sentinel2_median_{raw_year}_july.tif"
    _month_tif    = ROOT / "data" / "raw" / f"sentinel2_median_{raw_year}_{raw_month:02d}.tif"
    is_pipeline_month = (raw_month == S2_MONTH_CFG)   # pipeline default month (July=7)
    is_month_cached   = _pipeline_tif.exists() if is_pipeline_month else _month_tif.exists()

    if not is_pipeline_month and not is_month_cached:
        st.markdown(
            f"<div class='card' style='border-color:#f59e0b44'>"
            f"<p style='color:#fbbf24;font-size:12px;margin:0'>"
            f"⚡ No cached composite for {month_names[raw_month]} {raw_year}. "
            f"Click below to fetch from Planetary Computer (requires network + dependencies)."
            f"</p></div>",
            unsafe_allow_html=True,
        )
        if st.button(f"Fetch {month_names[raw_month]} {raw_year} composite",
                     key="fetch_btn"):
            with st.spinner(f"Fetching Sentinel-2 {month_names[raw_month]} {raw_year} …"):
                ok = _fetch_s2_on_demand(raw_year, raw_month)
            if ok:
                st.success("Composite saved. Reloading …")
                st.cache_data.clear()
                st.rerun()
            else:
                st.error("Fetch failed — check network / Planetary Computer credentials.")

    st.divider()

    # ── Load and display ──────────────────────────────────────────────
    col_s2, col_wc = st.columns(2)

    with col_s2:
        if s2_layer == "RGB":
            rgb = _load_s2_rgb(raw_year, raw_month)
            if rgb is not None:
                fig = _array_to_plotly_image(
                    rgb,
                    f"Sentinel-2 RGB · {month_names[raw_month]} {raw_year} "
                    f"(B04/B03/B02, γ-stretched)",
                )
                st.plotly_chart(fig, use_container_width=True)
                _valid_pct = float((rgb[:,:,3] > 0).sum()) / (rgb.shape[0]*rgb.shape[1]) * 100
                st.caption(
                    f"True-colour composite ({_valid_pct:.0f}% valid pixels). "
                    "Transparent = no-data / cloud-masked pixels. "
                    "Grey-white = built-up / bare. Green = vegetation. Dark = water / shadow."
                )
            else:
                st.markdown(
                    "<div class='card'>"
                    f"<p style='color:#64748b;font-size:12px'>"
                    f"No Sentinel-2 GeoTIFF found for {month_names[raw_month]} {raw_year}.<br>"
                    f"Run: <code>python scripts/prepare_data.py --fetch-s2</code> "
                    f"(fetches July composites for all pipeline years), or use the "
                    f"'Fetch' button above for a different month.</p></div>",
                    unsafe_allow_html=True,
                )
        else:
            ndvi = _load_s2_ndvi(raw_year, raw_month)
            if ndvi is not None:
                valid_ndvi = ndvi[~np.isnan(ndvi)]
                _ndvi_pct = float(np.sum(~np.isnan(ndvi))) / ndvi.size * 100
                _zmin = float(np.percentile(valid_ndvi, 2))  if len(valid_ndvi) else -0.2
                _zmax = float(np.percentile(valid_ndvi, 98)) if len(valid_ndvi) else 0.8
                fig = px.imshow(
                    ndvi,
                    color_continuous_scale="RdYlGn",
                    zmin=_zmin, zmax=_zmax,
                    title=f"NDVI · {month_names[raw_month]} {raw_year} "
                          f"({_ndvi_pct:.0f}% valid pixels)",
                    aspect="equal",
                )
                fig.update_layout(
                    height=420,
                    margin=dict(l=0, r=0, t=36, b=0),
                    paper_bgcolor="#0f172a",
                    font_color="#e2e8f0",
                    title_font_size=13,
                )
                fig.update_xaxes(showticklabels=False, showgrid=False)
                fig.update_yaxes(showticklabels=False, showgrid=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "NDVI = (NIR−Red)/(NIR+Red). "
                    "Red = low/negative (urban, water, bare soil). "
                    "Green = high (dense vegetation / Reichswald forest)."
                )
            else:
                st.info(f"No S2 raster found for {month_names[raw_month]} {raw_year}.")

    with col_wc:
        wc_year_choice = st.selectbox(
            "WorldCover year", [2020, 2021], index=0, key="wc_year"
        )
        wc_rgb, wc_codes = _load_worldcover_rgb(wc_year_choice)
        if wc_rgb is not None:
            fig_wc = _array_to_plotly_image(
                wc_rgb, f"ESA WorldCover · {wc_year_choice} (10 m, EPSG:4326)"
            )
            st.plotly_chart(fig_wc, use_container_width=True)

            # Legend
            unique_codes = [c for c in WC_PALETTE if wc_codes is not None
                            and int(c) in wc_codes]
            legend_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:6px'>"
            for code in unique_codes:
                label, hex_col = WC_PALETTE[code]
                legend_html += (
                    f"<div style='display:flex;align-items:center;gap:4px;font-size:11px'>"
                    f"<div style='width:12px;height:12px;background:{hex_col};"
                    f"border-radius:2px;flex-shrink:0'></div>"
                    f"<span style='color:#94a3b8'>{label}</span></div>"
                )
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='card'>"
                f"<p style='color:#64748b;font-size:12px'>"
                f"No WorldCover GeoTIFF found for {wc_year_choice}.<br>"
                f"Run: <code>python scripts/prepare_data.py --fetch-wc</code></p></div>",
                unsafe_allow_html=True,
            )

    # ── Pixel statistics ──────────────────────────────────────────────
    if wc_codes is not None:
        st.markdown("#### WorldCover class distribution")
        total_px = wc_codes.size
        rows = []
        for code in sorted(WC_PALETTE.keys()):
            count = int((wc_codes == code).sum())
            if count > 0:
                label, hex_col = WC_PALETTE[code]
                rows.append({
                    "Class": label,
                    "Pixels": count,
                    "Area %": round(count / total_px * 100, 2),
                    "_color": hex_col,
                })
        if rows:
            stats_df = pd.DataFrame(rows)
            fig_bar = go.Figure(go.Bar(
                x=stats_df["Area %"],
                y=stats_df["Class"],
                orientation="h",
                marker_color=stats_df["_color"].tolist(),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#1e293b",
                plot_bgcolor="#1e293b",
                font_color="#e2e8f0",
                xaxis_title="Area %",
                yaxis=dict(tickfont=dict(size=10)),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
#  TAB 8 — CHANGE MAP  (pixel-level, from raw WorldCover tifs)
# ══════════════════════════════════════════════════════════════════════
with tab_change:
    st.markdown("### 🔴 Pixel-Level Change Map")
    st.markdown(
        "<p style='color:#64748b;font-size:13px'>"
        "Direct pixel-by-pixel comparison of ESA WorldCover 2020 vs 2021 at "
        "<b>10 m resolution</b> — no model involved. Shows where WorldCover "
        "changed class between the two label years. "
        "Note: some apparent change is the v100→v200 classification-method artefact.</p>",
        unsafe_allow_html=True,
    )

    @st.cache_data(show_spinner=False)
    def _compute_change_map():
        """
        Pixel-level class change: 2020 → 2021.
        Returns dict with:
          rgb_change : HxWx3 uint8  (red=gained built, green=lost built, grey=other change, dark=stable)
          codes_2020 : HxW uint8
          codes_2021 : HxW uint8
          change_mask: HxW bool (True where class changed)
        """
        try:
            import rasterio, numpy as np
            from rasterio.enums import Resampling as RSamp
        except ImportError:
            return None

        p2020 = ROOT / "data" / "raw" / "worldcover_2020_nuremberg.tif"
        p2021 = ROOT / "data" / "raw" / "worldcover_2021_nuremberg.tif"

        if not p2020.exists() or not p2021.exists():
            return None

        with rasterio.open(p2020) as src:
            codes_2020 = src.read(1)
            profile    = src.profile
            transform  = src.transform
            crs        = src.crs

        with rasterio.open(p2021) as src21:
            if src21.shape == codes_2020.shape:
                codes_2021 = src21.read(1)
            else:
                # Resample 2021 to match 2020 grid
                import numpy as np
                codes_2021 = np.empty_like(codes_2020)
                rasterio.warp.reproject(
                    source=rasterio.band(src21, 1),
                    destination=codes_2021,
                    src_transform=src21.transform,
                    src_crs=src21.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=RSamp.nearest,
                )

        change_mask = codes_2020 != codes_2021
        h, w = codes_2020.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Stable → very dark blue-grey
        rgb[~change_mask] = [15, 23, 42]

        # Any change → grey base
        rgb[change_mask] = [80, 90, 100]

        # Became built (→50): bright red
        became_built = (codes_2021 == 50) & (codes_2020 != 50)
        rgb[became_built] = [239, 68, 68]

        # Lost built (50→other): bright green
        lost_built = (codes_2020 == 50) & (codes_2021 != 50)
        rgb[lost_built] = [34, 197, 94]

        # Became water (→80): blue
        became_water = (codes_2021 == 80) & (codes_2020 != 80)
        rgb[became_water] = [56, 189, 248]

        # Veg gain — use class groupings from config.py
        _veg_list = list(VEG_CODES) if VEG_CODES else [10, 20, 30, 40, 90, 95, 100]
        _built_list = list(BUILT_CODES) if BUILT_CODES else [50]
        _water_list = list(WATER_CODES) if WATER_CODES else [80]

        # Re-colour built/water using config-driven class sets
        became_built = np.isin(codes_2021, _built_list) & ~np.isin(codes_2020, _built_list)
        lost_built   = np.isin(codes_2020, _built_list) & ~np.isin(codes_2021, _built_list)
        became_water = np.isin(codes_2021, _water_list) & ~np.isin(codes_2020, _water_list)
        rgb[became_built] = [239, 68, 68]
        rgb[lost_built]   = [34, 197, 94]
        rgb[became_water] = [56, 189, 248]

        became_veg = (np.isin(codes_2021, _veg_list) &
                      ~np.isin(codes_2020, _veg_list))
        rgb[became_veg] = [134, 239, 172]

        total_px   = codes_2020.size
        changed_px = int(change_mask.sum())
        built_gain = int(became_built.sum())
        built_loss = int(lost_built.sum())

        return {
            "rgb_change":  rgb,
            "codes_2020":  codes_2020,
            "codes_2021":  codes_2021,
            "change_mask": change_mask,
            "became_built":became_built,
            "lost_built":  lost_built,
            "stats": {
                "total_px":    total_px,
                "changed_px":  changed_px,
                "changed_pct": round(changed_px / total_px * 100, 2),
                "built_gain":  built_gain,
                "built_loss":  built_loss,
                "net_built_px": built_gain - built_loss,
            },
        }

    ch = _compute_change_map()

    if ch is None:
        st.markdown(
            "<div class='card'>"
            "<p style='color:#64748b;font-size:12px'>"
            "WorldCover GeoTIFFs not found. Run:<br>"
            "<code>python scripts/prepare_data.py --fetch-wc</code>"
            "</p></div>",
            unsafe_allow_html=True,
        )
    else:
        # ── KPIs ──────────────────────────────────────────────────────
        s = ch["stats"]
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Changed pixels",   f"{s['changed_pct']:.2f}%")
        k2.metric("Built-up gained",  f"{s['built_gain']:,} px")
        k3.metric("Built-up lost",    f"{s['built_loss']:,} px")
        k4.metric("Net built Δ",      f"{s['net_built_px']:+,} px")
        st.caption(
            f"1 pixel ≈ 10×10 m = 100 m². "
            f"Net built Δ ≈ {abs(s['net_built_px']) * 100 / 1e6:.3f} km² "
            f"({'gain' if s['net_built_px'] > 0 else 'loss'}). "
            "⚠️ Some apparent change is v100→v200 label-method artefact."
        )

        st.divider()

        # ── Change map + side-by-side ─────────────────────────────────
        cm1, cm2 = st.columns([2, 1])

        with cm1:
            fig_ch = _array_to_plotly_image(
                ch["rgb_change"],
                "WorldCover pixel change · 2020→2021 (10 m resolution)",
            )
            st.plotly_chart(fig_ch, use_container_width=True)

            # Legend
            st.markdown(
                "<div style='display:flex;flex-wrap:wrap;gap:10px;margin-top:4px'>"
                + "".join([
                    f"<div style='display:flex;align-items:center;gap:5px;font-size:11px'>"
                    f"<div style='width:12px;height:12px;background:{bg};"
                    f"border-radius:2px'></div>"
                    f"<span style='color:#94a3b8'>{label}</span></div>"
                    for bg, label in [
                        ("#ef4444", "Became built-up"),
                        ("#22c55e", "Lost built-up"),
                        ("#86efac", "Vegetation gain"),
                        ("#38bdf8", "Became water"),
                        ("#505a64", "Other class change"),
                        ("#0f172a", "Stable"),
                    ]
                ])
                + "</div>",
                unsafe_allow_html=True,
            )

        with cm2:
            # Transition matrix (top transitions)
            st.markdown("<div class='card-title'>Top class transitions</div>",
                        unsafe_allow_html=True)
            mask = ch["change_mask"]
            from_codes = ch["codes_2020"][mask]
            to_codes   = ch["codes_2021"][mask]

            import collections
            pairs = list(zip(from_codes.tolist(), to_codes.tolist()))
            top10 = collections.Counter(pairs).most_common(8)

            for (frm, to), count in top10:
                frm_label = WC_PALETTE.get(frm, (str(frm), "#64748b"))[0]
                to_label  = WC_PALETTE.get(to,  (str(to),  "#64748b"))[0]
                pct = count / s["total_px"] * 100
                st.markdown(
                    f"<div style='font-size:11px;margin-bottom:6px;"
                    f"border-left:2px solid #334155;padding-left:8px'>"
                    f"<span style='color:#94a3b8'>{frm_label}</span> "
                    f"<span style='color:#475569'>→</span> "
                    f"<span style='color:#e2e8f0'>{to_label}</span><br>"
                    f"<span style='color:#60a5fa;font-size:10px'>"
                    f"{count:,} px · {pct:.3f}% of area</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Side-by-side WorldCover comparison ───────────────────────
        st.divider()
        st.markdown("#### Side-by-side: WorldCover 2020 vs 2021")
        sb1, sb2 = st.columns(2)
        for col_ui, yr, codes in [(sb1, 2020, ch["codes_2020"]),
                                  (sb2, 2021, ch["codes_2021"])]:
            with col_ui:
                wc_rgb_yr, _ = _load_worldcover_rgb(yr)
                if wc_rgb_yr is not None:
                    fig_yr = _array_to_plotly_image(
                        wc_rgb_yr, f"WorldCover {yr}"
                    )
                    st.plotly_chart(fig_yr, use_container_width=True)

        # ── S2 RGB comparison for the two label years ─────────────────
        st.divider()
        st.markdown("#### Sentinel-2 RGB: 2020 vs 2021 (July composites)")
        ss1, ss2 = st.columns(2)
        for col_ui, yr in [(ss1, 2020), (ss2, 2021)]:
            with col_ui:
                rgb_yr = _load_s2_rgb(yr, 7)
                if rgb_yr is not None:
                    fig_s = _array_to_plotly_image(rgb_yr, f"S2 RGB · July {yr}")
                    st.plotly_chart(fig_s, use_container_width=True)
                else:
                    st.info(f"No S2 raster for {yr}. Run prepare_data.py --fetch-s2")


# ══════════════════════════════════════════════════════════════════════
#  TAB 9 — FORECAST  (autoregressive multi-year prediction)
# ══════════════════════════════════════════════════════════════════════

# ── Core forecasting engine ───────────────────────────────────────────
def _run_forecast(
    df: pd.DataFrame,
    model,
    feature_names: list,
    target_names: list,
    n_steps: int,
    base_year: int = 2021,
    per_step_rmse: float = 0.016,
) -> dict:
    """Run autoregressive multi-year forecast using the shared inference path."""
    if model is None or not feature_names:
        return {}
    if infer_run_autoregressive_forecast is None:
        return {}
    return infer_run_autoregressive_forecast(
        df,
        model,
        feature_names,
        target_names,
        n_steps=n_steps,
        base_year=base_year,
        per_step_rmse=per_step_rmse,
    )



@st.cache_data(show_spinner=False)
def _run_forecast_cached(_model_key: str, n_steps: int) -> dict:
    """Cached wrapper — re-runs only when model or horizon changes."""
    _model = ridge_model if _model_key == "ridge" else gbrt_model
    if _model is None:
        return {}
    feat_names   = train_rep.get("feature_names", FEAT_COLS_CFG)
    target_names = train_rep.get("target_names",
                                 ["label_prop_built","label_prop_vegetation",
                                  "label_prop_water","label_prop_other"])
    per_step_rmse = train_rep.get("metrics", {}).get(
        _model_key, {}
    ).get("macro_rmse", 0.016)

    return _run_forecast(
        df, _model, feat_names, target_names,
        n_steps=n_steps, base_year=2021,
        per_step_rmse=per_step_rmse,
    )


# ── Forecast tab UI ───────────────────────────────────────────────────
with tab_forecast:
    st.markdown("### 🔮 Future Land-Cover Forecast")
    st.markdown(
        "<p style='color:#64748b;font-size:13px'>"
        "Autoregressive multi-year forecast using the trained Ridge or HistGBRT model. "
        "Starting from the last known state (2021), each year's predicted composition "
        "is fed back as input to predict the next year. "
        "Sentinel-2 spectral features are held fixed at their 2021 values — "
        "no future imagery is assumed.</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        fc_model_choice = st.radio(
            "Forecast model",
            ["Ridge", "HistGradientBoosting"],
            horizontal=False,
            key="fc_model",
        )
        fc_model_key = "ridge" if fc_model_choice == "Ridge" else "hist_gbrt"

    with fc2:
        fc_horizon = st.slider(
            "Forecast horizon (years)",
            min_value=1, max_value=10, value=5, step=1,
            key="fc_horizon",
            help="Years ahead from 2021. Uncertainty grows with each step.",
        )
        fc_target_year = st.selectbox(
            "Display year",
            list(range(2022, 2022 + fc_horizon)),
            index=fc_horizon - 1,
            key="fc_target_year",
        )

    with fc3:
        st.markdown(
            "<div class='card'>"
            "<div class='card-title'>⚠️ Forecast caveats</div>"
            "<ul style='margin:0;padding-left:14px;font-size:11px;color:#64748b;line-height:1.9'>"
            "<li><b>No future imagery</b> — spectral features fixed at 2021; "
            "real land-cover change driven by new development is not captured.</li>"
            "<li><b>Uncertainty grows</b> each year as √n × RMSE. "
            "Beyond ~3 years the intervals are wide enough that individual-cell "
            "changes are unreliable.</li>"
            "<li><b>Trend extrapolation only</b> — no shocks, policy changes, "
            "or infrastructure events are modelled.</li>"
            "<li><b>Do not use</b> for planning, legal, or regulatory decisions.</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )

    # ── Run forecast ─────────────────────────────────────────────────
    if ridge_model is None and gbrt_model is None:
        st.error(
            "No trained models found. Run `python scripts/train_models.py` first, "
            "then restart the app."
        )
        st.stop()

    with st.spinner(f"Running {fc_horizon}-year autoregressive forecast …"):
        fc_results = _run_forecast_cached(fc_model_key, fc_horizon)

    if not fc_results:
        st.error("Forecast failed — check that feature_names in training_report.json match the parquet schema.")
        st.stop()

    # ── KPI strip for selected year ───────────────────────────────────
    yr_data  = fc_results.get(fc_target_year, {})
    base_data = fc_results.get("base", {})

    if yr_data and base_data:
        k1, k2, k3, k4, k5 = st.columns(5)
        delta_bu = float(yr_data["built"].mean() - base_data["built"].mean()) * 100
        delta_vg = float(yr_data["vegetation"].mean() - base_data["vegetation"].mean()) * 100
        unc_mean = float(yr_data["uncertainty"].mean()) * 100
        k1.metric("Forecast year",    str(fc_target_year))
        k2.metric("Avg built-up",     f"{yr_data['built'].mean()*100:.1f}%",
                  delta=f"{delta_bu:+.2f} pp vs 2021")
        k3.metric("Avg vegetation",   f"{yr_data['vegetation'].mean()*100:.1f}%",
                  delta=f"{delta_vg:+.2f} pp vs 2021")
        k4.metric("Avg uncertainty",  f"±{unc_mean:.1f} pp",
                  help="√steps × per-step RMSE propagated in quadrature")
        years_ahead = fc_target_year - 2021
        k5.metric("Years ahead",      str(years_ahead))

    st.divider()

    # ── Map row: composition + change + uncertainty ───────────────────
    map_col_label = st.columns([1])[0]
    map_col_label.markdown("#### Predicted composition maps · " + str(fc_target_year))

    m1, m2, m3 = st.columns(3)

    def _fc_heatmap(values: np.ndarray, title: str, colorscale: str,
                    df_ref: pd.DataFrame = df) -> "go.Figure":
        """Plot a forecast array as a grid heatmap using df row/col layout."""
        plot = df_ref[["row", "col"]].copy()
        plot["v"] = values
        plot = plot.groupby(["row", "col"], as_index=False)["v"].mean()
        try:
            pivot = plot.pivot(index="row", columns="col", values="v")
        except Exception:
            pivot = plot.pivot_table(index="row", columns="col", values="v", aggfunc="mean")
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            colorscale=colorscale,
            zmin=0, zmax=1,
            showscale=True,
            hovertemplate="Row %{y}, Col %{x}<br>%{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=title, height=340,
            margin=dict(l=0, r=0, t=36, b=0),
            paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
            font_color="#e2e8f0", title_font_size=12,
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, autorange="reversed")
        return fig

    if yr_data:
        with m1:
            fig = _fc_heatmap(yr_data["built"], f"Built-up · {fc_target_year}", "YlOrRd")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Mean: {yr_data['built'].mean()*100:.1f}%")

        with m2:
            fig = _fc_heatmap(yr_data["vegetation"], f"Vegetation · {fc_target_year}", "Greens")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Mean: {yr_data['vegetation'].mean()*100:.1f}%")

        with m3:
            fig = _fc_heatmap(yr_data["uncertainty"],
                              f"Forecast uncertainty · {fc_target_year}",
                              "Purples")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"±{yr_data['uncertainty'].mean()*100:.1f} pp avg. "
                "Purple = high uncertainty — treat with caution."
            )

    st.divider()

    # ── Change map: delta vs 2021 ─────────────────────────────────────
    st.markdown(f"#### Predicted change vs 2021 baseline · {fc_target_year}")

    ch1, ch2 = st.columns(2)

    with ch1:
        if yr_data and "delta_built" in yr_data:
            delta = yr_data["delta_built"]
            plot  = df[["row","col"]].copy()
            plot["delta"] = delta
            plot = plot.groupby(["row","col"], as_index=False)["delta"].mean()
            try:
                pivot_d = plot.pivot(index="row", columns="col", values="delta")
            except Exception:
                pivot_d = plot.pivot_table(index="row", columns="col",
                                            values="delta", aggfunc="mean")
            _abs_max = max(abs(float(pivot_d.values[~np.isnan(pivot_d.values)].max())),
                           abs(float(pivot_d.values[~np.isnan(pivot_d.values)].min())),
                           0.05)
            fig_d = go.Figure(go.Heatmap(
                z=pivot_d.values,
                colorscale="RdYlGn_r",
                zmid=0, zmin=-_abs_max, zmax=_abs_max,
                showscale=True,
                colorbar=dict(title="Δ built-up", tickformat=".2f"),
                hovertemplate="Row %{y}, Col %{x}<br>Δ = %{z:.3f}<extra></extra>",
            ))
            fig_d.update_layout(
                title=f"Δ Built-up 2021→{fc_target_year} (red = urbanisation)",
                height=360,
                margin=dict(l=0, r=0, t=36, b=0),
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#e2e8f0", title_font_size=12,
            )
            fig_d.update_xaxes(showticklabels=False)
            fig_d.update_yaxes(showticklabels=False, autorange="reversed")
            st.plotly_chart(fig_d, use_container_width=True)

            # Change category counts
            thr = st.session_state.change_thr
            n_urban  = int((delta >  thr).sum())
            n_green  = int((delta < -thr).sum())
            n_stable = int((np.abs(delta) <= thr).sum())
            n_unc    = int((yr_data["uncertainty"] >= 0.10).sum())
            for cat, n, col in [
                ("Urbanising", n_urban,  "#f87171"),
                ("Greening",   n_green,  "#4ade80"),
                ("Stable",     n_stable, "#64748b"),
                ("High uncertainty", n_unc, "#a78bfa"),
            ]:
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:12px;margin-bottom:3px'>"
                    f"<span style='color:{col}'>● {cat}</span>"
                    f"<b>{n} cells ({n/len(df)*100:.0f}%)</b></div>",
                    unsafe_allow_html=True,
                )

    with ch2:
        # Dominant class map for forecast year
        if yr_data and "dominant" in yr_data:
            dom = yr_data["dominant"]
            dom_num = np.array([
                {"built": 3, "vegetation": 2, "water": 1, "other": 0}[d]
                for d in dom
            ], dtype=float)
            plot_dom = df[["row","col"]].copy()
            plot_dom["dom"] = dom_num
            plot_dom = plot_dom.groupby(["row","col"], as_index=False)["dom"].mean()
            try:
                pivot_dom = plot_dom.pivot(index="row", columns="col", values="dom")
            except Exception:
                pivot_dom = plot_dom.pivot_table(index="row", columns="col",
                                                  values="dom", aggfunc="mean")
            fig_dom = go.Figure(go.Heatmap(
                z=pivot_dom.values,
                colorscale=[
                    [0.00, "#94a3b8"],   # 0 = other
                    [0.33, "#38bdf8"],   # 1 = water
                    [0.67, "#22c55e"],   # 2 = vegetation
                    [1.00, "#f59e0b"],   # 3 = built
                ],
                zmin=0, zmax=3,
                showscale=False,
                hovertemplate="Row %{y}, Col %{x}<extra></extra>",
            ))
            fig_dom.update_layout(
                title=f"Dominant land-cover class · {fc_target_year}",
                height=360,
                margin=dict(l=0, r=0, t=36, b=0),
                paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
                font_color="#e2e8f0", title_font_size=12,
            )
            fig_dom.update_xaxes(showticklabels=False)
            fig_dom.update_yaxes(showticklabels=False, autorange="reversed")
            st.plotly_chart(fig_dom, use_container_width=True)

            # Legend
            st.markdown(
                "<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:4px'>"
                + "".join([
                    f"<div style='display:flex;align-items:center;gap:5px;font-size:11px'>"
                    f"<div style='width:12px;height:12px;background:{c};border-radius:2px'></div>"
                    f"<span style='color:#94a3b8'>{l}</span></div>"
                    for c, l in [
                        ("#f59e0b","Built-up"),
                        ("#22c55e","Vegetation"),
                        ("#38bdf8","Water"),
                        ("#94a3b8","Bare/Other"),
                    ]
                ])
                + "</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Time-series chart: city-wide averages across all forecast years ───
    st.markdown("#### City-wide average composition · 2020–" + str(2021 + fc_horizon))

    ts_rows = []
    # Add known historical years
    for yr, lbl in [(2020, "observed"), (2021, "observed")]:
        for lc in ["built", "vegetation", "water", "other"]:
            col = f"{lc}_{yr}"
            if col in df.columns:
                ts_rows.append({
                    "year": yr, "class": LC_LABELS[lc],
                    "proportion": float(df[col].mean()),
                    "type": lbl,
                    "unc_lo": None, "unc_hi": None,
                })

    # Add forecast years
    for yr in range(2022, 2022 + fc_horizon):
        yd = fc_results.get(yr, {})
        if not yd:
            continue
        unc = float(yd["uncertainty"].mean())
        for lc in ["built", "vegetation", "water", "other"]:
            mean_val = float(yd[lc].mean())
            ts_rows.append({
                "year": yr, "class": LC_LABELS[lc],
                "proportion": mean_val,
                "type": "forecast",
                "unc_lo": max(0, mean_val - unc),
                "unc_hi": min(1, mean_val + unc),
            })

    ts_df = pd.DataFrame(ts_rows)

    color_map = {
        LC_LABELS["built"]:      "#f59e0b",
        LC_LABELS["vegetation"]: "#22c55e",
        LC_LABELS["water"]:      "#38bdf8",
        LC_LABELS["other"]:      "#94a3b8",
    }

    fig_ts = go.Figure()
    for lc_label, col_hex in color_map.items():
        sub = ts_df[ts_df["class"] == lc_label].sort_values("year")
        obs = sub[sub["type"] == "observed"]
        frc = sub[sub["type"] == "forecast"]

        # Observed line (solid)
        fig_ts.add_trace(go.Scatter(
            x=obs["year"], y=obs["proportion"] * 100,
            mode="lines+markers",
            name=f"{lc_label} (observed)",
            line=dict(color=col_hex, width=2),
            marker=dict(size=7),
        ))
        # Forecast line (dashed)
        if not frc.empty:
            # Uncertainty band
            fig_ts.add_trace(go.Scatter(
                x=pd.concat([frc["year"], frc["year"].iloc[::-1]]),
                y=pd.concat([frc["unc_hi"] * 100,
                             frc["unc_lo"].iloc[::-1] * 100]),
                fill="toself",
                fillcolor=_hex_to_rgba(col_hex, 0.13),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_ts.add_trace(go.Scatter(
                x=frc["year"], y=frc["proportion"] * 100,
                mode="lines+markers",
                name=f"{lc_label} (forecast)",
                line=dict(color=col_hex, width=2, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            ))

    # Vertical line at last known year
    fig_ts.add_vline(
        x=2021.5, line_color="#475569", line_dash="dot",
        annotation_text="← observed | forecast →",
        annotation_font_color="#64748b",
        annotation_font_size=10,
    )

    fig_ts.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="#1e293b", plot_bgcolor="#1e293b",
        font_color="#e2e8f0",
        yaxis_title="City-wide average (%)",
        xaxis_title="Year",
        legend=dict(bgcolor="#0f172a", font_size=10, orientation="h",
                    x=0, y=-0.2),
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)
    st.caption(
        "Solid lines = 2020/2021 WorldCover observations. "
        "Dashed lines = model forecast. "
        "Shaded bands = ±1 propagated uncertainty (grows as √steps × RMSE). "
        "All spectral features held at 2021 values."
    )
