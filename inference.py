"""Inference helpers for land-cover composition models.

This module centralises model metadata normalisation, feature-matrix building,
and autoregressive forecasting so training code and the Streamlit app use one
consistent inference path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

LC_ORDER = ("built", "vegetation", "water", "other")

_SPECTRAL_ALIASES = {
    "s2_b02": ("s2_b02", "b02", "B02"),
    "s2_b03": ("s2_b03", "b03", "B03"),
    "s2_b04": ("s2_b04", "b04", "B04"),
    "s2_b08": ("s2_b08", "b08", "B08"),
    "s2_b11": ("s2_b11", "b11", "B11"),
    "s2_b12": ("s2_b12", "b12", "B12"),
    "ndvi": ("ndvi", "NDVI"),
    "ndbi": ("ndbi", "NDBI"),
    "ndwi": ("ndwi", "NDWI"),
    "mndwi": ("mndwi", "MNDWI"),
    "mean_blue": ("mean_blue",),
    "mean_green": ("mean_green",),
    "mean_red": ("mean_red",),
    "mean_nir": ("mean_nir",),
    "mean_swir1": ("mean_swir1",),
    "mean_swir2": ("mean_swir2",),
}


def canonical_landcover_name(name: str | None) -> str | None:
    """Map raw target or feature names to the app's 4 coarse classes."""
    if not name:
        return None
    low = str(name).strip().lower()
    if "built" in low:
        return "built"
    if "veget" in low or "tree" in low or "grass" in low or "crop" in low:
        return "vegetation"
    if "water" in low:
        return "water"
    if "other" in low or "bare" in low:
        return "other"
    return None


def _is_composition_feature(name: str | None) -> bool:
    low = str(name or "").strip().lower()
    if low in LC_ORDER:
        return True
    lc = canonical_landcover_name(low)
    if lc is None:
        return False
    tokens = [tok for tok in low.replace("-", "_").split("_") if tok]
    return (
        low.startswith("label_prop_")
        or low.startswith("prop_")
        or low.startswith(f"{lc}_")
        or low.endswith(f"_{lc}")
        or lc in tokens
    )


def _canonical_spectral_name(name: str | None) -> str | None:
    low = str(name or "").strip().lower()
    for canonical, aliases in _SPECTRAL_ALIASES.items():
        if low in {a.lower() for a in aliases}:
            return canonical
    return None


def normalize_training_report(
    report: Mapping[str, Any] | None,
    fallback_feature_names: Sequence[str] | None = None,
    fallback_target_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalise older and newer training-report schemas into one format."""
    src = dict(report or {})

    feature_names = list(
        src.get("feature_names") or src.get("features") or fallback_feature_names or []
    )
    raw_target_names = list(
        src.get("raw_target_names") or src.get("target_names") or src.get("targets") or fallback_target_names or []
    )
    target_names = [canonical_landcover_name(name) or name for name in raw_target_names]

    metrics_src = src.get("metrics") if isinstance(src.get("metrics"), dict) else {}
    metrics: dict[str, dict[str, float]] = {}

    for model_key in ("ridge", "hist_gbrt"):
        out: dict[str, float] = {}
        existing = metrics_src.get(model_key, {}) if isinstance(metrics_src.get(model_key), dict) else {}
        out.update(existing)

        legacy = src.get(model_key, {}) if isinstance(src.get(model_key), dict) else {}
        if "macro_rmse" not in out and "rmse_macro" in legacy:
            out["macro_rmse"] = float(legacy["rmse_macro"])

        per_target = legacy.get("per_target", []) if isinstance(legacy.get("per_target"), list) else []
        for row in per_target:
            if not isinstance(row, dict):
                continue
            lc = canonical_landcover_name(row.get("target"))
            if not lc:
                continue
            if f"{lc}_rmse" not in out and row.get("rmse") is not None:
                out[f"{lc}_rmse"] = float(row["rmse"])
            if f"{lc}_mae" not in out and row.get("mae") is not None:
                out[f"{lc}_mae"] = float(row["mae"])
            if f"{lc}_r2" not in out and row.get("r2") is not None:
                out[f"{lc}_r2"] = float(row["r2"])

        metrics[model_key] = out

    out_report = dict(src)
    out_report["feature_names"] = feature_names
    out_report["raw_target_names"] = raw_target_names
    out_report["target_names"] = target_names
    out_report["features"] = feature_names
    out_report["targets"] = raw_target_names
    out_report["metrics"] = metrics
    out_report["holdout"] = src.get("holdout") or src.get("spatial_holdout") or "east"
    return out_report


def _column_candidates(name: str, prefer_year: int | None = None, fallback_year: int | None = None) -> list[str]:
    names: list[str] = []

    spectral = _canonical_spectral_name(name)
    if spectral:
        alias_group = list(_SPECTRAL_ALIASES[spectral])
    else:
        alias_group = [name]

    for year in (prefer_year, fallback_year):
        if year is None:
            continue
        for alias in alias_group:
            names.append(f"{alias}_{year}")

    names.extend(alias_group)
    return names


def _values_from_candidates(df: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    for col in candidates:
        if col in df.columns:
            return df[col].to_numpy(dtype=float, copy=True)
    return None


def _default_composition(n_rows: int) -> dict[str, np.ndarray]:
    return {lc: np.full(n_rows, 0.25, dtype=float) for lc in LC_ORDER}


def _renormalise_composition(state: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrs = {lc: np.clip(np.asarray(state[lc], dtype=float), 0.0, 1.0) for lc in LC_ORDER}
    total = np.zeros_like(arrs[LC_ORDER[0]], dtype=float)
    for lc in LC_ORDER:
        total += arrs[lc]
    total = np.where(total <= 1e-12, 1.0, total)
    return {lc: arrs[lc] / total for lc in LC_ORDER}


def get_composition_state(
    df: pd.DataFrame,
    composition_year: int | None = None,
    fallback_year: int | None = None,
) -> dict[str, np.ndarray]:
    """Read land-cover composition from a wide dataframe."""
    state = _default_composition(len(df))
    for lc in LC_ORDER:
        candidates = []
        for year in (composition_year, fallback_year):
            if year is None:
                continue
            candidates.extend(
                [
                    f"{lc}_{year}",
                    f"label_prop_{lc}_{year}",
                    f"prop_{lc}_{year}",
                ]
            )
        candidates.extend([lc, f"label_prop_{lc}", f"prop_{lc}"])
        values = _values_from_candidates(df, candidates)
        if values is not None:
            state[lc] = values
    return _renormalise_composition(state)


def build_feature_matrix(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    spectral_year: int,
    spectral_fallback_year: int | None = None,
    composition_state: Mapping[str, np.ndarray] | None = None,
    composition_year: int | None = None,
    composition_fallback_year: int | None = None,
) -> np.ndarray:
    """Build a model-ready feature matrix in the original training order."""
    if not feature_names:
        return np.empty((len(df), 0), dtype=float)

    state = (
        _renormalise_composition(composition_state)
        if composition_state is not None
        else get_composition_state(
            df,
            composition_year=composition_year,
            fallback_year=composition_fallback_year,
        )
    )

    X = np.zeros((len(df), len(feature_names)), dtype=float)
    for j, feature_name in enumerate(feature_names):
        if _is_composition_feature(feature_name):
            lc = canonical_landcover_name(feature_name)
            if lc:
                X[:, j] = state[lc]
            continue

        candidates = _column_candidates(
            feature_name,
            prefer_year=spectral_year,
            fallback_year=spectral_fallback_year,
        )
        values = _values_from_candidates(df, candidates)
        if values is not None:
            X[:, j] = values

    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def predict_composition(
    model: Any,
    df: pd.DataFrame,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    spectral_year: int,
    spectral_fallback_year: int | None = None,
    composition_state: Mapping[str, np.ndarray] | None = None,
    composition_year: int | None = None,
    composition_fallback_year: int | None = None,
) -> dict[str, np.ndarray]:
    """Run one-step inference and return a normalised 4-class composition."""
    X = build_feature_matrix(
        df,
        feature_names,
        spectral_year=spectral_year,
        spectral_fallback_year=spectral_fallback_year,
        composition_state=composition_state,
        composition_year=composition_year,
        composition_fallback_year=composition_fallback_year,
    )
    raw = np.asarray(model.predict(X), dtype=float)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)

    out: dict[str, np.ndarray] = {}
    for idx, name in enumerate(target_names):
        lc = canonical_landcover_name(name)
        if lc and idx < raw.shape[1]:
            out[lc] = np.clip(raw[:, idx], 0.0, 1.0)

    base_state = (
        _renormalise_composition(composition_state)
        if composition_state is not None
        else get_composition_state(
            df,
            composition_year=composition_year,
            fallback_year=composition_fallback_year,
        )
    )
    for lc in LC_ORDER:
        if lc not in out:
            out[lc] = base_state[lc]
    return _renormalise_composition(out)


def run_autoregressive_forecast(
    df: pd.DataFrame,
    model: Any,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    n_steps: int,
    base_year: int = 2021,
    per_step_rmse: float = 0.016,
) -> dict[Any, Any]:
    """Forecast coarse composition for future years with fixed spectral inputs."""
    if model is None or not feature_names:
        return {}

    base_state = get_composition_state(df, composition_year=base_year, fallback_year=base_year - 1)
    built_base = base_state["built"].copy()
    results: dict[Any, Any] = {
        "base": {
            "built": built_base,
            "vegetation": base_state["vegetation"],
            "water": base_state["water"],
            "other": base_state["other"],
            "uncertainty": np.zeros(len(df), dtype=float),
        }
    }

    current = {lc: base_state[lc].copy() for lc in LC_ORDER}
    for step in range(1, int(n_steps) + 1):
        next_state = predict_composition(
            model,
            df,
            feature_names,
            target_names,
            spectral_year=base_year,
            spectral_fallback_year=base_year - 1,
            composition_state=current,
        )
        uncertainty = np.full(len(df), np.sqrt(step) * float(per_step_rmse), dtype=float)
        stack = np.stack([next_state[lc] for lc in LC_ORDER], axis=1)
        dominant_idx = stack.argmax(axis=1)
        dominant = np.asarray([LC_ORDER[i] for i in dominant_idx], dtype=object)
        pred_year = base_year + step
        results[pred_year] = {
            "built": next_state["built"],
            "vegetation": next_state["vegetation"],
            "water": next_state["water"],
            "other": next_state["other"],
            "uncertainty": uncertainty,
            "delta_built": next_state["built"] - built_base,
            "dominant": dominant,
        }
        current = {lc: next_state[lc].copy() for lc in LC_ORDER}

    return results


def _wide_frame_from_composition(pred: Mapping[str, np.ndarray], cell_ids: np.ndarray | None = None) -> pd.DataFrame:
    out = pd.DataFrame({lc: np.asarray(pred[lc], dtype=float) for lc in LC_ORDER})
    if cell_ids is not None:
        out.insert(0, "cell_id", np.asarray(cell_ids))
    return out


def _cli() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run inference for trained land-cover models.")
    parser.add_argument("--features", required=True, help="Path to wide tabular features parquet/csv.")
    parser.add_argument("--model", required=True, help="Path to a joblib model artifact.")
    parser.add_argument("--training-report", required=True, help="Path to training_report.json.")
    parser.add_argument(
        "--mode",
        choices=("one-step", "forecast"),
        default="one-step",
        help="One-step t0→t1 inference or autoregressive future forecast.",
    )
    parser.add_argument("--base-year", type=int, default=2021, help="Base year for forecast mode.")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon in years.")
    parser.add_argument("--out", required=True, help="Output csv/json file path.")
    args = parser.parse_args()

    if joblib is None:
        raise ImportError("joblib is required for CLI inference.")

    feature_path = Path(args.features)
    if feature_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(feature_path)
    else:
        df = pd.read_csv(feature_path)

    model = joblib.load(args.model)
    report = normalize_training_report(json.loads(Path(args.training_report).read_text()))
    feature_names = report["feature_names"]
    target_names = report["target_names"]
    out_path = Path(args.out)

    if args.mode == "one-step":
        pred = predict_composition(
            model,
            df,
            feature_names,
            target_names,
            spectral_year=args.base_year - 1,
            composition_year=args.base_year - 1,
            composition_fallback_year=args.base_year,
        )
        out_df = _wide_frame_from_composition(pred, df["cell_id"].to_numpy() if "cell_id" in df.columns else None)
        out_df.to_csv(out_path, index=False)
    else:
        forecast = run_autoregressive_forecast(
            df,
            model,
            feature_names,
            target_names,
            n_steps=args.horizon,
            base_year=args.base_year,
            per_step_rmse=report.get("metrics", {}).get("ridge", {}).get("macro_rmse", 0.016),
        )
        serialisable: dict[str, dict[str, list[Any]]] = {}
        for key, value in forecast.items():
            if not isinstance(value, dict):
                continue
            serialisable[str(key)] = {}
            for inner_key, arr in value.items():
                serialisable[str(key)][inner_key] = np.asarray(arr).tolist()
        out_path.write_text(json.dumps(serialisable))


if __name__ == "__main__":  # pragma: no cover
    _cli()
