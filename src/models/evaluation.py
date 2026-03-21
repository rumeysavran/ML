"""Full evaluation report: spatial split, change metrics, stress tests, failure discussion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np

from src.config import PATHS, TARGET_COLUMNS_COMPOSITION_T1
from src.models.change_metrics import (
    built_change_event_metrics,
    delta_arrays,
    delta_rmse_macro,
)
from src.models.dataset import composition_forecasting_xy, spatial_train_test_mask
from src.models.stress_tests import stress_test_suite
from src.models.train import _per_target_metrics


def _macro_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    names = list(TARGET_COLUMNS_COMPOSITION_T1)
    return float(_per_target_metrics(y_true, y_pred, names)["rmse_macro"])


def run_evaluation(
    holdout: str = "east",
    models_dir: str | Path | None = None,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    models_dir = Path(models_dir or PATHS.models)
    ridge_path = models_dir / "ridge_composition.joblib"
    gbrt_path = models_dir / "hist_gbrt_composition.joblib"
    if not ridge_path.exists() or not gbrt_path.exists():
        raise FileNotFoundError(
            f"Train models first: missing {ridge_path.name} or {gbrt_path.name} under {models_dir}"
        )

    meta_path = models_dir / "test_fold_meta.joblib"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        saved_h = meta.get("holdout")
        if saved_h is not None and saved_h != holdout:
            raise ValueError(
                f"Holdout mismatch: test_fold_meta has holdout={saved_h!r} "
                f"but evaluate was called with holdout={holdout!r}. Re-run train_models.py with the same --holdout."
            )

    X_df, y_df, cell_ids = composition_forecasting_xy()
    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64)
    target_names = list(y_df.columns)

    train_mask, test_mask = spatial_train_test_mask(cell_ids, holdout=holdout)
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    ridge = joblib.load(ridge_path)
    gbrt = joblib.load(gbrt_path)

    rng = np.random.default_rng(rng_seed)

    def eval_model(name: str, model: Any) -> Dict[str, Any]:
        y_hat = model.predict(X_te)
        comp = _per_target_metrics(y_te, y_hat, target_names)
        d_true, d_pred = delta_arrays(y_te, y_hat, X_te)
        deltas = delta_rmse_macro(d_true, d_pred, target_names)
        built = built_change_event_metrics(d_true, d_pred, built_index=0)

        pred_fn: Callable[[np.ndarray], np.ndarray] = lambda Xv: model.predict(Xv)
        stress = stress_test_suite(
            pred_fn,
            X_tr,
            X_te,
            y_te,
            _macro_rmse,
            rng,
            noise_levels=(0.5, 1.0),
            dropout_feature="s2_b08",
        )

        return {
            "composition_metrics": comp,
            "delta_metrics": deltas,
            "built_change_diagnostics": built,
            "stress_tests_macro_rmse": stress,
        }

    report: Dict[str, Any] = {
        "holdout": holdout,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "targets": target_names,
        "ridge": eval_model("ridge", ridge),
        "hist_gbrt": eval_model("hist_gbrt", gbrt),
        "where_models_are_likely_wrong": _failure_discussion(),
    }

    out_json = models_dir / "evaluation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def _failure_discussion() -> List[str]:
    """Bullet points for the written report / app limitations panel."""
    return [
        "Cells dominated by agriculture often show spectral swings between July composites that are "
        "not matched one-to-one with WorldCover year-to-year label noise (v100 vs v200).",
        "Shallow water, riparian vegetation, and mixed pixels confuse NDWI/MNDWI-based features; "
        "errors cluster along rivers and floodplains.",
        "Urban fringe pixels split across 300 m cells: a partial built fraction can jump when fine "
        "10 m labels reclassify without large on-the-ground change.",
        "East–west spatial hold-out still leaves similar land systems on both sides of Nuremberg; "
        "metrics can look strong while extrapolation to a new city would degrade.",
        "Using 2020 composition as an input makes one-year-ahead forecasting easier; reported R² is "
        "partly 'persistence' of labels, not pure vision from Sentinel-2 alone.",
    ]


if __name__ == "__main__":
    r = run_evaluation()
    print(json.dumps({k: r[k] for k in ("holdout", "n_test")}, indent=2))
