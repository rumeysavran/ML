"""Fit both models, evaluate with spatial hold-out, persist artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    FEATURE_COLUMNS_T0,
    PATHS,
    PREDICTION_BASE_YEAR,
    PREDICTION_TARGET_YEAR,
    TARGET_COLUMNS_COMPOSITION_T1,
)
from src.models.dataset import composition_forecasting_xy, spatial_train_test_mask
from src.models.pipelines import build_hist_gbrt_pipeline, build_ridge_pipeline, ridge_coefficient_table

try:
    from inference import canonical_landcover_name, normalize_training_report
except Exception:
    def canonical_landcover_name(name: str | None) -> str | None:
        low = str(name or "").lower()
        if "built" in low:
            return "built"
        if "veget" in low:
            return "vegetation"
        if "water" in low:
            return "water"
        if "other" in low or "bare" in low:
            return "other"
        return None

    def normalize_training_report(report, fallback_feature_names=None, fallback_target_names=None):
        src = dict(report or {})
        raw_targets = list(src.get("targets") or fallback_target_names or [])
        return {
            **src,
            "feature_names": list(src.get("features") or fallback_feature_names or []),
            "raw_target_names": raw_targets,
            "target_names": [canonical_landcover_name(t) or t for t in raw_targets],
            "holdout": src.get("holdout") or src.get("spatial_holdout") or "east",
        }


def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict[str, Any]:
    rows = []
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append(
            {
                "target": name,
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                "mae": float(mean_absolute_error(yt, yp)),
                "r2": float(r2_score(yt, yp)),
            }
        )
    return {"per_target": rows, "rmse_macro": float(np.mean([r["rmse"] for r in rows]))}


def train_and_evaluate(
    holdout: str = "east",
    out_dir: str | Path | None = None,
) -> Dict[str, Any]:
    out_dir = Path(out_dir or PATHS.models)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_df, y_df, cell_ids = composition_forecasting_xy()
    X = X_df.to_numpy(dtype=np.float64)
    y = y_df.to_numpy(dtype=np.float64)
    feature_names = list(X_df.columns)
    target_names = list(y_df.columns)

    train_mask, test_mask = spatial_train_test_mask(cell_ids, holdout=holdout)
    X_tr, X_te = X[train_mask], X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]

    ridge = build_ridge_pipeline()
    ridge.fit(X_tr, y_tr)
    y_hat_r = ridge.predict(X_te)

    gbrt = build_hist_gbrt_pipeline()
    gbrt.fit(X_tr, y_tr)
    y_hat_g = gbrt.predict(X_te)

    report = {
        "task": "forecast_coarse_composition_t1_from_t0",
        "base_year": PREDICTION_BASE_YEAR,
        "target_year": PREDICTION_TARGET_YEAR,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "spatial_holdout": holdout,
        "features": feature_names,
        "targets": target_names,
        "ridge": _per_target_metrics(y_te, y_hat_r, target_names),
        "hist_gbrt": _per_target_metrics(y_te, y_hat_g, target_names),
        "model_notes": {
            "ridge": (
                "Multi-output Ridge on StandardScaler features; linear, globally interpretable "
                "via coefficients (after noting feature scaling)."
            ),
            "hist_gbrt": (
                "MultiOutputRegressor(HistGradientBoostingRegressor): nonlinear, handles thresholds "
                "and interactions; importances exported via permutation_importance on a test subsample."
            ),
        },
    }
    report = normalize_training_report(
        report,
        fallback_feature_names=feature_names,
        fallback_target_names=target_names,
    )

    joblib.dump(ridge, out_dir / "ridge_composition.joblib")
    joblib.dump(gbrt, out_dir / "hist_gbrt_composition.joblib")
    joblib.dump(
        {
            "cell_ids_test": cell_ids[test_mask],
            "y_test": y_te,
            "feature_names": feature_names,
            "holdout": holdout,
        },
        out_dir / "test_fold_meta.joblib",
    )

    coef = ridge_coefficient_table(ridge, feature_names, target_names)
    coef_df = pd.DataFrame({"feature": feature_names})
    for idx, raw_name in enumerate(target_names):
        coef_df[f"coef_{canonical_landcover_name(raw_name) or raw_name}"] = coef[idx]
    coef_df.to_csv(out_dir / "ridge_coefficients.csv", index=False)

    rng = np.random.default_rng(42)
    n_pi = min(1500, X_te.shape[0])
    idx = rng.choice(X_te.shape[0], size=n_pi, replace=False)
    X_sub, y_sub = X_te[idx], y_te[idx]
    imp_rows = []
    for i, (tname, est) in enumerate(zip(target_names, gbrt.estimators_)):
        pi = permutation_importance(
            est,
            X_sub,
            y_sub[:, i],
            n_repeats=8,
            random_state=42,
            n_jobs=1,
        )
        for fname, val in zip(feature_names, pi.importances_mean):
            imp_rows.append({"target": tname, "feature": fname, "importance_mean": float(val)})
    pd.DataFrame(imp_rows).to_csv(out_dir / "hist_gbrt_permutation_importance.csv", index=False)

    with open(out_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    r = train_and_evaluate()
    print(json.dumps({k: r[k] for k in ("n_train", "n_test", "spatial_holdout", "ridge", "hist_gbrt")}, indent=2))
