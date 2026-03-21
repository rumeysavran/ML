"""Stress tests: noisy inputs and missing-band style dropout (assignment §4)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import FEATURE_COLUMNS_T0


def _train_stds(X_tr: np.ndarray) -> np.ndarray:
    return np.std(X_tr, axis=0, ddof=0).clip(1e-9, None)


def gaussian_feature_noise(
    X_te: np.ndarray,
    X_tr: np.ndarray,
    rng: np.random.Generator,
    sigma_mult: float,
) -> np.ndarray:
    """Add N(0, (sigma_mult * std_train_j)^2) per column."""
    sig = _train_stds(X_tr)
    noise = rng.normal(0.0, 1.0, size=X_te.shape).astype(np.float64)
    noise *= sigma_mult * sig
    return X_te + noise


def zero_out_feature(X: np.ndarray, feature_name: str, fill: np.ndarray) -> np.ndarray:
    """Replace one column with a constant column (e.g. train median) — missing-band proxy."""
    out = X.copy()
    j = FEATURE_COLUMNS_T0.index(feature_name)
    out[:, j] = fill[j]
    return out


def stress_test_suite(
    predict_fn,
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    macro_rmse_fn,
    rng: np.random.Generator,
    noise_levels: Tuple[float, ...] = (0.5, 1.0),
    dropout_feature: str = "s2_b08",
) -> Dict[str, Any]:
    """
    ``predict_fn(X) -> y_hat``; ``macro_rmse_fn(y_true, y_pred) -> float``.
    """
    baseline = float(macro_rmse_fn(y_te, predict_fn(X_te)))
    rows: List[Dict[str, Any]] = [{"name": "baseline", "macro_rmse": baseline, "delta_vs_baseline": 0.0}]

    for mult in noise_levels:
        Xn = gaussian_feature_noise(X_te, X_tr, rng, sigma_mult=mult)
        rmse = float(macro_rmse_fn(y_te, predict_fn(Xn)))
        rows.append(
            {
                "name": f"gaussian_noise_sigma_{mult}x_train_std",
                "macro_rmse": rmse,
                "delta_vs_baseline": float(rmse - baseline),
            }
        )

    med = np.median(X_tr, axis=0)
    Xd = zero_out_feature(X_te, dropout_feature, fill=med)
    rmse_d = float(macro_rmse_fn(y_te, predict_fn(Xd)))
    rows.append(
        {
            "name": f"dropout_{dropout_feature}_filled_train_median",
            "macro_rmse": rmse_d,
            "delta_vs_baseline": float(rmse_d - baseline),
        }
    )

    return {"rows": rows, "baseline_macro_rmse": baseline}
