"""Change-oriented metrics beyond plain composition accuracy (assignment §4)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import FEATURE_COLUMNS_T0, TARGET_COLUMNS_COMPOSITION_T1


def _composition_t0_from_X(X: np.ndarray) -> np.ndarray:
    """Last four columns of X are t0 coarse composition (same order as targets)."""
    idx = [FEATURE_COLUMNS_T0.index(c) for c in TARGET_COLUMNS_COMPOSITION_T1]
    return X[:, idx]


def delta_arrays(y_t1: np.ndarray, y_pred: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """True and predicted year-to-year composition deltas (t1 − t0)."""
    c0 = _composition_t0_from_X(X)
    return y_t1 - c0, y_pred - c0


def delta_rmse_macro(delta_true: np.ndarray, delta_pred: np.ndarray, names: List[str]) -> Dict[str, Any]:
    per = []
    macro = []
    for i, name in enumerate(names):
        dt = delta_true[:, i]
        dp = delta_pred[:, i]
        rmse = float(np.sqrt(mean_squared_error(dt, dp)))
        mae = float(mean_absolute_error(dt, dp))
        per.append({"target": name, "delta_rmse": rmse, "delta_mae": mae})
        macro.append(rmse)
    return {"per_target": per, "delta_rmse_macro": float(np.mean(macro))}


def built_change_event_metrics(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    built_index: int = 0,
    true_stable_thresh: float = 0.02,
    pred_change_thresh: float = 0.03,
    true_gain_thresh: float = 0.04,
) -> Dict[str, Any]:
    """
    Built-up focused diagnostics (urban change narrative).

    - **false_change_rate**: among cells with negligible true Δbuilt, fraction where the model
      predicts a large positive Δbuilt (spurious urbanisation signal).
    - **missed_gain_rate**: among cells with real built-up gain, fraction where the model
      under-predicts gain (missed densification / expansion).
    - **stability_precision**: among cells predicted as stable (|Δpred| < pred_stable), fraction
      that are truly stable (|Δtrue| < true_stable_thresh).
    """
    dt = delta_true[:, built_index]
    dp = delta_pred[:, built_index]

    stable_true = np.abs(dt) < true_stable_thresh
    n_stable = int(stable_true.sum())
    false_change = stable_true & (dp > pred_change_thresh)
    false_change_rate = float(false_change.sum() / max(n_stable, 1))

    real_gain = dt > true_gain_thresh
    n_gain = int(real_gain.sum())
    missed = real_gain & (dp < true_gain_thresh * 0.5)
    missed_gain_rate = float(missed.sum() / max(n_gain, 1))

    pred_stable = np.abs(dp) < true_stable_thresh
    n_pred_stab = int(pred_stable.sum())
    stab_correct = pred_stable & stable_true
    stability_precision = float(stab_correct.sum() / max(n_pred_stab, 1))

    return {
        "built_false_change_rate": false_change_rate,
        "built_missed_gain_rate": missed_gain_rate,
        "built_stability_precision": stability_precision,
        "n_true_stable_cells": n_stable,
        "n_true_built_gain_cells": n_gain,
        "n_pred_stable_cells": n_pred_stab,
        "thresholds": {
            "true_stable_abs_delta": true_stable_thresh,
            "pred_spurious_positive_delta": pred_change_thresh,
            "true_built_gain": true_gain_thresh,
        },
    }
