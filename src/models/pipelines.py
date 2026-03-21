"""Sklearn pipelines: one interpretable linear model and one nonlinear ensemble."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_pipeline(**ridge_kw: Any) -> Pipeline:
    """
    Interpretable model (assignment requirement): multi-output Ridge regression on
    standardized features. Coefficients map directly to human-readable inputs after scaling.
    """
    kwargs = {"alpha": 1.0}
    kwargs.update(ridge_kw)
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", Ridge(**kwargs)),
        ]
    )


def build_hist_gbrt_pipeline(**hgb_kw: Any) -> MultiOutputRegressor:
    """
    Flexible nonlinear model: histogram-based gradient boosting per target.
    Captures threshold effects and interactions without hand-crafted cross-terms.
    """
    kwargs: Dict[str, Any] = {
        "max_depth": 8,
        "learning_rate": 0.06,
        "max_iter": 250,
        "min_samples_leaf": 20,
        "l2_regularization": 1e-3,
        "random_state": 42,
    }
    kwargs.update(hgb_kw)
    base = HistGradientBoostingRegressor(**kwargs)
    return MultiOutputRegressor(base)


def ridge_coefficient_table(pipeline: Pipeline, feature_names: list[str], target_names: list[str]) -> np.ndarray:
    """Return (n_targets, n_features) matrix of coefficients in *standardized* feature space."""
    ridge: Ridge = pipeline.named_steps["model"]
    coef = np.asarray(ridge.coef_)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return coef
