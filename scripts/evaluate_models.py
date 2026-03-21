#!/usr/bin/env python3
"""Evaluation beyond accuracy: change metrics, built-up event rates, stress tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.models.evaluation import run_evaluation  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--holdout",
        choices=("east", "north"),
        default="east",
        help="Must match the split used when running train_models.py.",
    )
    p.add_argument("--models-dir", default=None, help="Default: artifacts/models")
    args = p.parse_args()

    report = run_evaluation(holdout=args.holdout, models_dir=args.models_dir)
    slim = {
        "holdout": report["holdout"],
        "n_test": report["n_test"],
        "ridge_delta_rmse_macro": report["ridge"]["delta_metrics"]["delta_rmse_macro"],
        "gbrt_delta_rmse_macro": report["hist_gbrt"]["delta_metrics"]["delta_rmse_macro"],
        "ridge_false_built_change": report["ridge"]["built_change_diagnostics"]["built_false_change_rate"],
        "gbrt_false_built_change": report["hist_gbrt"]["built_change_diagnostics"]["built_false_change_rate"],
    }
    print(json.dumps(slim, indent=2))
    out = Path(args.models_dir or PATHS.models) / "evaluation_report.json"
    print(f"Full report: {out.resolve()}")


if __name__ == "__main__":
    main()
