#!/usr/bin/env python3
"""Train Ridge + HistGradientBoosting models (spatial hold-out) and write artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS  # noqa: E402
from src.models.train import train_and_evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--holdout",
        choices=("east", "north"),
        default="east",
        help="Spatial block axis for the test half of the grid (see src/models/dataset.py).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: artifacts/models from config).",
    )
    args = p.parse_args()

    report = train_and_evaluate(holdout=args.holdout, out_dir=args.out)
    slim = {
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "spatial_holdout": report["spatial_holdout"],
        "ridge_rmse_macro": report["ridge"]["rmse_macro"],
        "gbrt_rmse_macro": report["hist_gbrt"]["rmse_macro"],
    }
    print(json.dumps(slim, indent=2))
    print("Artifacts:", Path(args.out or PATHS.models).resolve())


if __name__ == "__main__":
    main()
