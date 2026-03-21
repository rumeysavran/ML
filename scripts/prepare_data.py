#!/usr/bin/env python3
"""
End-to-end data prep for the Nuremberg assignment:
  1) ESA WorldCover clips (public AWS bucket, HTTP range reads via GDAL /vsicurl/)
  2) Sentinel-2 L2A July composites (Copernicus data via Planetary Computer STAC)
  3) 300 m grid aggregation + tabular feature table (Parquet + GeoPackage grid)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS, S2_YEARS  # noqa: E402
from src.data.worldcover_fetch import fetch_all_worldcover  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Nuremberg land-cover dataset")
    parser.add_argument("--fetch-wc", action="store_true", help="Clip ESA WorldCover to the study bbox")
    parser.add_argument("--fetch-s2", action="store_true", help="Build Sentinel-2 median composites (slow, needs network)")
    parser.add_argument("--features", action="store_true", help="Engineer tabular features (needs S2 + WC rasters)")
    parser.add_argument("--all", action="store_true", help="WorldCover + Sentinel-2 + features")
    args = parser.parse_args()

    Path(PATHS.raw).mkdir(parents=True, exist_ok=True)
    Path(PATHS.processed).mkdir(parents=True, exist_ok=True)

    do_wc = args.all or args.fetch_wc
    do_s2 = args.all or args.fetch_s2
    do_feat = args.all or args.features

    if not (do_wc or do_s2 or do_feat):
        parser.error("Select at least one of --fetch-wc, --fetch-s2, --features, or --all")

    if do_wc:
        paths = fetch_all_worldcover()
        for y, p in paths.items():
            print(f"WorldCover {y}: {p}")

    if do_s2:
        from src.data.sentinel2_fetch import composite_all_years  # noqa: WPS433

        paths = composite_all_years()
        for y, p in paths.items():
            print(f"Sentinel-2 composite {y}: {p}")

    if do_feat:
        from src.features.engineering import run_default_pipeline  # noqa: WPS433

        if do_feat and not do_s2:
            missing = [Path(PATHS.raw) / f"sentinel2_median_{y}_july.tif" for y in S2_YEARS]
            missing = [p for p in missing if not p.exists()]
            if missing:
                print(
                    "WARNING: Sentinel-2 composites missing; run with --fetch-s2 first.\n"
                    f"Missing: {missing}",
                    file=sys.stderr,
                )
        if do_feat and not do_wc:
            wc_missing = any(
                not (Path(PATHS.raw) / f"worldcover_{y}_nuremberg.tif").exists() for y in S2_YEARS
            )
            if wc_missing:
                print("WARNING: WorldCover clips missing; run with --fetch-wc first.", file=sys.stderr)

        df = run_default_pipeline()
        print(f"Feature table rows: {len(df)}  columns: {len(df.columns)}")
        print(f"Written: {Path(PATHS.processed) / 'tabular_features.parquet'}")


if __name__ == "__main__":
    main()
