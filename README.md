# Nuremberg land-cover change (ML final assignment)

Machine-learning pipeline for **mapping urban land-cover change** around Nuremberg using **ESA WorldCover** labels and **Sentinel-2** summer composites. Models operate on **tabular features** only (no end-to-end CNNs), per course rules.

## How to run

Run everything from the **repository root** (the folder that contains `scripts/` and `src/`).

1. **Create a virtual environment and install dependencies** (Python **3.10–3.12**):

   ```bash
   cd /path/to/ML
   python3.12 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download data and build the feature table** (needs **internet** the first time; can take a while):

   ```bash
   python scripts/prepare_data.py --all
   ```

   To run stages separately: `--fetch-wc`, `--fetch-s2`, then `--features` (see below).

3. **Train both models and write metrics + artifacts**:

   ```bash
   python scripts/train_models.py
   ```

   Optional spatial split: `python scripts/train_models.py --holdout north`

Outputs: `data/processed/tabular_features.parquet`, then under `artifacts/models/` see `training_report.json` and the `.joblib` model files.

4. **Evaluation (change metrics + stress tests)** — use the **same `--holdout`** as training:

   ```bash
   python scripts/evaluate_models.py --holdout east
   ```

   Produces `artifacts/models/evaluation_report.json` (δ-RMSE, built false-change / missed-gain rates, Gaussian noise + NIR dropout stress tests, failure bullets). Details: [`docs/evaluation.md`](docs/evaluation.md).

## What this repository does

1. **Data** — Clips **ESA WorldCover** 2020/2021 to a Nuremberg bounding box (public AWS bucket) and builds **July median Sentinel-2 L2A** composites (Copernicus data via [Planetary Computer STAC](https://planetarycomputer.microsoft.com/)).
2. **Features** — Aggregates Sentinel-2 bands and spectral indices to a **300 m** UTM grid; derives **coarse composition** (built / vegetation / water / other) by counting 10 m WorldCover pixels inside each cell.
3. **Models** — Trains two regressors that forecast **next-year coarse composition** from the **previous year’s** imagery + composition:
   - **Ridge regression** (`StandardScaler` + `Ridge`): interpretable linear baseline; coefficients saved to `artifacts/models/ridge_coefficients.csv`.
   - **Histogram gradient boosting** (`MultiOutputRegressor(HistGradientBoostingRegressor)`): nonlinear; permutation importances in `artifacts/models/hist_gbrt_permutation_importance.csv`.
4. **Evaluation** — Spatial hold-out plus **change-specific** metrics (δ-RMSE, built false-change / stability-style rates) and **stress tests** (noisy features, NIR dropout); see `artifacts/models/evaluation_report.json` and [`docs/evaluation.md`](docs/evaluation.md). Baseline composition metrics remain in `training_report.json`.

Non-trivial data issues and the **one issue left deliberately unfixed** are documented in [`docs/data_issues.md`](docs/data_issues.md).

## Environment

Use **Python 3.10–3.12** (recommended: 3.12).

```bash
cd /path/to/ML
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Prepare data

Requires network access for first-time downloads.

```bash
python scripts/prepare_data.py --all
```

Stages:

- `--fetch-wc` — WorldCover clips → `data/raw/worldcover_*_nuremberg.tif`
- `--fetch-s2` — Sentinel-2 composites → `data/raw/sentinel2_median_*_july.tif`
- `--features` — Grid + Parquet → `data/processed/tabular_features.parquet`, `data/processed/grid_cells.gpkg`

## Train the two models

```bash
python scripts/train_models.py
# optional: python scripts/train_models.py --holdout north
```

Writes to `artifacts/models/`:

| File | Purpose |
|------|---------|
| `ridge_composition.joblib` | Fitted interpretable pipeline |
| `hist_gbrt_composition.joblib` | Fitted boosting ensemble |
| `ridge_coefficients.csv` | Linear coefficients (scaled feature space) |
| `hist_gbrt_permutation_importance.csv` | Approx. importances (permutation on a test subsample) |
| `training_report.json` | Split sizes, metrics, feature/target names |
| `test_fold_meta.joblib` | Test `cell_id`s, `y_test`, and `holdout` (after a fresh train) |

## Evaluate (beyond accuracy)

```bash
python scripts/evaluate_models.py --holdout east   # must match train_models --holdout
```

Writes **`evaluation_report.json`**: δ-RMSE on composition change, built-up false-change / missed-gain / stability-style metrics, and stress tests (Gaussian noise + NIR dropout). Methodology: [`docs/evaluation.md`](docs/evaluation.md).

## Task definition (for the report)

- **Spatial unit:** 300 m grid cells (`EPSG:32632`), `cell_id` encodes row-major order (`reference_grid()` in `src/geo/grid.py`).
- **Temporal setup:** Features from **2020**; targets are **2021** coarse composition proportions (built, vegetation, water, other).
- **Feature rationale:** Sentinel-2 reflectance and indices capture surface state; **2020 WorldCover composition** encodes the current land-cover mix as a strong baseline for short-term forecasting (acknowledge label noise and v100→v200 artefacts in the report).
- **Model rationale:** Ridge satisfies the **interpretable model** requirement; histogram GBRT is a **flexible nonlinear** tabular model allowed by the assignment (alternatives: random forest, MLP).

## Project layout

```
scripts/
  prepare_data.py      # download + feature table
  train_models.py      # train + evaluate + save artifacts
src/
  config.py            # bbox, years, paths, feature lists
  data/                # WorldCover + Sentinel-2 I/O
  geo/                 # grid / CRS helpers
  features/            # spectral indices + composition aggregation
  models/              # dataset split, pipelines, training
docs/
  data_issues.md       # exploration issues + unfixed choice
data/raw/              # GeoTIFF inputs (gitignored)
data/processed/        # Parquet / GPKG (gitignored)
artifacts/models/      # trained models + reports (gitignored)
```

## Data citation (minimum)

- **ESA WorldCover** — [Product & DOI](https://esa-worldcover.org/en/data-access) (e.g. 2021 v200 Zenodo DOI in registry); accessed via AWS Open Data `esa-worldcover` bucket.
- **Sentinel-2** — Copernicus programme; this code uses **Microsoft Planetary Computer** as a STAC/API access path to the same L2A product.

## Next steps (remaining coursework)

- Stress tests, change-specific metrics, and failure discussion (evaluation section).
- Explainability UI: helpful vs **misleading** explanations for non-experts.
- Two **“Arguing Against ChatGPT”** write-ups and usage log.
- **Streamlit / Gradio** app: map, year selection, predictions, uncertainty/limitations.

## License note

Respect **WorldCover CC-BY-4.0** and Copernicus/Sentinel terms when redistributing derivatives.
