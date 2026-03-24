# Nuremberg Land-Cover Change Monitor

Machine learning pipeline for mapping urban land-cover change around Nuremberg using ESA WorldCover and Sentinel-2 data.

Includes a full training pipeline and an interactive Streamlit dashboard.

Models use tabular features only. No CNNs. 

---

## Overview

This project builds a complete workflow:

* Collect satellite and land-cover data
* Generate structured features
* Train predictive models
* Evaluate land-cover change
* Visualize results in a dashboard

---

## Quick Start

Run from the repository root.

1. Set up environment

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data

```
python scripts/prepare_data.py --all
```

3. Train models

```
python scripts/train_models.py
```

4. Evaluate models

```
python scripts/evaluate_models.py --holdout east
```

5. Launch dashboard

```
streamlit run app.py
```

The app runs even without artifacts. It uses demo data if needed.

---

## Pipeline Summary

### Data

* ESA WorldCover 2020 and 2021
* Sentinel-2 L2A July composites
* Data clipped to Nuremberg

### Features

* Aggregated to 300 m grid
* Spectral indices from Sentinel-2
* Composition ratios from WorldCover

### Models

* Ridge regression

  * Interpretable baseline
  * Outputs coefficients

* Histogram Gradient Boosting

  * Nonlinear model
  * Uses permutation importance

### Prediction

* One-step inference: predict 2021 composition from 2020 features
* Autoregressive forecasting: multi-year future projections
* Uncertainty quantification: estimates confidence bounds per timestep
* Spectral inputs held constant (assumes future imagery similar to base year)

### Evaluation

* Spatial holdout split
* Change-specific metrics
* Stress tests

  * Noise injection
  * NIR dropout

### Dashboard

* Interactive Streamlit app
* Map, change view, evaluation, explainability

---

## Pipeline Steps

### 1. Prepare Data

```
python scripts/prepare_data.py --all
```

Outputs:

* Raw GeoTIFF files
* Processed feature table
* Grid geometry

---

### 2. Train Models

```
python scripts/train_models.py
```

Optional:

```
python scripts/train_models.py --holdout north
```

Outputs:

* Trained models
* Metrics report
* Feature importance files

---

### 3. Evaluate

```
python scripts/evaluate_models.py --holdout east
```

Outputs:

* Change metrics
* Stress test results

---

### 4. Run Dashboard

```
streamlit run app.py
```

---

## Results Snapshot

Latest run with east holdout:

* Ridge RMSE: 0.0156
* HistGBRT RMSE: 0.0164
* Ridge false-change: 5.1%
* HistGBRT false-change: 7.6%

Key points:

* Ridge performs slightly better overall
* Boosting captures nonlinear patterns but does not win here
* Results remain optimistic due to same-city setup

---

## Dashboard Structure

The app contains multiple views:

* Map

  * Composition per grid cell
  * Predicted change

* Data Exploration

  * Feature distributions
  * Correlations
  * Data issues

* ML Pipeline

  * Model structure
  * Feature importance

* Evaluation

  * Metrics
  * Error analysis
  * Stress tests

* Explainability

  * Clear vs misleading explanations
  * Model limitations

* Raw Imagery

  * Satellite RGB and NDVI
  * WorldCover overlays

* Change Map

  * Pixel-level ground truth
  * Transition matrix

* Forecast

  * Multi-year autoregressive prediction
  * Interactive horizon and model selection
  * Uncertainty bands per land-cover class
  * Export predictions as CSV

---

## Project Structure

```
app.py
scripts/
  prepare_data.py
  train_models.py
  evaluate_models.py
src/
  config.py
  data/
  geo/
  features/
  models/
docs/
  data_issues.md
  evaluation.md
data/
  raw/
  processed/
artifacts/
  models/
```

---

## Prediction & Forecasting

The app includes both inference and forecasting capabilities:

### One-Step Prediction

Predicts 2021 land-cover composition from 2020 features using trained models.
Outputs are displayed in the **Map** tab and integrated into the grid cells.

### Autoregressive Forecast

The **Forecast** tab enables multi-year projections:

* Select forecast horizon (1–10 years)
* Choose active model (Ridge or HistGradientBoosting)
* View predicted composition trajectories per class
* See uncertainty estimates that grow with forecast length
* Export predictions as CSV for further analysis

**Important:** Forecasting assumes constant spectral inputs (Sentinel-2 bands held at base year). This is a simplification; future real spectral data may differ.

---

## Task Definition

* Spatial unit: 300 m grid cells
* Input: 2020 features
* Target: 2021 composition

Classes:

* Built
* Vegetation
* Water
* Other

---

## Data Sources

* ESA WorldCover
* Sentinel-2 L2A via Planetary Computer

Follow licensing terms for redistribution.

---

## Notes on Data Quality

* Label differences between WorldCover versions introduce noise
* Some change signals reflect labeling artifacts
* This issue remains intentionally unresolved for analysis