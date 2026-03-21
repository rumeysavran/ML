# Evaluation (coursework §4)

This project evaluates models **beyond accuracy** using (1) a **spatial hold-out**, (2) **change-specific** metrics, (3) **stress tests**, and (4) a concise **failure discussion** (also embedded in `artifacts/models/evaluation_report.json`).

## 1. Spatial / temporal strategy

- **Spatial:** Train on one half of the grid, test on the other (`east` = test on eastern columns, or `north` for rows). Implemented in `src/models/dataset.py`.
- **Temporal:** Features are fixed to year **t₀** (2020); targets are composition at **t₁** (2021). That is a one-year **forecasting** setup, not random mixing of years.

## 2. Change-specific metrics

Let **c⁰** be coarse composition from the last four columns of **X** (2020 WorldCover fractions) and **c¹** the observed 2021 fractions.

- **Δ_true = c¹ − c⁰**, **Δ_pred = ĉ¹ − c⁰** (using predicted ĉ¹).
- Report **δ-RMSE / δ-MAE** per class and a **macro-averaged δ-RMSE** (`delta_rmse_macro` in JSON).
- **Built-focused rates** (tunable thresholds in `change_metrics.py`):
  - **False change rate:** among cells with small true |Δbuilt|, how often does the model predict a large positive Δbuilt?
  - **Missed gain rate:** among cells with real built-up gain, how often does the model under-react?
  - **Stability precision:** when the model predicts “no built change”, how often is that correct?

These address “**false change**” and **stability** wording from the assignment.

## 3. Stress tests

Implemented in `src/models/stress_tests.py` and run for **both** models:

1. **Gaussian noise** on test inputs, scaled by **training-set standard deviation** (×0.5 and ×1.0).
2. **Band dropout:** replace **NIR (`s2_b08`)** with the **training median** (proxy for missing or unreliable band).

The JSON records **macro composition RMSE** after each perturbation and **Δ vs baseline** so you can quote robustness in the report.

## 4. Where the model is likely wrong

See the `where_models_are_likely_wrong` list in `evaluation_report.json` and expand in your own words with **maps** (residuals by cell) for the final submission.

## Commands

```bash
# After training with the same holdout:
python scripts/train_models.py --holdout east
python scripts/evaluate_models.py --holdout east
```

Read: `artifacts/models/evaluation_report.json` (full detail) and `artifacts/models/training_report.json` (composition-only metrics from training).
