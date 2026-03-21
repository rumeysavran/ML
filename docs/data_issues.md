# Data exploration — non-trivial issues (assignment §2)

This note satisfies the requirement to **identify at least three non-trivial data issues** on engineered features, ESA WorldCover labels, and derived change labels, and to **explicitly leave one issue unfixed** with justification.

## Issue 1 — Algorithm change between WorldCover 2020 (v100) and 2021 (v200)

**What we observe:** ESA’s own open-data documentation states that the 2020 and 2021 maps were produced with **different algorithm versions**, so year-to-year differences mix **real land-cover change** with **product-version change** (classification shifts that are not geographic change).

**Why it matters for ML:** Any model trained to predict “change” between 2020 and 2021 can learn **spurious corrections** in stable areas (e.g. cropland vs grassland swaps) that are artefacts of the classifier, not urban processes.

**Mitigation in this repo (partial):** We aggregate to coarse groups (built / vegetation / water / other) to dampen fine-class flips, and we document the limitation in the report instead of treating deltas as pure physical ground truth.

---

## Issue 2 — Single-month (July) composites and seasonal confusion

**What we observe:** Sentinel-2 features are built from **July** stacks only (`S2_MONTH = 7`) to limit seasonal variability, but agriculture and deciduous vegetation still exhibit **intra-month phenology** (crop stage, mowing, harvest windows). A median composite reduces cloud noise but **does not remove** all seasonal effects.

**Why it matters for ML:** Spectral indices (NDVI, NDMI-family) can shift between years because of **calendar weather**, not land-cover change, especially on cropland and grassland fringes around Nuremberg.

**Mitigation in this repo (partial):** July is held fixed across years; the report should stress that **change on agricultural cells is ambiguous** without multi-month features or crop calendars.

---

## Issue 3 — Spatial autocorrelation and non-independent grid cells

**What we observe:** Grid cells are **geographically contiguous**. Neighbouring 300 m cells share similar atmosphere, illumination geometry, soil background, and urban texture. Standard random train/test splits leak information across space (model sees “copies” of the same landscape patch).

**Why it matters for ML:** Validation accuracy can be **optimistically biased**, and models may learn **local texture shortcuts** that fail when extrapolating across the city.

**Mitigation in this repo (partial):** The evaluation section of the coursework should use **spatial or temporal blocking** (assignment requirement). The codebase keeps `cell_id` and a `grid_cells.gpkg` geometry layer so you can build spatial folds (e.g. by district or by north/south blocks).

---

## Issue we deliberately do *not* fully fix (with justification)

**Choice: spatial autocorrelation (Issue 3).**

**Why not “fix” it in the data pipeline:** Removing spatial dependence entirely would require either **aggregating to very coarse units** (losing the assignment’s goal of a detailed Nuremberg map) or **substantial thinning / complex graph-based splitting** that would **discard most training pixels** in a small city AOI. That trade-off would make the interactive map less useful for the final product.

**What we do instead:** We **surface** the issue in evaluation (blocked splits, error maps) and in the “limitations” panel of the app, so users do not treat pixel-level metrics as equivalent to independent samples.

---

## Quick quantitative hooks (optional checks)

After running `python scripts/prepare_data.py --all`, you can load `data/processed/tabular_features.parquet` and examine:

- **`label_wc_valid_pixels`**: low values flag cells where few 10 m WorldCover samples fell inside the 300 m square (edge / resampling effects).
- **Correlation of NDVI / NDBI between neighbouring cells** (join on spatial neighbours from `grid_cells.gpkg`): high correlation is evidence of autocorrelation.
- **Cloud screening**: STAC metadata (`eo:cloud_cover`) is only a **scene-level** proxy; thin cirrus and haze can remain in July medians — tie this to **residual noise in water / vegetation indices**.
