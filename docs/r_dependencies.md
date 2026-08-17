# R dependencies (maxnet baseline)

Required only for **R maxnet** rows in Ginkgo CV and NCEAS benchmarks.
Additive B-spline IPP / GAM-IPP / Deep ablation do **not** need R.

## Versions used in published runs

| Package | Version | Role |
|:--------|:--------|:-----|
| R | system `Rscript` | subprocess bridge |
| **maxnet** | **0.1.4** | MaxEnt-style GLMNet features |
| glmnet | 4.x / 5.x (maxnet dep) | |
| **disdat** | **1.1.0** | optional: original NCEAS fetch |

## Install (example)

```r
install.packages(c("maxnet", "jsonlite"))
# optional for data provenance:
# install.packages("disdat")
```

## Bridge

Python calls `Rscript` via:

- `kanmaxent/reference/maxnet_r.py` (Ginkgo-style)
- `kanmaxent/reference/maxnet_nceas.py` (factors + class by n_po)

Ensure `Rscript` is on `PATH`. No `rpy2` required.
