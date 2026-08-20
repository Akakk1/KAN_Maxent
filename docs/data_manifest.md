# Data manifest & evaluation contract

**Primary benchmark:** NCEAS six-region presence-only → independent PA (Elith et al. 2020, **CC BY-NC 4.0**; Valavi et al. 2022). Data under `data/nceas/` are not MIT; see `data/nceas/LICENSE`.

## Estimand

| Field | Value |
|:------|:------|
| Primary estimand | Global conditional density on declared domain |
| Background role | Integration support with weights \(w_j=1\) (not a true absence class) |
| PA labels | Evaluation only — never for tuning or early stopping |
| Main integration | `full_background_50k` per region |

## Regions & layout

Root: `data/nceas/{awt,can,nsw,nz,sa,swi}/`

| Region | Species (valid AUC scale) | Notes |
|:-------|--------------------------:|:------|
| AWT | 40 | bird / plant env-pa splits |
| CAN | 20 | categorical `ontveg` |
| NSW | 53 (+ skips n_PO&lt;5) | multi taxon groups; `vegsys` |
| NZ | 52 | plant; `age`, `toxicats` |
| SA | 30 | plant |
| SWI | 30 | tree; `calc` |

Typical files per region: `{REG}_po.csv`, `{REG}_pa*.csv`, `{REG}_env*.csv`, `{REG}_bg_50k.csv`, `MANIFEST.json`.

## Modelling contract (main tables)

| Item | Value |
|:-----|:------|
| Continuous prep | z-score on PO∪BG |
| Categorical | one-hot + L2 |
| B-spline | G=6, K=3; \(\lambda_s=10^{-2}\), \(\lambda_r=10^{-6}\) (frozen default) |
| Skip | n_PO &lt; 5 |
| maxnet | R maxnet 0.1.4; BG ≤10k subsample; raw env scale |
| Deep fair | residual, LBFGS, no SiLU, warm-start, freeze edges — `docs/fair_deep_protocol.md` |

## Optional / not shipped

- Valavi SI zip and OSF MaxNet prediction dumps (lab backup only).  
- Ginkgo spatial-CV tables are not included in this package.

## Loader

```python
from kanmaxent.data.nceas_io import load_region
data = load_region("CAN")
```
