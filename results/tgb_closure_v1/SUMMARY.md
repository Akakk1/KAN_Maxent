# TGB Closure v1 — SUMMARY

**Protocol:** `tgb_closure_v1`
**Region / species:** NSW, n = 12 pre-registered species
**λ\* source:** `outputs/methodological_closure_full_v1/lambda_star.csv` (species-level, frozen for both arms)
**Arms:** `random_50k` (full background_50k) vs `tgb` (other PO same taxon_group, max 50k)
**Models:** additive_kan_ipp, gam_ipp_same_basis, additive_kan_bce, maxnet_bg10k, maxnet_bg50k
**Generated (UTC):** 20260809T161426Z

## Species list and λ\*

| species | n_PO | λ\* | path |
|:--------|-----:|----:|:-----|
| nsw04 | 49 | 0.0001 | po_random_5fold |
| nsw06 | 44 | 1 | po_random_5fold |
| nsw09 | 426 | 1 | po_random_5fold |
| nsw14 | 315 | 0.01 | po_random_5fold |
| nsw16 | 120 | 1 | po_random_5fold |
| nsw17 | 148 | 0.1 | po_random_5fold |
| nsw18 | 69 | 0.1 | po_random_5fold |
| nsw24 | 68 | 1 | po_random_5fold |
| nsw28 | 53 | 1 | po_random_5fold |
| nsw39 | 16 | 0.01 | frozen_n_po_lt_30 |
| nsw43 | 42 | 0.1 | po_random_5fold |
| nsw52 | 186 | 1 | po_random_5fold |

## Model-level paired ΔAUC (TGB − random)

### Pair rule: `finite_pair`

| model | n | mean AUC random | mean AUC TGB | mean Δ | 95% CI |
|:------|--:|----------------:|-------------:|-------:|:-------|
| additive_kan_bce | 12 | 0.6565 | 0.6469 | -0.0096 | [-0.0506, +0.0296] |
| additive_kan_ipp | 12 | 0.6532 | 0.6522 | -0.0010 | [-0.0432, +0.0432] |
| gam_ipp_same_basis | 12 | 0.6532 | 0.6517 | -0.0014 | [-0.0436, +0.0429] |
| maxnet_bg10k | 5 | 0.6311 | 0.6920 | +0.0609 | [+0.0105, +0.1135] |
| maxnet_bg50k | 5 | 0.6363 | 0.6920 | +0.0557 | [+0.0071, +0.1066] |

### Pair rule: `both_converged`

| model | n | mean AUC random | mean AUC TGB | mean Δ | 95% CI |
|:------|--:|----------------:|-------------:|-------:|:-------|
| additive_kan_bce | 12 | 0.6565 | 0.6469 | -0.0096 | [-0.0506, +0.0296] |
| additive_kan_ipp | 12 | 0.6532 | 0.6522 | -0.0010 | [-0.0432, +0.0432] |
| gam_ipp_same_basis | 12 | 0.6532 | 0.6517 | -0.0014 | [-0.0436, +0.0429] |
| maxnet_bg10k | 5 | 0.6311 | 0.6920 | +0.0609 | [+0.0105, +0.1135] |
| maxnet_bg50k | 5 | 0.6363 | 0.6920 | +0.0557 | [+0.0071, +0.1066] |

## maxnet convergence

- random_50k maxnet@10k: 12/12 converged
- tgb maxnet@10k: 5/12 converged
  - failed: nsw04, nsw06, nsw18, nsw24, nsw28, nsw39, nsw43

## TGB background size (per species)

- nsw04: n_bg = 138
- nsw06: n_bg = 143
- nsw09: n_bg = 1087
- nsw14: n_bg = 1198
- nsw16: n_bg = 148
- nsw17: n_bg = 120
- nsw18: n_bg = 270
- nsw24: n_bg = 271
- nsw28: n_bg = 114
- nsw39: n_bg = 48
- nsw43: n_bg = 68
- nsw52: n_bg = 489

## Files

- `metrics.csv` — long format, all models × both arms
- `species_paired_delta.csv` — species×model random/TGB/Δ
- `model_summary.csv` — model-level means and bootstrap CIs
- `manifest.json` — protocol fingerprint
