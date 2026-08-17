# NCEAS Phase 3 Full Benchmark SUMMARY

**Run:** `nceas_full_20260714_phase3`
**Protocol:** PO train → independent PA test (Valavi); main background = `random_50k`; `integration_support=full_background_50k`; PA never used for tuning.
**Scale:** 226 species × 4 models × 1 seed = 904 metric rows (225 with valid AUC).

## 1. Mean AUC by model (main, random_50k)

- **additive_kan_bce**: 0.7163 ± 0.1337 (n=225)
- **additive_kan_ipp**: 0.7223 ± 0.1392 (n=225)
- **gam_ipp**: 0.7224 ± 0.1393 (n=225)
- **maxnet**: 0.7138 ± 0.1465 (n=225)

## 2. Mean AUC by region × model

| region | additive_kan_bce | additive_kan_ipp | gam_ipp | maxnet |
| --- | --- | --- | --- | --- |
| AWT | 0.6605 | 0.6681 | 0.6680 | 0.6775 |
| CAN | 0.6085 | 0.6005 | 0.6005 | 0.5813 |
| NSW | 0.7093 | 0.7082 | 0.7088 | 0.7069 |
| NZ | 0.7224 | 0.7381 | 0.7383 | 0.7288 |
| SA | 0.7810 | 0.7920 | 0.7918 | 0.7620 |
| SWI | 0.7994 | 0.8034 | 0.8034 | 0.7882 |

## 3. Paired ΔAUC (species-level; bootstrap 95% CI, 1000 resamples)

- **KAN-GAM**: mean Δ=-0.0002 [-0.0004, +0.0000], |Δ|mean=0.0006, r=0.9999, %Δ>0=44.4% (n=225)
- **KAN-maxnet**: mean Δ=+0.0085 [+0.0015, +0.0156], |Δ|mean=0.0378, r=0.9340, %Δ>0=52.9% (n=225)
- **IPP-BCE**: mean Δ=+0.0060 [+0.0018, +0.0100], |Δ|mean=0.0236, r=0.9717, %Δ>0=58.2% (n=225)

### 3.1 By region

**KAN-GAM**

| region | n | Δmean | CI_lo | CI_hi | r |
| --- | --- | --- | --- | --- | --- |
| AWT | 40 | 0.0000 | -0.0002 | 0.0003 | 1.0000 |
| CAN | 20 | 0.0000 | -0.0000 | 0.0001 | 1.0000 |
| NSW | 53 | -0.0007 | -0.0016 | 0.0001 | 0.9998 |
| NZ | 52 | -0.0002 | -0.0004 | 0.0001 | 1.0000 |
| SA | 30 | 0.0002 | -0.0001 | 0.0004 | 1.0000 |
| SWI | 30 | -0.0000 | -0.0002 | 0.0002 | 1.0000 |

**KAN-maxnet**

| region | n | Δmean | CI_lo | CI_hi | r |
| --- | --- | --- | --- | --- | --- |
| AWT | 40 | -0.0095 | -0.0254 | 0.0061 | 0.9282 |
| CAN | 20 | 0.0192 | 0.0033 | 0.0384 | 0.9667 |
| NSW | 53 | 0.0013 | -0.0137 | 0.0166 | 0.9175 |
| NZ | 52 | 0.0094 | -0.0058 | 0.0264 | 0.9128 |
| SA | 30 | 0.0299 | 0.0132 | 0.0479 | 0.9165 |
| SWI | 30 | 0.0151 | 0.0087 | 0.0223 | 0.9841 |

**IPP-BCE**

| region | n | Δmean | CI_lo | CI_hi | r |
| --- | --- | --- | --- | --- | --- |
| AWT | 40 | 0.0076 | -0.0040 | 0.0199 | 0.9465 |
| CAN | 20 | -0.0081 | -0.0190 | 0.0021 | 0.9815 |
| NSW | 53 | -0.0012 | -0.0104 | 0.0086 | 0.9702 |
| NZ | 52 | 0.0157 | 0.0061 | 0.0255 | 0.9631 |
| SA | 30 | 0.0109 | 0.0012 | 0.0211 | 0.9700 |
| SWI | 30 | 0.0039 | 0.0013 | 0.0068 | 0.9965 |

### 3.2 By region × taxon_group (KAN−GAM)

| region | taxon_group | n | Δmean | CI_lo | CI_hi |
| --- | --- | --- | --- | --- | --- |
| AWT | bird | 20 | 0.0002 | -0.0000 | 0.0004 |
| AWT | plant | 20 | -0.0001 | -0.0005 | 0.0003 |
| NSW | ba | 7 | 0.0001 | -0.0008 | 0.0013 |
| NSW | db | 8 | 0.0001 | -0.0002 | 0.0004 |
| NSW | nb | 2 | 0.0000 | -0.0001 | 0.0002 |
| NSW | ot | 8 | -0.0031 | -0.0076 | 0.0002 |
| NSW | ou | 7 | -0.0001 | -0.0004 | 0.0001 |
| NSW | rt | 7 | -0.0018 | -0.0054 | 0.0008 |
| NSW | ru | 6 | 0.0001 | -0.0001 | 0.0005 |
| NSW | sr | 8 | 0.0000 | -0.0001 | 0.0002 |

## 4. Valavi OSF MaxNet alignment (DoD: per-region r > 0.90)

| region | n | Pearson r | missing_files | pass |
| --- | --- | --- | --- | --- |
| AWT | 40 | 0.9439 | 0 | True |
| CAN | 20 | 0.9835 | 0 | True |
| NSW | 53 | 0.9654 | 1 | True |
| NZ | 52 | 0.9394 | 0 | True |
| SA | 30 | 0.9101 | 0 | True |
| SWI | 30 | 0.9717 | 0 | True |

Missing OSF pred files: NSW:['nsw30']

## 5. TGB sensitivity (NSW, ≥10 species)

- **Species (n=12):** nsw04, nsw06, nsw09, nsw14, nsw16, nsw17, nsw18, nsw24, nsw28, nsw39, nsw43, nsw52
- **Definition:** other PO sites in same taxon group as background; excludes focal presence; `background_scheme=tgb`; `integration_support=tgb_other_po_same_group`
- **Main table remains random_50k**; TGB does not replace estimand for global density claims

### 5.1 Mean AUC: random_50k vs TGB (same species)

| model | n_paired | mean AUC random | mean AUC TGB | Δ (TGB−random) | CI_lo | CI_hi |
| --- | --- | --- | --- | --- | --- | --- |
| additive_kan_ipp | 12 | 0.6542 | 0.6514 | -0.0029 | -0.0466 | 0.0435 |
| gam_ipp | 12 | 0.6542 | 0.6511 | -0.0030 | -0.0469 | 0.0434 |
| additive_kan_bce | 12 | 0.6536 | 0.6426 | -0.0110 | -0.0509 | 0.0294 |
| maxnet | 5 | 0.6414 | 0.6920 | 0.0609 | 0.0105 | 0.1135 |

- maxnet TGB failures: **7/12** species (glmnet unstable on small TGB sets; KAN/GAM/BCE still produce scores).
- Artifacts: `metrics_tgb.csv`, `tgb_vs_random50k.csv`, `tgb_species_detail.csv`

**Readout:** On this NSW sample, KAN-IPP / GAM mean AUC under TGB is nearly unchanged vs random_50k (Δ≈−0.003); BCE slightly lower; maxnet only comparable on the subset that converged. Supports treating TGB as sensitivity, not main protocol.

## 6. Skips / failures (main)

- NSW/nsw30: n_po<5

## 7. Protocol notes

- λs: freeze 1e-2 when n_po<30 (default); PO 4-fold optional
- maxnet: BG subsample ≤10k; classes l/lq fallback
- Main estimand: `global_conditional_density` on `random_50k`
- KAN vs GAM: global r≈0.9999 (expected near-equivalence under same basis)

*Plan_Phase3_v2 DoD blocks D–G.*