# Phase 4 Fair Deep-2 — Six-region multi-seed SUMMARY

**Run:** `deepkan_fair_six_3seed_20260715`
**Protocol:** residual + warm-start additive + LBFGS + disable SiLU
**Scale:** 6 regions, 225 species, seeds=[0, 1, 2], 1356 metric rows, 225 species-level paired means

## Global (species-level mean over seeds)

- mean ΔAUC = **-0.0054** [-0.0098, -0.0009]
- median Δ = -0.0032
- n(Δ>0.01) = 50/225; n(Δ<−0.01) = 75/225

## By region (species means)

| region | n_sp | mean Δ | CI_lo | CI_hi | n(Δ>0.01) | mean seeds |
|:-------|-----:|-------:|------:|------:|----------:|-----------:|
| AWT | 40 | +0.0040 | -0.0067 | +0.0152 | 11 | 3.00 |
| CAN | 20 | +0.0046 | -0.0068 | +0.0204 | 2 | 3.00 |
| NSW | 53 | -0.0095 | -0.0210 | +0.0015 | 16 | 3.00 |
| NZ | 52 | -0.0132 | -0.0233 | -0.0041 | 10 | 3.00 |
| SA | 30 | -0.0069 | -0.0216 | +0.0052 | 10 | 3.00 |
| SWI | 30 | -0.0021 | -0.0047 | +0.0004 | 1 | 3.00 |

## Multi-seed stability

- mean within-species SD(Δ) = 0.0044
- median within-species SD(Δ) = 0.0022

## Mean AUC by model (all rows)

- **additive_kan_ipp**: 0.7222 ± 0.1390 (n=675)
- **deep2_kan_ipp**: 0.7169 ± 0.1351 (n=675)

## Decision

- **No systematic interaction gain.** Global mean Δ is small and slightly negative
  (≈ −0.005; 95% CI excludes 0 but effect size is tiny vs Phase 3 between-model gaps).
- Multi-seed SD(Δ) ≈ 0.004 → ranking stable across seeds.
- Positive Δ>0.01 on 50/225 species, negative on 75/225 — heterogeneous, not a global win.
- Paper claim: **six regions, 3 seeds, fair residual Deep-2 → no systematic gain over additive**.

*Fair protocol: residual / warm-start / LBFGS / no-SiLU / freeze-edges (Φ only).*