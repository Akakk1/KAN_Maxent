# Phase 4 Deep KAN Ablation SUMMARY

**Run:** `deepkan_deep3_can20_fair_20260715`
**Architecture:** residual η = Σφ + mixer(φ); deep2: Φ:P→1; deep3: [P, h=4, 1] (grid frozen, no SiLU)
**Protocol:** PO + random_50k → independent PA (same as Phase 3)

## Mean AUC

- **additive_kan_ipp**: 0.6005 ± 0.1324 (n=20)
- **deep2_kan_ipp**: 0.6047 ± 0.1183 (n=20)
- **deep3_kan_ipp**: 0.5846 ± 0.1183 (n=20)

## Paired ΔAUC (deep2 − additive)

- mean Δ = **+0.0042** [-0.0070, +0.0200] (bootstrap 95% CI)
- median Δ = -0.0019
- n species = 20
- n with Δ > 0.01: **2**; Δ < −0.01: 2

### Per species

| species | AUC add | AUC deep2 | Δ |
|:--------|--------:|----------:|--:|
| can14 | 0.3892 | 0.5070 | +0.1179 |
| can13 | 0.3833 | 0.4419 | +0.0586 |
| can17 | 0.4704 | 0.4741 | +0.0037 |
| can08 | 0.5417 | 0.5432 | +0.0015 |
| can11 | 0.5955 | 0.5970 | +0.0015 |
| can06 | 0.8842 | 0.8854 | +0.0013 |
| can02 | 0.7030 | 0.7035 | +0.0006 |
| can16 | 0.4971 | 0.4974 | +0.0004 |
| can19 | 0.4641 | 0.4628 | -0.0013 |
| can09 | 0.7125 | 0.7110 | -0.0015 |
| can15 | 0.6810 | 0.6788 | -0.0023 |
| can05 | 0.6314 | 0.6284 | -0.0030 |
| can18 | 0.7809 | 0.7756 | -0.0053 |
| can20 | 0.6240 | 0.6182 | -0.0058 |
| can07 | 0.6682 | 0.6622 | -0.0060 |
| can04 | 0.5667 | 0.5605 | -0.0062 |
| can10 | 0.6698 | 0.6630 | -0.0068 |
| can03 | 0.4815 | 0.4722 | -0.0094 |
| can12 | 0.5301 | 0.5175 | -0.0126 |
| can01 | 0.7348 | 0.6945 | -0.0403 |

## Paired ΔAUC (deep3 − additive)

- mean Δ = **-0.0159** [-0.0396, -0.0011] (bootstrap 95% CI)
- median Δ = -0.0052
- n species = 20
- n with Δ > 0.01: **2**; Δ < −0.01: 7

### Per species

| species | AUC add | AUC deep3 | Δ |
|:--------|--------:|----------:|--:|
| can13 | 0.3833 | 0.4274 | +0.0441 |
| can03 | 0.4815 | 0.4953 | +0.0138 |
| can16 | 0.4971 | 0.4982 | +0.0012 |
| can19 | 0.4641 | 0.4651 | +0.0010 |
| can14 | 0.3892 | 0.3895 | +0.0004 |
| can08 | 0.5417 | 0.5414 | -0.0003 |
| can17 | 0.4704 | 0.4701 | -0.0003 |
| can11 | 0.5955 | 0.5949 | -0.0007 |
| can02 | 0.7030 | 0.7009 | -0.0021 |
| can12 | 0.5301 | 0.5259 | -0.0042 |
| can18 | 0.7809 | 0.7747 | -0.0062 |
| can07 | 0.6682 | 0.6599 | -0.0082 |
| can20 | 0.6240 | 0.6152 | -0.0088 |
| can05 | 0.6314 | 0.6185 | -0.0129 |
| can15 | 0.6810 | 0.6589 | -0.0222 |
| can10 | 0.6698 | 0.6454 | -0.0244 |
| can06 | 0.8842 | 0.8584 | -0.0258 |
| can04 | 0.5667 | 0.5358 | -0.0309 |
| can09 | 0.7125 | 0.6784 | -0.0341 |
| can01 | 0.7348 | 0.5376 | -0.1972 |

## Paired ΔAUC (deep3 − deep2)

- mean Δ = **-0.0201** [-0.0411, -0.0046] (bootstrap 95% CI)

## Interpretation (appendix null)

| contrast | mean ΔAUC | 95% CI | verdict |
|:---------|----------:|:-------|:--------|
| deep2 − add | **+0.0042** | [−0.007, +0.020] | null / slight scatter |
| deep3 − add | **−0.0159** | [−0.040, −0.001] | **worse than additive** |
| deep3 − deep2 | **−0.0201** | [−0.041, −0.005] | depth hurts |

- Fair protocol: residual `η = Σφ + mixer(φ)`, LBFGS, freeze-edges, no-SiLU, warm-start additive, grid frozen.
- Deep-3: pykan stack `[P, h=4, 1]`; init keeps hidden expressive, shrinks final layer only.
- All 20 species converged; runtime ~3.8 min add / 4.3 min deep2 / 12.3 min deep3 (serial, OMP=1).
- **Conclusion:** extra depth does not recover interaction signal; if anything it overfits relative to additive under the same IPP budget. Supports additivity-as-null (Phase 4 main claim); no multi-region Deep-3 expansion indicated.

## Note

- Deep-3 is an **appendix null** under the fair residual protocol (not a novelty rescue).

*Phase 4 Deep KAN interaction ablation.*