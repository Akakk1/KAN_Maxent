# Standard KAN end-to-end CAN pilot — SUMMARY

**Protocol:** `docs/standard_kan_e2e_can20_protocol.md`
**Run dir:** `standard_kan_e2e_can20_seed0_20260727`
**Architecture:** η = KAN_[P, h=4, 1](T(x)) + X_cat @ β
**KAN input:** scaled continuous covariates (not additive φ edges)
**Seeds:** [0]
**Species requested:** 20

## Formula

```
eta = KAN(T(x)) + c^T beta
T = per-feature quantile affine map into [-1, 1] fitted on PO+BG only
```

## Paired ΔAUC (standard_kan − additive)

- Complete-case species-seed rows: **20** (failures excluded from mean: **0**)
- Mean ΔAUC: **-0.0100** (bootstrap 95% CI [-0.0444, +0.0226])
- Median ΔAUC: **-0.0152**
- n(Δ > 0.01): 5; n(Δ < −0.01): 11

This experiment does **not** replace the fair residual Deep protocol.
