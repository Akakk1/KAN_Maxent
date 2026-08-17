# Fair Deep residual protocol (canonical)

**Use only this protocol for manuscript Deep NOT results.**  
Primary run: `deepkan_fair_six_3seed_20260715`.

**Not the same experiment:** end-to-end standard KAN on continuous \(x\) is defined in `docs/standard_kan_e2e_can20_protocol.md`. Residual hybrid flag flips are not that protocol.

## Formula

\[
\eta = \sum_p \phi_p(x_p) + \Phi(\phi_1,\ldots,\phi_P) + X_{\mathrm{cat}}\beta
\]

- Layer-1 edges: self-contained `AdditiveSplineKAN` (G=6, K=3)
- \(\Phi\): pykan `KANLayer(P → 1)`, **grid frozen**
- Deep-3 appendix: \(\Psi(\Phi(\phi))\) with width \(h=4\) → `[P, 4, 1]`

## Fair defaults

| Item | Value |
|:-----|:------|
| residual | True |
| optimizer | L-BFGS |
| SiLU base | **off** (Identity, `scale_base=0`) |
| warm-start | from fitted additive edges |
| freeze-edges | True (train mixer only after warm-start) |
| \(\lambda_s\) | \(10^{-2}\) |
| \(\lambda_r\) | \(10^{-6}\) |
| \(\lambda_\phi\) | \(10^{-4}\) (mixer ridge) |
| seeds (Deep-2 main) | 0, 1, 2 |
| evaluation | same as Phase 3: PO + random_50k → independent PA |

## CLI

```bash
python benchmarks/deepkan_ablation.py \
  --regions CAN \
  --seeds 0,1,2 \
  --models additive,deep2 \
  --deep-optimizer lbfgs \
  --lbfgs-steps 12
# freeze-edges on by default; disable with --no-freeze-edges
```

Deep-3 smoke (appendix):

```bash
python benchmarks/deepkan_ablation.py \
  --regions CAN --seeds 0 \
  --models additive,deep2,deep3 \
  --hidden-width 4 \
  --outdir outputs/deepkan_deep3_can20_fair_<id>
```

## Deprecated (do not use as main claim)

| Run pattern | Why |
|:------------|:----|
| Early Adam / non-residual / no warm-start ablations | Inflates negative Δ; unfair to additive |
| `deepkan_ablation_can20_20260715` without “fair” in name | Check manifest before citing |

## Reporting

- Species-level Δ = deep − additive; multi-seed: mean Δ per species then bootstrap over species.
- Main claim: **no systematic PA-AUC gain** under fair residual Deep-2 (six regions × 3 seeds).
