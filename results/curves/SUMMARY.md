# Phase 5 Curves SUMMARY

**Run:** `phase5_curves_20260715`  
**Identity:** additive B-spline φ_p under IPP (not deep KAN)

## NCEAS sample (primary)

- 18 species (3 per region: high AUC ×2 + mid ×1)
- KAN vs GAM edge correlation: **mean r = 0.988**, **median r = 0.9997** (n=159 feature×species)
- FigN1: KAN solid + GAM dashed

## Optional Ginkgo

- Full-variable φ for KAN-IPP and GAM-IPP
- FigG1 available; not required for main claim

## Bootstrap (B=50)

- Ginkgo: all env vars → `bootstrap/ginkgo/phi_*_boot.csv`
- NCEAS: 6 species → `bootstrap/nceas/{reg}/{sp}/`

## Device

- Pilot decision: **cpu_only** (no GPU LBFGS)

See `REPORT_Phase5.md` and `ENHANCE_SUMMARY.md`.
