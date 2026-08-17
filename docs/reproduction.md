# Reproduction guide

## Claims (for checking numbers)

1. Additive B-spline IPP matches same-basis GAM and is competitive with maxnet on independent PA evaluation (NCEAS, 225 species).
2. End-to-end multilayer KAN underperforms maxnet.
3. Fair residual Deep-2 / Deep-3 do not systematically improve independent-PA AUC over the additive baseline.

## Environment

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate kanmaxent
pip install -e .
# R maxnet optional: docs/r_dependencies.md
pytest tests/ -q
```

CPU is sufficient for additive fits and fair Deep L-BFGS.

## Level 0 — unit tests

```bash
pytest tests/ -q
```

## Level 1 — smoke (minutes)

```bash
bash scripts/reproduce_smoke.sh
```

Or:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python benchmarks/deepkan_ablation.py \
  --regions CAN --species can02,can13,can14 --seeds 0 \
  --models additive,deep2,deep3 --hidden-width 4 \
  --outdir outputs/smoke_deep3_can3
```

## Level 2 — full tables (hours–days)

| Product | Entry point | Frozen copy |
|:--------|:------------|:------------|
| Methodological closure | `benchmarks/methodological_closure.py` | `results/methodological_closure_full_v1/` |
| NCEAS additive | `benchmarks/nceas_runner.py` | `results/nceas/` |
| Deep-2 fair (6×3 seeds) | `benchmarks/deepkan_ablation.py` + `scripts/merge_deepkan_fair.py` | `results/deep2_fair/` |
| Response curves | `benchmarks/phase5_curves.py` | `results/curves/` |
| Target-group background | `benchmarks/tgb_closure_v1.py` | `results/tgb_closure_v1/` |

Long runs write under `outputs/` (gitignored). Prefer frozen tables under `results/` for manuscript numbers.

## Regenerate figures

```bash
python benchmarks/plot_ms_closure_v7.2.py
```

Reads `results/methodological_closure_full_v1/metrics.csv` (or `outputs/…` if present)
and writes `figures/ms_results/v7/` + `SI/figures/` + `SI/tables/*.csv`.
Does not overwrite `SI/SI.md` prose.

## Data

| Path | Content |
|:-----|:--------|
| `data/nceas/{awt,can,nsw,nz,sa,swi}/` | PO / PA / BG per region |
| `data/ginkgo/` | Optional Ginkgo table |

Cite Elith et al. (2020) and Valavi et al. (2022) for NCEAS.

## Protocol notes

- Fair residual Deep: `docs/fair_deep_protocol.md`
- End-to-end standard KAN: `docs/standard_kan_e2e_can20_protocol.md`
