# KAN-MaxEnt

Presence-only species distribution models with learnable B-spline edge functions
(discrete IPP), maxnet / same-basis GAM baselines, and fair residual Deep-KAN probes.

**Finding in brief:** additive B-splines match same-basis GAMs and slightly exceed
maxnet; multilayer / residual deep interactions do not improve independent PA AUC.

**Licence (split):** software MIT; NCEAS tables under `data/nceas/` are
**CC BY-NC 4.0** (not MIT). See [Licence](#licence) below.

This repository is a **reproduction package**: code, environment, input data layout,
frozen metrics, and figure assets.

## Install

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate kanmaxent
pip install -e .
pytest tests/ -q
```

## Layout

```text
kanmaxent/                 # Python package
benchmarks/                # Experiment + figure CLIs
tests/                     # Unit tests
data/nceas/                 # Six-region NCEAS PO/PA/BG
results/                   # Frozen run tables (metrics, λ★, curves, …)
figures/ms_results/v7/     # Main-text figures (v7.2 render)
SI/                        # SI figures, CSV tables, figure captions
docs/                      # Install, data, and protocol notes
scripts/reproduce_smoke.sh # Short end-to-end smoke
```

## Reproduce figures

Frozen metrics live under `results/` (no multi-day re-run required):

```bash
python benchmarks/plot_ms_closure_v7.2.py
```

| Output | Path |
|:-------|:-----|
| Main-text Fig. 1–5 | `figures/ms_results/v7/` |
| SI Fig. S1–S5 | `SI/figures/` |
| SI Table S1–S14 CSVs | `SI/tables/` |

Primary metrics file: `results/methodological_closure_full_v1/metrics.csv`.

## Full experiment CLIs

| Script | Role |
|:-------|:-----|
| `benchmarks/methodological_closure.py` | Stage A–C closure (additive / residual / e2e) |
| `benchmarks/nceas_runner.py` | Additive NCEAS benchmark |
| `benchmarks/deepkan_ablation.py` | Fair residual Deep-2/3 |
| `benchmarks/standard_kan_e2e.py` | End-to-end standard KAN |
| `benchmarks/tgb_closure_v1.py` | Target-group background arm |
| `benchmarks/phase5_curves.py` | Response-curve agreement |
| `benchmarks/compare_valavi_maxnet.py` | Alignment vs Valavi OSF maxnet |

Smoke (minutes):

```bash
bash scripts/reproduce_smoke.sh
```

Long runs write under `outputs/` (gitignored). See `docs/reproduction.md`.

## Frozen results

| Path | Contents |
|:-----|:---------|
| `results/methodological_closure_full_v1/` | Metrics for v7.2 figures / SI |
| `results/nceas/` | Earlier additive tables |
| `results/deep2_fair/`, `results/deep3_can/` | Fair residual deep |
| `results/standard_kan_e2e/` | e2e CAN pilot |
| `results/curves/` | Curve agreement + φ CSVs |
| `results/tgb_closure_v1/` | Target-group background |

## Licence

| Material | Licence |
|:---------|:--------|
| Python package, scripts, tests, docs authored here | [MIT](LICENSE) |
| `data/nceas/` (Elith et al. 2020 tables) | [CC BY-NC 4.0](data/nceas/LICENSE) — attribution required; **no commercial use** |
| `data/ginkgo/` (optional) | Mixed GBIF + WorldClim 2.1 (CC BY-SA); [not MIT](data/ginkgo/LICENSE) |
| Author-generated `results/` and `SI/tables/` (metrics, not occurrence coordinates) | MIT, without changing CC BY-NC terms on the source tables |

The root MIT file does **not** relicense `data/`. Redistributing this repository
commercially, or stripping the NC restriction from the NCEAS CSVs, would
violate Elith et al. (2020). Details: `NOTICE`, `data/LICENSE`.

## Citation

- NCEAS data: Elith et al. (2020), doi:10.17161/bi.v15i2.13384 (CC BY-NC 4.0); Valavi et al. (2022)
- KAN: Liu et al. (2025)
- Code: this repository (tag `ms-submission`)

## AI use statement

Artificial intelligence tools were used to assist software development and figure
preparation during this study. The computational workflow was developed with the
assistance of OpenClaw and Grok Build, together with the DeepSeek-V4 API, to support
code generation, debugging, and iterative workflow development under continuous
author supervision. All AI-assisted code was manually reviewed, tested, and validated
by the authors before implementation. All study design, analyses, interpretation of
results, figures, and manuscript content were conceived, verified, and approved by
the authors.

This statement matches the manuscript AI use disclosure. AI tools are **not** listed
as repository co-authors or GitHub collaborators.
