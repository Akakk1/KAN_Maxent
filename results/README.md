# Frozen results

Pre-computed tables so reviewers can check numbers and regenerate figures without
re-running multi-day jobs.

| Directory | Contents |
|:----------|:---------|
| `methodological_closure_full_v1/` | Primary metrics for main-text / SI figures (`metrics.csv`, `lambda_star.csv`, …) |
| `nceas/` | Additive benchmark tables |
| `deep2_fair/`, `deep3_can/` | Fair residual Deep-2 / Deep-3 |
| `standard_kan_e2e/` | End-to-end standard KAN (CAN) |
| `curves/` | Response-curve agreement CSVs and φ exports |
| `tgb_closure_v1/` | Target-group background summaries |

Figures are **not** duplicated here; see `figures/ms_results/v7/` and `SI/figures/`.

Regenerate figures:

```bash
python benchmarks/plot_ms_closure_v7.2.py
```
