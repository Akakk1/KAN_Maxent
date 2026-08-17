# Data and code availability (draft for manuscript)

Copy/adapt into the English Word manuscript:

---

Code for additive B-spline IPP models, fair residual deep-interaction probes, evaluation metrics (including COR), and figure generation is available in the accompanying GitHub repository (MIT license). The Python package `kanmaxent` installs with `pip install -e .`; unit tests and a short CAN three-species smoke script verify the environment. Pre-computed metrics tables and manuscript figures are under `results/` (see `results/README.md` and `results/SHA256SUMS.txt`). Presence-only and presence–absence inputs follow the NCEAS benchmark of Elith et al. (2020) and Valavi et al. (2021/2022); processed region tables are included under `data/nceas/`. Optional R `maxnet` baselines require `maxnet` 0.1.4 (`docs/r_dependencies.md`). Full multi-region re-runs write to a local `outputs/` directory and are not required to recover the published summary statistics.

---
