# Protocol: end-to-end standard KAN–IPP on CAN (20 species)

**Status:** design locked; implementation in progress under KAN-MaxEnt.  
**Code:** `kanmaxent/models/standard_kan_ipp.py`, `benchmarks/standard_kan_e2e.py`  
**Do not** treat residual-hybrid flag flips as this experiment.  
**Workspace:** KAN-MaxEnt only. Sync to repro only after explicit confirmation.  
**Related (different experiment):** `docs/fair_deep_protocol.md` — residual interaction after a fitted additive model.

---

## 1. Scientific question

Under the same presence-only training and independent presence–absence evaluation as Phase 3, does a **standard multi-layer Kolmogorov–Arnold Network trained end-to-end on the continuous environmental covariates** improve independent-PA AUC relative to the **additive B-spline IPP** baseline on the CAN 20-species set?

This is **not** the same question as: after fitting additive edges, does a residual KAN mixer help? The residual experiment is already the manuscript’s main Deep result. This protocol is a separate, reviewer-facing check that the paper did not avoid a conventional deep KAN by only testing a residual hybrid.

---

## 2. What “end-to-end standard KAN” means here

### 2.1 Definition (normative)

Let \(x \in \mathbb{R}^{P}\) be the vector of **continuous** covariates for a site, after a fixed preprocessing map \(T\) defined below. Let \(z = T(x) \in \mathbb{R}^{P}\). Let \(c\) be the one-hot encoding of categorical covariates (same scheme as Phase 3).

The **standard KAN–IPP model** is:

\[
\eta(x, c) = \mathrm{KAN}_{[P,\,h,\,1]}(z) + c^{\top}\beta
\]

where:

1. \(\mathrm{KAN}_{[P,\,h,\,1]}\) is a two-hidden-edge-layer Kolmogorov–Arnold network in the sense of Liu et al.: fully connected layers whose **edges** are univariate learnable functions (B-splines plus optional base activation), implemented with **pykan** `KANLayer` stacked as  
   \(z \xrightarrow{\text{KANLayer } P\to h} u \xrightarrow{\text{KANLayer } h\to 1} \eta_{\mathrm{cont}}\).  
2. **Input to the KAN is \(z = T(x)\)**, not the vector of additive edge outputs \(\phi(x)\) from `AdditiveSplineKAN`.  
3. **There is no additive skip** \(\sum_p \phi_p(x_p)\) and no residual path that injects a pre-fitted additive \(\eta\).  
4. **All KAN parameters and \(\beta\) are trained jointly** from random initialization under the IPP objective (end-to-end). No warm-start from the additive fit; no freezing of a first “edge bank” taken from the additive model.  
5. Categorical effects remain a **linear head on one-hot codes**, identical in role to Phase 3 / additive hybrid fits. They are not folded into the KAN graph in the first implementation (documented limitation, not a silent shortcut).

### 2.2 What is explicitly out of scope for this protocol

The following are **different models**. They may be useful elsewhere, but they **do not satisfy** this protocol and must not be labeled “standard KAN e2e” in outputs or text:

| Construction | Why it is not this experiment |
|--------------|--------------------------------|
| Fair residual Deep-2/3: \(\eta = \sum\phi + \mathrm{mixer}(\phi)\) with warm-start and frozen edges | Residual interaction test after additive fit (`fair_deep_protocol.md`) |
| Same hybrid with residual off, SiLU on, joint training of \(\phi\) and mixer | Still a **separable first layer of our B-spline edges** feeding a mixer; input is \(\phi(x)\), not standard KAN on \(z\) |
| Only flipping CLI flags on `DeepKanHybrid` | Implementation convenience, wrong scientific object |

### 2.3 Honest naming in code and manifests

Use a distinct model id, for example:

- `standard_kan_ipp` or `e2e_kan_ipp`

Manifest must include at least:

- `protocol: standard_kan_e2e_can20`
- `kan_input: continuous_z` (not `phi_from_additive_edges`)
- `architecture: pykan_[P,h,1]`
- `residual_additive_skip: false`
- `warm_start_additive: false`
- `freeze_edges_from_additive: false` (N/A if no additive edge module)
- spline grid settings, SiLU on/off, optimizer schedule, seeds, software versions

---

## 3. Preprocessing map \(T\) (continuous inputs)

KAN layers expect coordinates on a controlled range. Define \(T\) **once per species fold** using **training presence + background only** (no PA leakage):

1. For each continuous feature \(p\), compute training quantiles or min/max on the PO+BG design matrix (prefer **1%–99% quantiles** clipped, then affine map into the KAN grid range).  
2. Default **grid range for pykan layers:** \([-1, 1]\) after affine scaling of each coordinate into that interval (document the exact affine coefficients in status JSON).  
3. Apply the same \(T\) to independent PA sites at evaluation time.

Rationale: raw environmental units differ by orders of magnitude; feeding unscaled \(x\) into a fixed spline grid is not a meaningful “standard KAN” test.

Categorical one-hot encoding: same as `prepare_matrices` / Phase 3 (including any reference-level handling already used for additive and maxnet hybrids).

---

## 4. Architecture defaults (CAN pilot)

CAN has \(P = 6\) continuous covariates (`alt`, `asp2`, `ontprec`, `ontslp`, `onttemp`, `watdist`) and one categorical (`ontveg`).

| Item | Default | Notes |
|------|---------|--------|
| Width | \([P,\,h,\,1]\) with **\(h = 4\)** | Matches depth used in fair Deep-3 appendix for fair comparison of “depth budget,” not because \(h=4\) is optimal |
| Optional sensitivity (only if main run is stable and interesting) | \(h = P = 6\) | Single extra arm, seed 0 only |
| B-spline intervals / degree on KAN edges | \(G = 6\), \(K = 3\) | Align with additive defaults where possible |
| Base activation (SiLU) | **On** (pykan-like default) | Record `base_fun=SiLU` |
| Grid update during training | **Off** (frozen grids) | Same freeze policy as fair deep for optimizability and comparability |
| pykan entropy / sparsity penalties | **Off** | IPP + our explicit penalties only |

Do not sweep large grids of \(h\), \(G\), or learning rates in the first complete pass.

---

## 5. Objective and regularisation

Same estimand and data construction as Phase 3:

- Training: presence-only + random 50k background (CAN background already fixed in data package).  
- Evaluation: independent PA; PA never used for fitting or for choosing \(T\).  
- Loss: inhomogeneous Poisson process (IPP) logistic form already used for additive KAN–IPP.  
- Skip species with \(n_{\mathrm{PO}} < 5\).

Regularisation (first pass, aligned in spirit with additive / fair deep):

| Penalty | Default | Applies to |
|---------|---------|------------|
| Smoothness on KAN spline coefficients | \(\lambda_s = 10^{-2}\) if an analogous smoothness form is available on pykan coefficients; otherwise document substitution | Prefer true smoothness; if only ridge on all KAN parameters is practical in v1, set \(\lambda_{\mathrm{kan}} = 10^{-4}\) and state that smoothness is approximate |
| Ridge on KAN parameters | \(\lambda_{\mathrm{kan}} = 10^{-4}\) | All trainable KAN weights |
| Ridge on \(\beta\) | \(\lambda_r = 10^{-6}\) | Categorical linear head |

Exact penalty implementation must be written in code comments and `export_manifest()` so it is auditable. Prefer matching additive’s \(\lambda_s,\lambda_r\) philosophy rather than inventing a third unrelated scale without recording it.

---

## 6. Optimisation (end-to-end)

Random init of KAN + \(\beta\). No additive warm-start.

**Primary schedule:**

1. Adam, learning rate \(0.03\), \(150\) steps (full-batch PO+BG if memory allows; otherwise document mini-batch size).  
2. L-BFGS refinement, \(10\) steps (or until grad norm / loss stall), same IPP + penalty.

Record per species: final loss, whether non-finite, wall time, parameter count.

**Failure rule:** if loss or PA-AUC is non-finite, mark `status=failed`. Primary tables use complete cases; report failure count in the summary.

**Seeds:**

| Stage | Seeds | Scope |
|-------|-------|--------|
| Smoke | 0 | 3 species (e.g. can01, can06, can14) |
| Main | 0 | All 20 CAN species |
| Follow-up | 1, 2 | Only if main mean \(\Delta\)AUC is scientifically interesting or unstable |

---

## 7. Comparators (what to put on the same table)

For each species (and seed when multi-seed):

| Column | Source |
|--------|--------|
| `auc_additive` | Existing additive KAN–IPP (Phase 3 / published metrics); re-fit only if seed or pipeline mismatch requires it |
| `auc_standard_kan` | This protocol |
| `delta_auc` | `auc_standard_kan - auc_additive` |
| `auc_fair_deep2` / `auc_fair_deep3` | Optional join from published fair residual runs (same species, preferably same seed) for discussion only |

Fair residual numbers are **side context**. They are not a substitute for the additive baseline and not evidence that “standard KAN was tested” if this protocol was not run.

---

## 8. Outputs and reporting

Suggested directory:

```text
outputs/standard_kan_e2e_can20_seed0_<YYYYMMDD>/
  manifest.json
  metrics.csv
  paired_delta_auc.csv
  status/<region>_<species>_s<seed>.json
  SUMMARY.md
```

`SUMMARY.md` must state in plain language:

- the formula \(\eta = \mathrm{KAN}(T(x)) + c^{\top}\beta\);  
- that input is scaled continuous \(x\), not additive \(\phi\);  
- mean/median \(\Delta\)AUC vs additive, bootstrap CI over species, failure count.

Manuscript use (after results exist):

- Methods / SI: short definition + CAN pilot table.  
- Do not change the main Deep residual claim unless this pilot shows a clear, stable advantage that survives multi-seed checks.

**Promotion rules:**

| Outcome | Action |
|---------|--------|
| Mean \(\Delta\)AUC near 0 or negative, low failure rate | SI or Methods note: standard KAN e2e does not beat additive on CAN under this protocol |
| High failure / wild variance | Report instability; do not spin as a success |
| Clear positive mean \(\Delta\) with CI excluding 0 | Multi-seed; then discuss; still do not silently replace fair residual narrative |
| Expand beyond CAN | Only if CAN is stable and positive enough to justify cost |

---

## 9. Implementation plan (engineering, after this doc is approved)

Work only under KAN-MaxEnt until confirmed.

1. **New module** (preferred name): `kanmaxent/models/standard_kan_ipp.py`  
   - Build stacked pykan `KANLayer`s \([P,h,1]\) on \(z = T(x)\).  
   - Linear categorical head.  
   - `forward` / IPP fit API parallel to `fit_deep_kan_ipp` but **no** `AdditiveSplineKAN` trunk.  
2. **Runner:** extend `benchmarks/deepkan_ablation.py` with a distinct model name **or** add `benchmarks/standard_kan_e2e.py` dedicated to this protocol (dedicated script is clearer and avoids flag confusion).  
3. **Smoke:** 3 species, seed 0.  
4. **Main:** 20 species, seed 0.  
5. **Write-up:** table + optional figure only after numbers exist.

Do **not** implement this experiment by only adding `--no-residual` paths on `DeepKanHybrid`.

---

## 10. Relation to the manuscript story

| Experiment | Question | Role in paper |
|------------|----------|----------------|
| Additive vs GAM vs maxnet | Additive B-spline IPP as a competitive, interpretable SDM | Main results |
| Fair residual Deep-2/3 | Marginal value of interaction **after** additive fit | Main Deep claim (null / no systematic gain) |
| **This protocol** | Does a **standard multi-layer KAN on \(T(x)\)** beat additive IPP on CAN? | Completeness / reviewer defense; SI or short Methods note unless result forces more |

Keeping these three separate is the point of this document.

---

## 11. Approval checkpoint

Implementation and long runs start only after explicit approval of this definition.  
If the definition changes (for example: include categoricals inside the KAN, or use depth-2 \([P,1]\) only), update this file first, then code.
