#!/usr/bin/env python
"""Methodological Closure v1 — full six-region re-analysis.

Protocol: ``docs/plans/Plan_Methodological_Closure_v1.md``

Stages
------
A  Species-level λ* (5-fold PO CV) + additive + same-basis GAM + maxnet@10k/50k
B  Fair Deep-2 residual, mixer_input ∈ {phi, raw_scaled}, seeds 0,1,2 @ λ*
C  Fair Deep-3 residual, both mixer inputs, six regions, seeds 0,1,2 @ λ*
D  Standard KAN e2e six regions + diagnostics + pre-registered remediation
E  Summaries / master tables

Resume-safe: metrics rows keyed by (stage, region, species, seed, model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.data.categorical import prepare_matrices
from kanmaxent.data.nceas_io import (
    build_po_bg_frame,
    load_region,
    pa_labels_and_env_for_species,
    split_species,
)
from kanmaxent.evaluation.lambda_selector import (
    CLOSURE_DEFAULT_LAMBDA,
    CLOSURE_K_FOLDS,
    CLOSURE_LAMBDA_GRID,
    CLOSURE_MIN_PO_FOR_CV,
    select_lambda_s_po_cv,
)
from kanmaxent.evaluation.metrics import ranking_metrics
from kanmaxent.models.deep_kan import fit_deep_kan_ipp
from kanmaxent.models.standard_kan_ipp import fit_standard_kan_ipp
from kanmaxent.nceas_fit import HybridSplineCat, fit_hybrid_gam_ipp, fit_hybrid_ipp
from kanmaxent.reference.maxnet_nceas import fit_predict_maxnet_nceas

PROTOCOL_ID = "methodological_closure_v1"
ALL_REGIONS = ("AWT", "CAN", "NSW", "NZ", "SA", "SWI")

# model ids
M_ADD = "additive_kan_ipp"
M_GAM = "gam_ipp_same_basis"
M_MX10 = "maxnet_bg10k"
M_MX50 = "maxnet_bg50k"
M_D2_PHI = "deep2_rphi"
M_D2_RX = "deep2_rx"
M_D3_PHI = "deep3_rphi"
M_D3_RX = "deep3_rx"
M_SKAN = "standard_kan_ipp"

DEEP2_MODELS = (M_D2_PHI, M_D2_RX)
DEEP3_MODELS = (M_D3_PHI, M_D3_RX)


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _limit_blas_threads(n: int = 1) -> None:
    """Avoid oversubscription when many process workers are active.

    Only set env vars + ``torch.set_num_threads``. Do **not** call
    ``set_num_interop_threads`` after any torch work (raises c10::Error and
    aborts the worker process).
    """
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        import torch

        # Safe after import; interop threads must not be touched mid-run.
        torch.set_num_threads(n)
    except Exception:
        pass


def _init_parallel_worker(blas_threads: int = 1) -> None:
    # Env first, before any torch import in this process if possible
    n = max(1, int(blas_threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[key] = str(n)
    _limit_blas_threads(n)


def _warm_from_ser(ws_ser: Optional[Dict]) -> Optional[Dict]:
    if not ws_ser:
        return None
    return {
        "spline_coeffs": np.asarray(ws_ser["spline_coeffs"], dtype=np.float64),
        "beta_cat": None
        if ws_ser.get("beta_cat") is None
        else np.asarray(ws_ser["beta_cat"], dtype=np.float64),
    }


def _worker_deep_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool worker: one fair residual fit. Protocol unchanged."""
    _limit_blas_threads(int(job.get("blas_threads", 1)))
    t0 = time.perf_counter()
    data = load_region(job["region"])
    row = run_deep_species(
        data,
        job["species_id"],
        stage=job["stage"],
        depth=int(job["depth"]),
        mixer_input=job["mixer_input"],
        model_id=job["model_id"],
        lambda_s=float(job["lambda_s"]),
        lambda_path=str(job["lambda_path"]),
        warm_state=_warm_from_ser(job.get("warm_state")),
        seed=int(job["seed"]),
        lbfgs_steps=int(job.get("lbfgs_steps", 12)),
        hidden_width=int(job.get("hidden_width", 4)),
    )
    return {
        "ok": True,
        "rows": [row],
        "label": (
            f"{job['stage']} {job['region']}/{job['species_id']} "
            f"s{job['seed']} {job['model_id']}"
        ),
        "wall_s": time.perf_counter() - t0,
    }


def _worker_e2e_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process-pool worker: e2e + pre-registered remediation chain."""
    _limit_blas_threads(int(job.get("blas_threads", 1)))
    t0 = time.perf_counter()
    data = load_region(job["region"])
    mx = job.get("auc_maxnet_10k")
    try:
        mx_f = float(mx) if mx is not None and mx == mx else None  # NaN-safe
    except (TypeError, ValueError):
        mx_f = None
    rows = remediate_e2e(
        data,
        job["species_id"],
        lambda_s=float(job["lambda_s"]),
        lambda_path=str(job["lambda_path"]),
        maxnet_auc=mx_f,
        primary_seed=int(job.get("primary_seed", 0)),
    )
    last = rows[-1] if rows else {}
    return {
        "ok": True,
        "rows": rows,
        "label": f"D {job['region']}/{job['species_id']}",
        "wall_s": time.perf_counter() - t0,
        "final_rem": last.get("remediation"),
        "auc": last.get("auc_roc"),
        "failed": last.get("failed"),
    }


def _run_jobs_parallel(
    jobs: List[Dict[str, Any]],
    *,
    workers: int,
    worker_fn,
    blas_threads: int,
    on_result,
    label: str,
) -> None:
    """Execute independent jobs with a process pool; protocol identical to serial."""
    if not jobs:
        print(f"  ({label}: nothing pending)", flush=True)
        return
    n_workers = max(1, min(int(workers), len(jobs)))
    print(
        f"  {label}: {len(jobs)} jobs × {n_workers} workers "
        f"(BLAS threads/worker={blas_threads})",
        flush=True,
    )
    if n_workers == 1:
        for j in jobs:
            j = {**j, "blas_threads": blas_threads}
            res = worker_fn(j)
            on_result(res)
        return
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_parallel_worker,
        initargs=(blas_threads,),
    ) as ex:
        futs = {
            ex.submit(worker_fn, {**j, "blas_threads": blas_threads}): j for j in jobs
        }
        done_n = 0
        for fut in as_completed(futs):
            done_n += 1
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                j = futs[fut]
                print(
                    f"    FAIL [{done_n}/{len(jobs)}] {j.get('region')}/{j.get('species_id')}: {e}",
                    flush=True,
                )
                continue
            on_result(res)
            if done_n % 10 == 0 or done_n == len(jobs):
                print(f"    progress {done_n}/{len(jobs)}", flush=True)


def _row_key(r: Dict) -> Tuple:
    return (
        str(r.get("stage", "")),
        str(r.get("region", "")),
        str(r.get("species_id", "")),
        int(r.get("seed", 0)),
        str(r.get("model", "")),
        str(r.get("remediation", "primary")),
    )


def _metrics_block(y_pa: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    m = ranking_metrics(y_pa, scores)
    return {
        "auc_roc": m.get("auc_roc", float("nan")),
        "auprc": m.get("auprc", float("nan")),
        "prg": m.get("prg", float("nan")),
        "cor": m.get("cor", float("nan")),
    }


def _base_row(
    region: str,
    species_id: str,
    taxon_group: str,
    n_po: int,
    n_pa: int,
    seed: int,
    stage: str,
    lambda_s: float,
    lambda_path: str,
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL_ID,
        "stage": stage,
        "region": region,
        "species_id": species_id,
        "taxon_group": taxon_group,
        "seed": int(seed),
        "n_presence": int(n_po),
        "n_PA_test": int(n_pa),
        "lambda_s": float(lambda_s),
        "lambda_selection": lambda_path,
        "background_scheme": "random_50k",
        "integration_support": "full_background_50k",
        "objective": "ipp",
        "remediation": "primary",
    }


def select_lambda_for_species(
    train_df: pd.DataFrame,
    n_po: int,
    continuous: Sequence[str],
    categorical: Sequence[str],
    *,
    seed: int = 0,
    lambda_r: float = 1e-6,
    additive_steps_cv: int = 8,
) -> Dict[str, Any]:
    n_bg = int((train_df["occ"] == 0).sum())

    def fit_eval(po_tr_idx, po_val_idx, lam: float) -> float:
        po_block = train_df.iloc[:n_po]
        bg_block = train_df.iloc[n_po:]
        bg_s = bg_block.sample(n=min(8000, len(bg_block)), random_state=seed)
        tr = pd.concat([po_block.iloc[po_tr_idx], bg_s], ignore_index=True)
        val = pd.concat(
            [
                po_block.iloc[po_val_idx],
                bg_s.sample(n=min(2000, len(bg_s)), random_state=seed + 1),
            ],
            ignore_index=True,
        )
        prep = prepare_matrices(tr, val, list(continuous), list(categorical))
        # evaluate on val frame env with fake "test" = val rows
        _, scores, _ = fit_hybrid_ipp(
            prep["X_cont_train"],
            prep["X_cat_train"],
            tr["occ"].to_numpy(dtype=np.float64),
            prep["X_cont_test"],
            prep["X_cat_test"],
            lambda_s=float(lam),
            lambda_r=lambda_r,
            seed=seed,
            steps=additive_steps_cv,
        )
        return ranking_metrics(val["occ"].to_numpy(dtype=np.float64), scores)["auc_roc"]

    return select_lambda_s_po_cv(
        n_po,
        n_bg,
        fit_eval,
        grid=CLOSURE_LAMBDA_GRID,
        k_folds=CLOSURE_K_FOLDS,
        min_po_for_cv=CLOSURE_MIN_PO_FOR_CV,
        default_lambda=CLOSURE_DEFAULT_LAMBDA,
        seed=seed,
    )


def fit_maxnet_arm(
    train_df: pd.DataFrame,
    pa_env: pd.DataFrame,
    covars: List[str],
    categorical: List[str],
    n_po: int,
    max_bg: int,
    seed: int,
) -> Tuple[np.ndarray, Dict]:
    tr_mx = train_df[covars + ["occ"]].copy()
    is_bg = tr_mx["occ"].to_numpy() == 0
    if int(is_bg.sum()) > max_bg:
        po_mx = tr_mx.loc[~is_bg]
        bg_mx = tr_mx.loc[is_bg].sample(n=max_bg, random_state=seed)
        tr_mx = pd.concat([po_mx, bg_mx], ignore_index=True)
    te_mx = pa_env[covars].copy()
    scores, info = fit_predict_maxnet_nceas(
        tr_mx,
        te_mx,
        covars,
        categorical,
        n_presence=n_po,
        regmult=1.0,
    )
    return scores, info


def is_e2e_failure(
    *,
    converged: bool,
    final_loss: float,
    auc: float,
    maxnet_auc: Optional[float],
) -> bool:
    if not converged:
        return True
    if not np.isfinite(final_loss):
        return True
    if not np.isfinite(auc):
        return True
    if maxnet_auc is not None and np.isfinite(maxnet_auc):
        # Collapse relative to classical baseline (can01-class): large gap
        # while absolute discrimination remains weak/mediocre.
        if (maxnet_auc - auc) > 0.15 and auc < 0.55:
            return True
    return False


def run_stage_a_species(
    region_data,
    species_id: str,
    *,
    seed: int = 0,
    lambda_r: float = 1e-6,
    additive_steps: int = 15,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Return metric rows + cache dict (lambda, warm_state, maxnet aucs)."""
    reg = region_data.region
    po_sp = split_species(region_data.po, species_id)
    n_po = len(po_sp)
    y_pa, pa_env, taxon_group = pa_labels_and_env_for_species(region_data, species_id)
    n_pa = int(len(y_pa))
    covars = region_data.continuous + region_data.categorical
    continuous = list(region_data.continuous)
    categorical = list(region_data.categorical)

    if n_po < 5:
        base = _base_row(
            reg, species_id, taxon_group, n_po, n_pa, seed, "A",
            CLOSURE_DEFAULT_LAMBDA, "skipped_n_po_lt_5",
        )
        rows = []
        for m in (M_ADD, M_GAM, M_MX10, M_MX50):
            rows.append({
                **base, "model": m, "n_background": 0, "converged": False,
                "runtime_s": 0.0, "auc_roc": float("nan"), "auprc": float("nan"),
                "prg": float("nan"), "cor": float("nan"), "skip_reason": "n_po<5",
            })
        return rows, {"lambda_s": CLOSURE_DEFAULT_LAMBDA, "lambda_path": "skipped_n_po_lt_5"}

    train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    n_bg = int((y_tr == 0).sum())

    sel = select_lambda_for_species(
        train_df, n_po, continuous, categorical, seed=seed, lambda_r=lambda_r
    )
    lam = float(sel["lambda_s"])
    lam_path = str(sel["path"])

    mats = prepare_matrices(train_df, pa_env, continuous, categorical)
    Xc_tr, Xk_tr = mats["X_cont_train"], mats["X_cat_train"]
    Xc_te, Xk_te = mats["X_cont_test"], mats["X_cat_test"]

    base = _base_row(reg, species_id, taxon_group, n_po, n_pa, seed, "A", lam, lam_path)
    base["n_background"] = n_bg
    base["lambda_cv_scores"] = json.dumps(
        {str(k): v for k, v in (sel.get("scores") or {}).items()}
    )
    base["lambda_best_cv_score"] = sel.get("best_score", float("nan"))

    rows: List[Dict] = []
    cache: Dict[str, Any] = {
        "lambda_s": lam,
        "lambda_path": lam_path,
        "lambda_sel": sel,
        "warm_state": None,
        "auc_maxnet_10k": float("nan"),
    }

    # Additive
    t0 = time.perf_counter()
    try:
        add_model, scores_add, meta_add = fit_hybrid_ipp(
            Xc_tr, Xk_tr, y_tr, Xc_te, Xk_te,
            lambda_s=lam, lambda_r=lambda_r, seed=seed, steps=additive_steps,
        )
        mb = _metrics_block(y_pa, scores_add)
        cache["warm_state"] = {
            "spline_coeffs": add_model.spline.coeffs.detach().cpu().numpy().copy(),
            "beta_cat": None
            if add_model.beta_cat is None
            else add_model.beta_cat.detach().cpu().numpy().copy(),
        }
        cache["auc_additive"] = mb["auc_roc"]
        rows.append({
            **base, "model": M_ADD,
            "converged": bool(meta_add.get("converged", True)),
            "runtime_s": float(meta_add.get("runtime_s", time.perf_counter() - t0)),
            "final_loss": float("nan"),
            "skip_reason": "",
            **mb,
        })
        # same-basis GAM
        t1 = time.perf_counter()
        gam_model = HybridSplineCat(
            Xc_tr.shape[1],
            Xk_tr.shape[1] if getattr(Xk_tr, "ndim", 0) == 2 else 0,
            n_intervals=6, degree=3, lambda_s=lam, lambda_r=lambda_r,
        )
        gam_model.fit_bounds(Xc_tr)
        scores_gam, meta_gam = fit_hybrid_gam_ipp(
            gam_model, Xc_tr, Xk_tr, y_tr, Xc_te, Xk_te,
            B_train=meta_add.get("B_train"),
        )
        mg = _metrics_block(y_pa, scores_gam)
        rows.append({
            **base, "model": M_GAM,
            "converged": bool(meta_gam.get("converged", True)),
            "runtime_s": float(meta_gam.get("runtime_s", time.perf_counter() - t1)),
            "skip_reason": "",
            **mg,
        })
    except Exception as e:  # noqa: BLE001
        rows.append({
            **base, "model": M_ADD, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "skip_reason": str(e)[:300],
        })
        rows.append({
            **base, "model": M_GAM, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "skip_reason": "additive_failed",
        })

    # maxnet 10k / 50k
    for mid, mbg in ((M_MX10, 10000), (M_MX50, 50000)):
        t2 = time.perf_counter()
        try:
            scores_mx, info = fit_maxnet_arm(
                train_df, pa_env, covars, categorical, n_po, mbg, seed
            )
            mm = _metrics_block(y_pa, scores_mx) if info.get("ok", True) else {
                "auc_roc": float("nan"), "auprc": float("nan"),
                "prg": float("nan"), "cor": float("nan"),
            }
            if mid == M_MX10:
                cache["auc_maxnet_10k"] = mm["auc_roc"]
            rows.append({
                **base, "model": mid,
                "maxnet_max_bg": mbg,
                "converged": bool(info.get("ok", np.isfinite(mm["auc_roc"]))),
                "runtime_s": time.perf_counter() - t2,
                "skip_reason": "" if info.get("ok", True) else str(info)[:200],
                **mm,
            })
        except Exception as e:  # noqa: BLE001
            rows.append({
                **base, "model": mid, "maxnet_max_bg": mbg,
                "converged": False, "runtime_s": time.perf_counter() - t2,
                "auc_roc": float("nan"), "auprc": float("nan"),
                "prg": float("nan"), "cor": float("nan"),
                "skip_reason": str(e)[:300],
            })

    return rows, cache


def run_deep_species(
    region_data,
    species_id: str,
    *,
    stage: str,
    depth: int,
    mixer_input: str,
    model_id: str,
    lambda_s: float,
    lambda_path: str,
    warm_state: Optional[Dict],
    seed: int,
    lbfgs_steps: int = 12,
    hidden_width: int = 4,
) -> Dict[str, Any]:
    reg = region_data.region
    po_sp = split_species(region_data.po, species_id)
    n_po = len(po_sp)
    y_pa, pa_env, taxon_group = pa_labels_and_env_for_species(region_data, species_id)
    n_pa = int(len(y_pa))
    covars = region_data.continuous + region_data.categorical
    base = _base_row(
        reg, species_id, taxon_group, n_po, n_pa, seed, stage, lambda_s, lambda_path
    )
    base["model"] = model_id
    base["depth"] = depth
    base["mixer_input"] = mixer_input
    base["hidden_width"] = hidden_width if depth == 3 else None
    base["residual"] = True
    base["disable_silu"] = True
    base["freeze_edges"] = True
    base["warm_start"] = True

    if n_po < 5:
        return {
            **base, "n_background": 0, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "skip_reason": "n_po<5",
        }

    train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    n_bg = int((y_tr == 0).sum())
    mats = prepare_matrices(
        train_df, pa_env, list(region_data.continuous), list(region_data.categorical)
    )
    try:
        _, scores, meta = fit_deep_kan_ipp(
            mats["X_cont_train"], mats["X_cat_train"], y_tr,
            mats["X_cont_test"], mats["X_cat_test"],
            lambda_s=lambda_s, seed=seed,
            residual=True, disable_silu=True,
            warm_start_additive=warm_state is None,
            warm_start_state=warm_state,
            optimizer="lbfgs", lbfgs_steps=lbfgs_steps,
            freeze_edges_after_warmstart=True,
            depth=depth, hidden_width=hidden_width,
            mixer_input=mixer_input,
            record_loss_history=True,
        )
        mb = _metrics_block(y_pa, scores)
        return {
            **base, "n_background": n_bg,
            "converged": bool(meta.get("converged", True)),
            "runtime_s": float(meta.get("runtime_s", 0.0)),
            "final_loss": meta.get("final_loss", float("nan")),
            "last_grad_norm": meta.get("last_grad_norm", float("nan")),
            "param_l2_norm": meta.get("param_l2_norm", float("nan")),
            "loss_history": json.dumps(meta.get("loss_history") or []),
            "skip_reason": "",
            **mb,
        }
    except Exception as e:  # noqa: BLE001
        return {
            **base, "n_background": n_bg, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "skip_reason": str(e)[:300],
        }


def run_e2e_species(
    region_data,
    species_id: str,
    *,
    lambda_s: float,
    lambda_path: str,
    seed: int = 0,
    maxnet_auc: Optional[float] = None,
    adam_lr: float = 0.03,
    adam_steps: int = 150,
    lbfgs_steps: int = 10,
    disable_silu: bool = False,
    remediation: str = "primary",
) -> Dict[str, Any]:
    reg = region_data.region
    po_sp = split_species(region_data.po, species_id)
    n_po = len(po_sp)
    y_pa, pa_env, taxon_group = pa_labels_and_env_for_species(region_data, species_id)
    n_pa = int(len(y_pa))
    covars = region_data.continuous + region_data.categorical
    base = _base_row(
        reg, species_id, taxon_group, n_po, n_pa, seed, "D", lambda_s, lambda_path
    )
    base["model"] = M_SKAN
    base["remediation"] = remediation
    base["adam_lr"] = adam_lr
    base["disable_silu"] = disable_silu

    if n_po < 5:
        return {
            **base, "n_background": 0, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "skip_reason": "n_po<5", "failed": True,
        }

    train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    n_bg = int((y_tr == 0).sum())
    mats = prepare_matrices(
        train_df, pa_env, list(region_data.continuous), list(region_data.categorical)
    )
    try:
        _, scores, meta = fit_standard_kan_ipp(
            mats["X_cont_train"], mats["X_cat_train"], y_tr,
            mats["X_cont_test"], mats["X_cat_test"],
            seed=seed, adam_lr=adam_lr, adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps, disable_silu=disable_silu,
            adaptive_budget=True,
        )
        mb = _metrics_block(y_pa, scores)
        failed = is_e2e_failure(
            converged=bool(meta.get("converged", False)),
            final_loss=float(meta.get("final_loss", float("nan"))),
            auc=float(mb["auc_roc"]),
            maxnet_auc=maxnet_auc,
        )
        return {
            **base, "n_background": n_bg,
            "converged": bool(meta.get("converged", True)),
            "runtime_s": float(meta.get("runtime_s", 0.0)),
            "final_loss": meta.get("final_loss", float("nan")),
            "last_grad_norm": meta.get("last_grad_norm", float("nan")),
            "param_l2_norm": meta.get("param_l2_norm", float("nan")),
            "loss_history": json.dumps(meta.get("loss_history") or []),
            "steps_adam": meta.get("steps_adam"),
            "steps_lbfgs": meta.get("steps_lbfgs"),
            "failed": failed,
            "skip_reason": "",
            **mb,
        }
    except Exception as e:  # noqa: BLE001
        return {
            **base, "n_background": n_bg, "converged": False, "runtime_s": 0.0,
            "auc_roc": float("nan"), "auprc": float("nan"), "prg": float("nan"),
            "cor": float("nan"), "failed": True, "skip_reason": str(e)[:300],
        }


def remediate_e2e(
    region_data, species_id: str, *, lambda_s: float, lambda_path: str,
    maxnet_auc: Optional[float], primary_seed: int = 0,
) -> List[Dict]:
    """Primary fit + up to R1/R2 if failure (Closure §1.4)."""
    rows: List[Dict] = []
    r0 = run_e2e_species(
        region_data, species_id,
        lambda_s=lambda_s, lambda_path=lambda_path, seed=primary_seed,
        maxnet_auc=maxnet_auc, remediation="primary",
    )
    rows.append(r0)
    if not r0.get("failed"):
        return rows

    # R1
    r1 = run_e2e_species(
        region_data, species_id,
        lambda_s=lambda_s, lambda_path=lambda_path,
        seed=1000 + primary_seed, maxnet_auc=maxnet_auc,
        adam_lr=0.01, adam_steps=int(150 * 1.5), lbfgs_steps=int(10 * 1.5),
        remediation="R1",
    )
    rows.append(r1)
    if not r1.get("failed"):
        return rows

    # R2
    r2 = run_e2e_species(
        region_data, species_id,
        lambda_s=lambda_s, lambda_path=lambda_path,
        seed=2000 + primary_seed, maxnet_auc=maxnet_auc,
        adam_lr=0.005, adam_steps=int(150 * 1.5), lbfgs_steps=int(10 * 1.5),
        disable_silu=True, remediation="R2",
    )
    rows.append(r2)
    return rows


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=x.size, replace=True).mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_summaries(outdir: Path, metrics: pd.DataFrame) -> None:
    lines = [
        f"# Methodological Closure v1 — SUMMARY",
        "",
        f"**Run dir:** `{outdir.name}`",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**Rows:** {len(metrics)}",
        "",
    ]
    if metrics.empty:
        lines.append("(no metrics yet)")
        (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
        return

    # Stage A: additive vs maxnet
    a = metrics[metrics["stage"] == "A"]
    if len(a):
        lines.append("## Stage A — additive(λ*) vs maxnet")
        lines.append("")
        piv = a.pivot_table(
            index=["region", "species_id"], columns="model", values="auc_roc", aggfunc="first"
        )
        if M_ADD in piv.columns and M_MX10 in piv.columns:
            d = (piv[M_ADD] - piv[M_MX10]).dropna()
            mean, lo, hi = bootstrap_mean_ci(d.values)
            lines.append(
                f"- Additive(λ*) − maxnet@10k: mean ΔAUC = **{mean:+.4f}** "
                f"[{lo:+.4f}, {hi:+.4f}] (n={len(d)})"
            )
        if M_ADD in piv.columns and M_MX50 in piv.columns:
            d = (piv[M_ADD] - piv[M_MX50]).dropna()
            mean, lo, hi = bootstrap_mean_ci(d.values)
            lines.append(
                f"- Additive(λ*) − maxnet@50k: mean ΔAUC = **{mean:+.4f}** "
                f"[{lo:+.4f}, {hi:+.4f}] (n={len(d)})"
            )
        if M_ADD in piv.columns and M_GAM in piv.columns:
            d = (piv[M_ADD] - piv[M_GAM]).dropna()
            mean, lo, hi = bootstrap_mean_ci(d.values)
            lines.append(
                f"- Additive(λ*) − same-basis GAM: mean ΔAUC = **{mean:+.4f}** "
                f"[{lo:+.4f}, {hi:+.4f}] (n={len(d)}); Pearson r = "
                f"{piv[[M_ADD, M_GAM]].dropna().corr().iloc[0,1]:.6f}"
                if len(piv[[M_ADD, M_GAM]].dropna()) > 2 else
                f"- Additive(λ*) − GAM: mean Δ = {mean:+.4f}"
            )
        # lambda distribution
        if "lambda_s" in a.columns:
            la = a[a["model"] == M_ADD][["species_id", "lambda_s", "lambda_selection"]].drop_duplicates()
            lines.append("")
            lines.append("### λ* distribution (additive rows)")
            lines.append(f"- path counts: {la['lambda_selection'].value_counts().to_dict()}")
            lines.append(f"- λ* value counts: {la['lambda_s'].value_counts().sort_index().to_dict()}")
        lines.append("")

    # Deep stages
    for stage, dmodels, label in (
        ("B", DEEP2_MODELS, "Deep-2"),
        ("C", DEEP3_MODELS, "Deep-3"),
    ):
        sub = metrics[metrics["stage"] == stage]
        if sub.empty:
            continue
        lines.append(f"## Stage {stage} — fair {label} @ λ*")
        lines.append("")
        add_map = (
            metrics[(metrics.stage == "A") & (metrics.model == M_ADD)]
            .set_index(["region", "species_id"])["auc_roc"]
        )
        for dm in dmodels:
            sd = sub[sub.model == dm]
            if sd.empty:
                continue
            # mean over seeds within species
            g = sd.groupby(["region", "species_id"])["auc_roc"].mean()
            common = g.index.intersection(add_map.index)
            if len(common) == 0:
                continue
            delta = (g.loc[common] - add_map.loc[common]).values
            mean, lo, hi = bootstrap_mean_ci(delta)
            lines.append(
                f"- **{dm}** − Additive(λ*): mean ΔAUC = **{mean:+.4f}** "
                f"[{lo:+.4f}, {hi:+.4f}] (n_sp={len(common)})"
            )
            # by region
            reg_lines = []
            for reg, idx in g.groupby(level=0).groups.items():
                # rebuild properly
                pass
            reg_df = pd.DataFrame({
                "region": [i[0] for i in common],
                "delta": delta,
            })
            for reg, part in reg_df.groupby("region"):
                m, l, h = bootstrap_mean_ci(part["delta"].values)
                reg_lines.append(f"  - {reg}: {m:+.4f} [{l:+.4f}, {h:+.4f}] n={len(part)}")
            lines.extend(reg_lines)
            lines.append("")

    # Stage D e2e — use last remediation per species if failed chain
    dsub = metrics[metrics["stage"] == "D"]
    if len(dsub):
        lines.append("## Stage D — Standard KAN e2e (ITT after remediation)")
        lines.append("")
        # pick best successful remediation else last
        picks = []
        for (reg, sp), g in dsub.groupby(["region", "species_id"]):
            ok = g[g["failed"] == False] if "failed" in g.columns else g
            if len(ok):
                # prefer primary, then R1, then R2
                order = {"primary": 0, "R1": 1, "R2": 2}
                ok = ok.copy()
                ok["_ord"] = ok["remediation"].map(lambda x: order.get(str(x), 9))
                picks.append(ok.sort_values("_ord").iloc[0])
            else:
                picks.append(g.iloc[-1])
        if picks:
            pdf = pd.DataFrame(picks)
            add_map = (
                metrics[(metrics.stage == "A") & (metrics.model == M_ADD)]
                .set_index(["region", "species_id"])["auc_roc"]
            )
            mx_map = (
                metrics[(metrics.stage == "A") & (metrics.model == M_MX10)]
                .set_index(["region", "species_id"])["auc_roc"]
            )
            idx = pd.MultiIndex.from_frame(pdf[["region", "species_id"]])
            common_a = idx.intersection(add_map.index)
            if len(common_a):
                # align
                pdf2 = pdf.set_index(["region", "species_id"])
                da = (pdf2.loc[common_a, "auc_roc"] - add_map.loc[common_a]).values
                m, l, h = bootstrap_mean_ci(da)
                lines.append(
                    f"- e2e − Additive(λ*): mean ΔAUC = **{m:+.4f}** [{l:+.4f}, {h:+.4f}]"
                )
            common_m = idx.intersection(mx_map.index)
            if len(common_m):
                pdf2 = pdf.set_index(["region", "species_id"])
                dm = (pdf2.loc[common_m, "auc_roc"] - mx_map.loc[common_m]).values
                m, l, h = bootstrap_mean_ci(dm)
                lines.append(
                    f"- e2e − maxnet@10k: mean ΔAUC = **{m:+.4f}** [{l:+.4f}, {h:+.4f}]"
                )
            n_fail = int(pdf["failed"].sum()) if "failed" in pdf.columns else -1
            lines.append(f"- species still failed after R2: {n_fail}/{len(pdf)}")
            lines.append(
                f"- remediation used: {pdf['remediation'].value_counts().to_dict()}"
            )
        lines.append("")

    lines.append("*Methodological Closure v1*")
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Methodological Closure v1 full runner")
    p.add_argument("--regions", default=",".join(ALL_REGIONS))
    p.add_argument("--species", default="", help="comma list; default all in region")
    p.add_argument(
        "--stages",
        default="A,B,C,D,E",
        help="comma list among A,B,C,D,E",
    )
    p.add_argument("--seeds-deep", default="0,1,2")
    p.add_argument("--seed-stage-a", type=int, default=0)
    p.add_argument("--outdir", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--lbfgs-steps", type=int, default=12)
    p.add_argument("--additive-steps", type=int, default=15)
    p.add_argument(
        "--skip-maxnet-50k",
        action="store_true",
        help="skip maxnet 50k arm (still run 10k)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "process-pool size for stages B/C/D (independent species jobs). "
            "0 = auto: floor(0.8 * n_cpu), leaving ~20%% cores free for interactive use. "
            "1 = serial."
        ),
    )
    p.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP/torch threads per worker (default 1; avoid oversubscription)",
    )
    args = p.parse_args()

    regions = tuple(x.strip().upper() for x in args.regions.split(",") if x.strip())
    stages = {x.strip().upper() for x in args.stages.split(",") if x.strip()}
    seeds_deep = tuple(int(x) for x in args.seeds_deep.split(",") if x.strip() != "")

    n_cpu = os.cpu_count() or 4
    if int(args.workers) <= 0:
        # leave ~20% of logical CPUs free for the desktop / interactive use
        n_workers = max(1, int(n_cpu * 0.8))
    else:
        n_workers = max(1, int(args.workers))
    # hard cap: never take more than 80% even if user passes a large number
    n_workers_cap = max(1, int(n_cpu * 0.8))
    if n_workers > n_workers_cap:
        print(
            f"Note: clamping --workers {n_workers} → {n_workers_cap} "
            f"(≤80% of {n_cpu} CPUs)",
            flush=True,
        )
        n_workers = n_workers_cap
    blas_threads = max(1, int(args.blas_threads))

    if args.outdir:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            s = str(outdir).replace("\\", "/")
            outdir = ROOT / s if not s.startswith("/") else Path(s)
            if not str(outdir).startswith(str(ROOT)):
                outdir = ROOT / args.outdir
    else:
        outdir = ROOT / "outputs" / f"methodological_closure_{_now_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "status").mkdir(exist_ok=True)
    (outdir / "lambda").mkdir(exist_ok=True)

    metrics_path = outdir / "metrics.csv"
    lambda_path = outdir / "lambda_star.csv"
    cache_path = outdir / "species_cache.json"

    all_rows: List[Dict] = []
    if args.resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_rows = prev.to_dict(orient="records")
        print(f"Resumed {len(all_rows)} metric rows", flush=True)

    done: Set[Tuple] = {_row_key(r) for r in all_rows}

    # species cache: lambda + warm_state + maxnet auc (JSON; numpy via lists)
    species_cache: Dict[str, Any] = {}
    if cache_path.is_file():
        species_cache = json.loads(cache_path.read_text(encoding="utf-8"))

    def save_metrics() -> None:
        pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    def save_cache() -> None:
        cache_path.write_text(json.dumps(species_cache, indent=2), encoding="utf-8")

    def append_rows(rows: List[Dict]) -> None:
        for r in rows:
            k = _row_key(r)
            if k in done:
                continue
            all_rows.append(r)
            done.add(k)
        save_metrics()

    manifest = {
        "protocol": PROTOCOL_ID,
        "plan": "docs/plans/Plan_Methodological_Closure_v1.md",
        "regions": list(regions),
        "stages": sorted(stages),
        "seeds_deep": list(seeds_deep),
        "lambda_grid": list(CLOSURE_LAMBDA_GRID),
        "k_folds": CLOSURE_K_FOLDS,
        "min_po_for_cv": CLOSURE_MIN_PO_FOR_CV,
        "workers_bcd": n_workers,
        "blas_threads_per_worker": blas_threads,
        "n_cpu": n_cpu,
        "cpu_reserve_frac": 0.2,
        "created_unix": time.time(),
        "outdir": str(outdir),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Closure v1 → {outdir}", flush=True)
    print(
        f"Regions={regions} stages={sorted(stages)} deep_seeds={seeds_deep} "
        f"B/C/D workers={n_workers}/{n_cpu} CPUs (~20% reserved) "
        f"BLAS/worker={blas_threads}",
        flush=True,
    )

    # ---------- Stage A ----------
    if "A" in stages:
        print("=== Stage A: λ* + additive + GAM + maxnet ===", flush=True)
        lambda_rows: List[Dict] = []
        for reg in regions:
            data = load_region(reg)
            species = (
                [s.strip() for s in args.species.split(",") if s.strip()]
                if args.species.strip()
                else list(data.species_ids)
            )
            print(f"== {reg}: {len(species)} species ==", flush=True)
            for sp in species:
                # skip if all stage A models done
                need = [M_ADD, M_GAM, M_MX10] + ([] if args.skip_maxnet_50k else [M_MX50])
                if all(
                    _row_key({
                        "stage": "A", "region": reg, "species_id": sp,
                        "seed": args.seed_stage_a, "model": m, "remediation": "primary",
                    }) in done
                    for m in need
                ):
                    print(f"  skip A {reg}/{sp}", flush=True)
                    # ensure cache exists
                    ck = f"{reg}::{sp}"
                    if ck not in species_cache:
                        # reconstruct lambda from metrics
                        sub = [r for r in all_rows if r.get("stage") == "A"
                               and r.get("region") == reg and r.get("species_id") == sp
                               and r.get("model") == M_ADD]
                        if sub:
                            species_cache[ck] = {
                                "lambda_s": float(sub[0].get("lambda_s", 1e-2)),
                                "lambda_path": str(sub[0].get("lambda_selection", "")),
                                "auc_maxnet_10k": float("nan"),
                                "warm_state": None,
                            }
                    continue

                print(f"  A {reg}/{sp} ...", flush=True)
                t0 = time.perf_counter()
                rows, cache = run_stage_a_species(
                    data, sp, seed=args.seed_stage_a, additive_steps=args.additive_steps,
                )
                if args.skip_maxnet_50k:
                    rows = [r for r in rows if r.get("model") != M_MX50]
                append_rows(rows)
                # serialize warm_state
                ws = cache.get("warm_state")
                ws_ser = None
                if ws is not None:
                    ws_ser = {
                        "spline_coeffs": np.asarray(ws["spline_coeffs"]).tolist(),
                        "beta_cat": None
                        if ws.get("beta_cat") is None
                        else np.asarray(ws["beta_cat"]).tolist(),
                    }
                ck = f"{reg}::{sp}"
                species_cache[ck] = {
                    "lambda_s": cache["lambda_s"],
                    "lambda_path": cache["lambda_path"],
                    "auc_maxnet_10k": cache.get("auc_maxnet_10k", float("nan")),
                    "auc_additive": cache.get("auc_additive", float("nan")),
                    "warm_state": ws_ser,
                    "lambda_sel": {
                        k: (dict(v) if isinstance(v, dict) else v)
                        for k, v in (cache.get("lambda_sel") or {}).items()
                        if k != "fold_scores"
                    },
                }
                save_cache()
                lambda_rows.append({
                    "region": reg, "species_id": sp,
                    "lambda_s": cache["lambda_s"],
                    "lambda_path": cache["lambda_path"],
                    "n_presence": next(
                        (r["n_presence"] for r in rows if r.get("model") == M_ADD), 0
                    ),
                })
                pd.DataFrame(lambda_rows).to_csv(lambda_path, index=False)
                print(
                    f"    λ*={cache['lambda_s']} path={cache['lambda_path']} "
                    f"t={time.perf_counter()-t0:.1f}s",
                    flush=True,
                )

    # helper to get cache / rebuild warm state
    def get_species_assets(reg: str, sp: str, data) -> Dict[str, Any]:
        ck = f"{reg}::{sp}"
        if ck in species_cache and species_cache[ck].get("warm_state") is not None:
            c = species_cache[ck]
            ws = c["warm_state"]
            warm = {
                "spline_coeffs": np.asarray(ws["spline_coeffs"], dtype=np.float64),
                "beta_cat": None
                if ws.get("beta_cat") is None
                else np.asarray(ws["beta_cat"], dtype=np.float64),
            }
            return {
                "lambda_s": float(c["lambda_s"]),
                "lambda_path": str(c["lambda_path"]),
                "warm_state": warm,
                "auc_maxnet_10k": c.get("auc_maxnet_10k"),
            }
        # re-fit additive for warm state
        print(f"  (rebuild cache) {reg}/{sp}", flush=True)
        rows, cache = run_stage_a_species(
            data, sp, seed=args.seed_stage_a, additive_steps=args.additive_steps,
        )
        # only store if A not already done
        append_rows(rows)
        ws = cache.get("warm_state")
        ws_ser = None
        if ws is not None:
            ws_ser = {
                "spline_coeffs": np.asarray(ws["spline_coeffs"]).tolist(),
                "beta_cat": None
                if ws.get("beta_cat") is None
                else np.asarray(ws["beta_cat"]).tolist(),
            }
        species_cache[ck] = {
            "lambda_s": cache["lambda_s"],
            "lambda_path": cache["lambda_path"],
            "auc_maxnet_10k": cache.get("auc_maxnet_10k", float("nan")),
            "auc_additive": cache.get("auc_additive", float("nan")),
            "warm_state": ws_ser,
        }
        save_cache()
        return {
            "lambda_s": cache["lambda_s"],
            "lambda_path": cache["lambda_path"],
            "warm_state": cache.get("warm_state"),
            "auc_maxnet_10k": cache.get("auc_maxnet_10k"),
        }

    # ---------- Stages B / C (parallel across independent jobs) ----------
    deep_specs = []
    if "B" in stages:
        for mi, mid in (("phi", M_D2_PHI), ("raw_scaled", M_D2_RX)):
            deep_specs.append(("B", 2, mi, mid))
    if "C" in stages:
        for mi, mid in (("phi", M_D3_PHI), ("raw_scaled", M_D3_RX)):
            deep_specs.append(("C", 3, mi, mid))

    if deep_specs:
        print(
            f"=== Stages B/C: fair residual depth "
            f"(workers={n_workers}, ~20% CPU reserved) ===",
            flush=True,
        )
        pending_deep: List[Dict[str, Any]] = []
        for reg in regions:
            data = load_region(reg)
            species = (
                [s.strip() for s in args.species.split(",") if s.strip()]
                if args.species.strip()
                else list(data.species_ids)
            )
            for sp in species:
                # ensure λ* + warm_state before dispatching (serial; cheap if cached)
                assets = get_species_assets(reg, sp, data)
                ws = assets.get("warm_state")
                ws_ser = None
                if ws is not None:
                    ws_ser = {
                        "spline_coeffs": np.asarray(ws["spline_coeffs"]).tolist(),
                        "beta_cat": None
                        if ws.get("beta_cat") is None
                        else np.asarray(ws["beta_cat"]).tolist(),
                    }
                for seed in seeds_deep:
                    for stage, depth, mi, mid in deep_specs:
                        k = _row_key({
                            "stage": stage, "region": reg, "species_id": sp,
                            "seed": seed, "model": mid, "remediation": "primary",
                        })
                        if k in done:
                            continue
                        pending_deep.append({
                            "region": reg,
                            "species_id": sp,
                            "stage": stage,
                            "depth": depth,
                            "mixer_input": mi,
                            "model_id": mid,
                            "seed": seed,
                            "lambda_s": float(assets["lambda_s"]),
                            "lambda_path": str(assets["lambda_path"]),
                            "warm_state": ws_ser,
                            "lbfgs_steps": args.lbfgs_steps,
                            "hidden_width": 4,
                        })

        def _on_deep(res: Dict[str, Any]) -> None:
            append_rows(res.get("rows") or [])
            row0 = (res.get("rows") or [{}])[0]
            print(
                f"    {res.get('label')}: AUC={row0.get('auc_roc')} "
                f"conv={row0.get('converged')} wall={res.get('wall_s'):.1f}s",
                flush=True,
            )

        _run_jobs_parallel(
            pending_deep,
            workers=n_workers,
            worker_fn=_worker_deep_job,
            blas_threads=blas_threads,
            on_result=_on_deep,
            label="B/C deep residual",
        )

    # ---------- Stage D (parallel across species; remediation stays sequential per species) ----------
    if "D" in stages:
        print(
            f"=== Stage D: Standard KAN e2e + remediation "
            f"(workers={n_workers}, ~20% CPU reserved) ===",
            flush=True,
        )
        pending_e2e: List[Dict[str, Any]] = []
        for reg in regions:
            data = load_region(reg)
            species = (
                [s.strip() for s in args.species.split(",") if s.strip()]
                if args.species.strip()
                else list(data.species_ids)
            )
            for sp in species:
                existing = [
                    r for r in all_rows
                    if r.get("stage") == "D" and r.get("region") == reg
                    and r.get("species_id") == sp and r.get("model") == M_SKAN
                ]
                if existing:
                    if any(not r.get("failed", True) for r in existing):
                        continue
                    if any(r.get("remediation") == "R2" for r in existing):
                        continue
                assets = get_species_assets(reg, sp, data)
                mx = assets.get("auc_maxnet_10k")
                try:
                    mx_f = float(mx) if mx is not None else None
                    if mx_f is not None and not np.isfinite(mx_f):
                        mx_f = None
                except (TypeError, ValueError):
                    mx_f = None
                pending_e2e.append({
                    "region": reg,
                    "species_id": sp,
                    "lambda_s": float(assets["lambda_s"]),
                    "lambda_path": str(assets["lambda_path"]),
                    "auc_maxnet_10k": mx_f,
                    "primary_seed": 0,
                })

        def _on_e2e(res: Dict[str, Any]) -> None:
            append_rows(res.get("rows") or [])
            print(
                f"    {res.get('label')}: rem={res.get('final_rem')} "
                f"AUC={res.get('auc')} failed={res.get('failed')} "
                f"wall={res.get('wall_s'):.1f}s",
                flush=True,
            )

        _run_jobs_parallel(
            pending_e2e,
            workers=n_workers,
            worker_fn=_worker_e2e_job,
            blas_threads=blas_threads,
            on_result=_on_e2e,
            label="D standard KAN e2e",
        )

    # ---------- Stage E ----------
    if "E" in stages or True:
        metrics = pd.DataFrame(all_rows)
        if len(metrics):
            metrics.to_csv(metrics_path, index=False)
            write_summaries(outdir, metrics)
            # master wide table at species level (seed 0 / mean deep)
            try:
                a = metrics[metrics.stage == "A"]
                wide = a.pivot_table(
                    index=["region", "species_id", "lambda_s", "lambda_selection"],
                    columns="model",
                    values="auc_roc",
                    aggfunc="first",
                ).reset_index()
                wide.to_csv(outdir / "master_stageA_auc.csv", index=False)
            except Exception as e:  # noqa: BLE001
                print("master table warn:", e, flush=True)
        print("Done.", outdir, flush=True)


if __name__ == "__main__":
    # spawn is safer with PyTorch under process pools than fork
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    # mild renice so interactive desktop keeps priority (~20% cores already reserved)
    try:
        os.nice(10)
    except Exception:
        pass
    main()
