"""PO-internal random k-fold λs selection (never uses PA labels)."""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from kanmaxent.evaluation.metrics import auc_roc


# Methodological Closure v1 defaults (Plan_Methodological_Closure_v1.md)
CLOSURE_LAMBDA_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
CLOSURE_K_FOLDS: int = 5
CLOSURE_MIN_PO_FOR_CV: int = 30
CLOSURE_DEFAULT_LAMBDA: float = 1e-2


def select_lambda_s_po_cv(
    n_po: int,
    n_bg: int,
    fit_eval_fn: Callable[[np.ndarray, np.ndarray, float], float],
    *,
    grid: Sequence[float] = CLOSURE_LAMBDA_GRID,
    k_folds: int = CLOSURE_K_FOLDS,
    min_po_for_cv: int = CLOSURE_MIN_PO_FOR_CV,
    default_lambda: float = CLOSURE_DEFAULT_LAMBDA,
    seed: int = 0,
) -> Dict[str, object]:
    """Select λs by random k-fold on presence indices.

    Parameters
    ----------
    fit_eval_fn
        Callable (po_train_idx, po_val_idx, lambda_s) -> AUC on
        val-PO vs background (or similar PO-internal score). Indices are
        into the presence block [0, n_po).
    """
    if n_po < 5:
        return {
            "lambda_s": float(default_lambda),
            "path": "skipped_n_po_lt_5",
            "scores": {},
            "k_folds": int(k_folds),
            "grid": [float(x) for x in grid],
        }
    if n_po < min_po_for_cv:
        return {
            "lambda_s": float(default_lambda),
            "path": f"frozen_n_po_lt_{int(min_po_for_cv)}",
            "scores": {},
            "k_folds": int(k_folds),
            "grid": [float(x) for x in grid],
        }

    rng = np.random.default_rng(seed)
    idx = np.arange(n_po)
    rng.shuffle(idx)
    folds = np.array_split(idx, k_folds)

    mean_scores: Dict[float, float] = {}
    fold_detail: Dict[str, List[float]] = {}
    for lam in grid:
        fold_aucs = []
        for f in range(k_folds):
            val = folds[f]
            tr = np.concatenate([folds[i] for i in range(k_folds) if i != f])
            if len(val) == 0 or len(tr) == 0:
                continue
            try:
                a = fit_eval_fn(tr, val, float(lam))
            except Exception:
                a = float("nan")
            if np.isfinite(a):
                fold_aucs.append(float(a))
        mean_scores[float(lam)] = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        fold_detail[str(float(lam))] = fold_aucs

    # pick best finite; ties → smaller λ (prefer smoother)
    best_lam = float(default_lambda)
    best_sc = -np.inf
    for lam in sorted(mean_scores.keys()):
        sc = mean_scores[lam]
        if np.isfinite(sc) and sc > best_sc + 1e-15:
            best_sc = sc
            best_lam = float(lam)

    return {
        "lambda_s": float(best_lam),
        "path": f"po_random_{int(k_folds)}fold",
        "scores": mean_scores,
        "fold_scores": fold_detail,
        "best_score": float(best_sc) if np.isfinite(best_sc) else float("nan"),
        "k_folds": int(k_folds),
        "grid": [float(x) for x in grid],
    }
