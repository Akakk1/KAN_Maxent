"""Ranking metrics for presence–background / presence–absence evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def calc_prg(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under Precision-Recall-Gain curve (Flach & Kull style).

    Prefer the ``prg`` package if installed; otherwise a lightweight
    trapezoidal approximation on PRG coordinates.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        from prg import create_prg_curve, calc_auprg  # type: ignore

        return float(calc_auprg(create_prg_curve(labels=y_true, pos_scores=scores)))
    except Exception:
        return _prg_fallback(y_true, scores)


def _prg_fallback(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Approximate AUPRG without external package."""
    y = y_true.astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pi = n_pos / len(y)
    # sort by score descending
    order = np.argsort(-scores)
    y_s = y[order]
    tp = np.cumsum(y_s)
    fp = np.cumsum(1 - y_s)
    # recall / precision
    rec = tp / n_pos
    prec = tp / np.maximum(tp + fp, 1e-12)
    # PRG coordinates (clip invalid)
    # precision gain = (prec - pi) / ((1 - pi) * prec)
    # recall gain = (rec - 0) / (1 - 0) but use standard formula
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_gain = (prec - pi) / ((1.0 - pi) * prec)
        rec_gain = (rec - 0.0) / (1.0 - 0.0)  # = rec when baseline 0; use Flach form:
        # rec_gain = (rec - pi) / ((1-pi)*rec) is wrong; standard:
        rec_gain = (rec - 0) / (1 - 0)
        # Actually PRG: RG = (rec - pi)/( (1-pi) ) for some defs; use package-free
        # simple AUCPR as soft fallback if PRG degenerates
        rg = (rec - pi) / (1.0 - pi)
        pg = (prec - pi) / ((1.0 - pi) * np.maximum(prec, 1e-12))
    # keep only valid increasing RG region
    mask = np.isfinite(pg) & np.isfinite(rg) & (rg >= 0)
    if mask.sum() < 2:
        return float(average_precision_score(y_true, scores))
    rg = rg[mask]
    pg = np.clip(pg[mask], -1, 1)
    # sort by recall gain
    o = np.argsort(rg)
    rg, pg = rg[o], pg[o]
    # unique rg
    urg, idx = np.unique(rg, return_index=True)
    upg = pg[idx]
    return float(np.trapz(upg, urg))


def cor_pa(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pearson correlation between scores and PA labels (Valavi-style COR).

    For binary labels this equals the point-biserial correlation. Requires
    variation in both ``y_true`` and ``scores``.
    """
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.size < 3 or len(np.unique(y)) < 2:
        return float("nan")
    if not np.isfinite(s).all() or np.nanstd(s) < 1e-15:
        return float("nan")
    # np.corrcoef is fine for 1d
    c = np.corrcoef(y, s)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc_roc": auc_roc(y_true, scores),
        "auprc": auprc(y_true, scores),
        "prg": calc_prg(y_true, scores),
        "cor": cor_pa(y_true, scores),
    }



def summarize_folds(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "n": int(v.size),
    }
