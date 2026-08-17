"""Ginkgo outer 5-fold spatial CV orchestration (Phase 1)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from kanmaxent.data.ginkgo_io import env_columns, extract_xy, load_ginkgo, split_outer_fold
from kanmaxent.data.preprocess import FoldPreprocessor
from kanmaxent.evaluation.metrics import ranking_metrics
from kanmaxent.interpret.curves import export_component_curves
from kanmaxent.reference.maxnet_r import fit_predict_maxnet
from kanmaxent.trainers import fit_bce_kan, fit_gam_ipp_shared, fit_ipp_kan
from kanmaxent.utils import as_numpy_f64


ROW_FIELDS = [
    "region",
    "taxon_group",
    "species_id",
    "split_id",
    "seed",
    "estimand",
    "model",
    "objective",
    "background_scheme",
    "integration_support",
    "n_presence",
    "n_background",
    "n_PA_test",
    "basis_df",
    "knots",
    "lambda_s",
    "lambda_r",
    "weight_decay",
    "lambda_selection",
    "converged",
    "runtime_s",
    "auc_roc",
    "auprc",
    "artifact_path",
]


def _base_row(**kwargs) -> Dict[str, Any]:
    row = {
        "region": "China_Ginkgo",
        "taxon_group": "plant",
        "species_id": "Ginkgo_biloba",
        "estimand": "global_conditional_density",
        "background_scheme": "ginkgo_table_bg",
        "integration_support": "outer_train_rows",
        "n_PA_test": 0,
        "weight_decay": 0.0,
        "lambda_selection": "frozen_default",
    }
    row.update(kwargs)
    return row


def run_ginkgo_cv(
    data_path: Optional[str] = None,
    outdir: str | Path = "outputs/ginkgo_cv_run",
    *,
    seeds: Sequence[int] = (0, 1, 2),
    folds: Optional[Sequence[int]] = None,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    lbfgs_steps: int = 35,
    models: Sequence[str] = ("kan_ipp", "gam_ipp", "kan_bce", "maxnet"),
    export_curves: bool = True,
    check_sha256: bool = True,
) -> pd.DataFrame:
    """Run Phase 1 outer CV; write metrics.csv, manifest.json, fold artifacts."""
    outdir = Path(outdir)
    if outdir.exists() and any(outdir.iterdir()):
        # never overwrite: require empty or new path
        raise FileExistsError(
            f"outdir {outdir} already exists and is non-empty; choose a new run_id"
        )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "folds").mkdir(exist_ok=True)
    (outdir / "curves").mkdir(exist_ok=True)

    df = load_ginkgo(data_path, check_sha256=check_sha256)
    cols = env_columns(df)
    if folds is None:
        folds = sorted(int(x) for x in df["fold"].unique())

    rows: List[Dict[str, Any]] = []
    basis_df = n_intervals + degree

    for f in folds:
        split = split_outer_fold(df, f)
        X_all, y_all, _ = extract_xy(df)
        X_tr_raw = X_all[split.train_idx]
        y_tr = y_all[split.train_idx]
        X_te_raw = X_all[split.test_idx]
        y_te = y_all[split.test_idx]

        # Per-fold scaler (KAN/GAM/BCE only)
        pre = FoldPreprocessor(meta={"outer_fold": f})
        X_tr = pre.fit_transform(X_tr_raw)
        X_te = pre.transform(X_te_raw)

        fold_dir = outdir / "folds" / f"fold_{f}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            fold_dir / "preprocess.npz",
            mean=pre.mean_,
            scale=pre.scale_,
            train_idx=split.train_idx,
            test_idx=split.test_idx,
        )

        # --- maxnet: raw env, deterministic; still loop seeds for schema parity ---
        if "maxnet" in models:
            for seed in seeds:
                t0 = time.perf_counter()
                scores, info = fit_predict_maxnet(
                    X_tr_raw,
                    y_tr,
                    X_te_raw,
                    cols,
                    regmult=1.0,
                )
                rt = time.perf_counter() - t0
                m = ranking_metrics(y_te, scores) if info.get("ok") else {"auc_roc": float("nan"), "auprc": float("nan")}
                art = fold_dir / f"maxnet_seed{seed}_pred.csv"
                pd.DataFrame({"y": y_te, "score": scores}).to_csv(art, index=False)
                rows.append(
                    _base_row(
                        split_id=f,
                        seed=int(seed),
                        model="maxnet",
                        objective="maxnet_default",
                        n_presence=split.n_train_presence,
                        n_background=split.n_train_background,
                        basis_df="",
                        knots="",
                        lambda_s="",
                        lambda_r="",
                        lambda_selection="maxnet_default_regmult1",
                        converged=bool(info.get("ok")),
                        runtime_s=round(rt, 4),
                        auc_roc=m["auc_roc"],
                        auprc=m["auprc"],
                        artifact_path=str(art.relative_to(outdir)),
                    )
                )
                # maxnet is deterministic: one fit is enough; still write seed rows
                if seed == seeds[0] and not info.get("ok"):
                    (fold_dir / "maxnet_error.txt").write_text(str(info), encoding="utf-8")

        for seed in seeds:
            # KAN IPP
            if "kan_ipp" in models:
                model_ipp, res_ipp = fit_ipp_kan(
                    X_tr,
                    y_tr,
                    X_te,
                    n_intervals=n_intervals,
                    degree=degree,
                    lambda_s=lambda_s,
                    lambda_r=lambda_r,
                    seed=int(seed),
                    steps=lbfgs_steps,
                )
                m = ranking_metrics(y_te, res_ipp.scores_test)
                art = fold_dir / f"kan_ipp_seed{seed}_pred.csv"
                pd.DataFrame({"y": y_te, "score": res_ipp.scores_test}).to_csv(art, index=False)
                rows.append(
                    _base_row(
                        split_id=f,
                        seed=int(seed),
                        model="additive_kan_ipp",
                        objective="ipp",
                        n_presence=split.n_train_presence,
                        n_background=split.n_train_background,
                        basis_df=basis_df,
                        knots=f"G={n_intervals},K={degree}",
                        lambda_s=lambda_s,
                        lambda_r=lambda_r,
                        converged=res_ipp.converged,
                        runtime_s=round(res_ipp.runtime_s, 4),
                        auc_roc=m["auc_roc"],
                        auprc=m["auprc"],
                        artifact_path=str(art.relative_to(outdir)),
                    )
                )

                # GAM shares same fit_bounds / design matrix as this KAN shell
                if "gam_ipp" in models:
                    B_tr = res_ipp.extras.get("B_train")
                    res_gam = fit_gam_ipp_shared(
                        model_ipp,
                        X_tr,
                        y_tr,
                        X_te,
                        lambda_s=lambda_s,
                        lambda_r=lambda_r,
                        B_train=B_tr,
                    )

                    m_g = ranking_metrics(y_te, res_gam.scores_test)
                    art_g = fold_dir / f"gam_ipp_seed{seed}_pred.csv"
                    pd.DataFrame({"y": y_te, "score": res_gam.scores_test}).to_csv(art_g, index=False)
                    # Spearman vs KAN
                    rho, _ = spearmanr(res_ipp.scores_test, res_gam.scores_test)
                    rows.append(
                        _base_row(
                            split_id=f,
                            seed=int(seed),
                            model="gam_ipp",
                            objective="ipp",
                            n_presence=split.n_train_presence,
                            n_background=split.n_train_background,
                            basis_df=basis_df,
                            knots=f"G={n_intervals},K={degree}",
                            lambda_s=lambda_s,
                            lambda_r=lambda_r,
                            converged=res_gam.converged,
                            runtime_s=round(res_gam.runtime_s, 4),
                            auc_roc=m_g["auc_roc"],
                            auprc=m_g["auprc"],
                            artifact_path=str(art_g.relative_to(outdir)),
                        )
                    )
                    (fold_dir / f"kan_gam_spearman_seed{seed}.txt").write_text(
                        f"{rho}\n", encoding="utf-8"
                    )

                if export_curves and seed == seeds[0]:
                    import torch

                    Xtr_t = torch.as_tensor(X_tr, dtype=torch.float64)
                    export_component_curves(
                        model_ipp,
                        cols,
                        outdir / "curves" / f"fold_{f}_kan_ipp",
                        X_ref=Xtr_t,
                    )

            if "kan_bce" in models:
                model_bce, res_bce = fit_bce_kan(
                    X_tr,
                    y_tr,
                    X_te,
                    n_intervals=n_intervals,
                    degree=degree,
                    lambda_s=lambda_s,
                    lambda_r=lambda_r,
                    seed=int(seed),
                    steps=lbfgs_steps,
                )
                m_b = ranking_metrics(y_te, res_bce.scores_test)
                art_b = fold_dir / f"kan_bce_seed{seed}_pred.csv"
                pd.DataFrame({"y": y_te, "score": res_bce.scores_test}).to_csv(art_b, index=False)
                rows.append(
                    _base_row(
                        split_id=f,
                        seed=int(seed),
                        model="additive_kan_bce",
                        objective="bce",
                        n_presence=split.n_train_presence,
                        n_background=split.n_train_background,
                        basis_df=basis_df,
                        knots=f"G={n_intervals},K={degree}",
                        lambda_s=lambda_s,
                        lambda_r=lambda_r,
                        converged=res_bce.converged,
                        runtime_s=round(res_bce.runtime_s, 4),
                        auc_roc=m_b["auc_roc"],
                        auprc=m_b["auprc"],
                        artifact_path=str(art_b.relative_to(outdir)),
                    )
                )

    metrics = pd.DataFrame(rows)
    # stable column order
    for c in ROW_FIELDS:
        if c not in metrics.columns:
            metrics[c] = ""
    metrics = metrics[ROW_FIELDS]
    metrics_path = outdir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    manifest = {
        "run_id": outdir.name,
        "data_path": str(data_path) if data_path else "data/ginkgo/ginkgo_training_with_coords.csv",
        "estimand": "global_conditional_density",
        "integration_support": "outer_train_rows",
        "scoring": "oof_presence_vs_bg",
        "n_intervals": n_intervals,
        "degree": degree,
        "lambda_s": lambda_s,
        "lambda_r": lambda_r,
        "lambda_selection": "frozen_default",
        "weight_decay": 0.0,
        "seeds": list(seeds),
        "folds": list(folds),
        "models": list(models),
        "lbfgs_steps": lbfgs_steps,
        "env_columns": cols,
        "input_scaling": {
            "kan_gam_bce": "per_fold_standard_scaler",
            "maxnet": "raw_env_unscaled",
        },
        "n_rows": int(len(df)),
        "created_unix": time.time(),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_summary(metrics, outdir)
    return metrics


def _write_summary(metrics: pd.DataFrame, outdir: Path) -> None:
    lines = ["# Ginkgo Phase 1 CV SUMMARY", ""]
    lines.append("## OOF AUC / AUPRC by model (mean ± std over fold×seed)")
    lines.append("")
    for model, g in metrics.groupby("model"):
        auc = g["auc_roc"].astype(float)
        ap = g["auprc"].astype(float)
        lines.append(
            f"- **{model}**: AUC {auc.mean():.4f} ± {auc.std(ddof=1):.4f} "
            f"(n={len(auc)}); AUPRC {ap.mean():.4f} ± {ap.std(ddof=1):.4f}"
        )
    lines.append("")
    lines.append("## Judgment (v6.1 / Phase1 v2)")
    lines.append("")

    def mean_auc(name: str) -> float:
        sub = metrics.loc[metrics["model"] == name, "auc_roc"].astype(float)
        return float(sub.mean()) if len(sub) else float("nan")

    kan_ipp = mean_auc("additive_kan_ipp")
    kan_bce = mean_auc("additive_kan_bce")
    gam = mean_auc("gam_ipp")
    mx = mean_auc("maxnet")

    if np.isfinite(kan_ipp) and np.isfinite(gam):
        diff = abs(kan_ipp - gam)
        lines.append(
            f"- KAN-IPP vs GAM-IPP mean AUC: {kan_ipp:.4f} vs {gam:.4f} (|Δ|={diff:.4f}). "
            "High agreement is **expected** (same basis)."
        )
    if np.isfinite(kan_ipp) and np.isfinite(kan_bce):
        if kan_ipp + 1e-6 < kan_bce:
            lines.append(
                f"- IPP-KAN ({kan_ipp:.4f}) did **not** beat BCE-KAN ({kan_bce:.4f}). "
                "No superiority claim. Diagnostics: background scheme, λs capacity, prevalence sensitivity of BCE."
            )
        else:
            lines.append(
                f"- IPP-KAN mean AUC {kan_ipp:.4f} vs BCE-KAN {kan_bce:.4f} (descriptive only)."
            )
    if np.isfinite(mx):
        if 0.85 <= mx <= 0.95:
            lines.append(f"- maxnet mean AUC {mx:.4f} is within expected PO→BG band 0.85–0.95.")
        else:
            lines.append(
                f"- maxnet mean AUC {mx:.4f} is **outside** 0.85–0.95 — check data/bridge."
            )
    lines.append("")
    lines.append("Protocol: `integration_support=outer_train_rows`; λ selection: frozen_default.")
    lines.append("Metrics are presence-vs-background discrimination, not true absence performance.")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
