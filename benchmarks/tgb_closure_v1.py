#!/usr/bin/env python3
"""TGB sensitivity under Methodological Closure v1 protocol.

Full dual-arm re-analysis (random 50k background vs target-group background)
for the pre-registered NSW 12-species set, using species-level λ* from
``outputs/methodological_closure_full_v1/lambda_star.csv``.

Models (both arms):
  - additive_kan_ipp
  - gam_ipp_same_basis
  - additive_kan_bce
  - maxnet_bg10k
  - maxnet_bg50k

Outputs under ``outputs/tgb_closure_v1/``:
  metrics.csv, species_paired_delta.csv, SUMMARY.md, manifest.json

Usage:
  python benchmarks/tgb_closure_v1.py
  python benchmarks/tgb_closure_v1.py --outdir outputs/tgb_closure_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.data.categorical import prepare_matrices
from kanmaxent.data.nceas_io import (
    build_po_bg_frame,
    build_tgb_frame,
    load_region,
    pa_labels_and_env_for_species,
    split_species,
)
from kanmaxent.evaluation.metrics import ranking_metrics
from kanmaxent.nceas_fit import HybridSplineCat, fit_hybrid_bce, fit_hybrid_gam_ipp, fit_hybrid_ipp
from kanmaxent.reference.maxnet_nceas import fit_predict_maxnet_nceas

PROTOCOL_ID = "tgb_closure_v1"
CLOSURE_LAMBDA = ROOT / "outputs" / "methodological_closure_full_v1" / "lambda_star.csv"

# Pre-registered NSW TGB species (same set as Phase-3 / Table S10 historical)
TGB_SPECIES = [
    "nsw04",
    "nsw06",
    "nsw09",
    "nsw14",
    "nsw16",
    "nsw17",
    "nsw18",
    "nsw24",
    "nsw28",
    "nsw39",
    "nsw43",
    "nsw52",
]

M_ADD = "additive_kan_ipp"
M_GAM = "gam_ipp_same_basis"
M_BCE = "additive_kan_bce"
M_MX10 = "maxnet_bg10k"
M_MX50 = "maxnet_bg50k"
ALL_MODELS = (M_ADD, M_GAM, M_BCE, M_MX10, M_MX50)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _metrics_block(y_pa: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    m = ranking_metrics(y_pa, scores)
    return {
        "auc_roc": float(m.get("auc_roc", np.nan)),
        "auprc": float(m.get("auprc", np.nan)),
        "prg": float(m.get("prg", np.nan)),
        "cor": float(m.get("cor", np.nan)),
    }


def _boot_mean_ci(x: np.ndarray, B: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


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
    info = dict(info or {})
    info["maxnet_bg_n"] = int((tr_mx["occ"] == 0).sum())
    return scores, info


def run_species_arm(
    region_data,
    species_id: str,
    *,
    background_scheme: str,
    lambda_s: float,
    lambda_path: str,
    seed: int = 0,
    lambda_r: float = 1e-6,
    additive_steps: int = 15,
) -> List[Dict[str, Any]]:
    """Fit all models on one species × background arm."""
    reg = region_data.region
    po_sp = split_species(region_data.po, species_id)
    n_po = len(po_sp)
    y_pa, pa_env, taxon_group = pa_labels_and_env_for_species(region_data, species_id)
    n_pa = int(len(y_pa))
    continuous = list(region_data.continuous)
    categorical = list(region_data.categorical)
    covars = continuous + categorical

    base: Dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "region": reg,
        "species_id": species_id,
        "taxon_group": taxon_group,
        "seed": int(seed),
        "n_presence": int(n_po),
        "n_PA_test": n_pa,
        "lambda_s": float(lambda_s),
        "lambda_selection": lambda_path,
        "lambda_source": "methodological_closure_full_v1/lambda_star.csv",
        "background_scheme": background_scheme,
        "objective": "ipp",
        "remediation": "primary",
    }

    tgb_meta: Dict[str, Any] = {}
    if background_scheme == "tgb":
        train_df, tgb_meta = build_tgb_frame(
            region_data.po, species_id, covars, max_bg=50000, seed=seed
        )
        base["integration_support"] = "tgb_other_po_same_group"
        n_tgb = int(tgb_meta.get("tgb_n_used", 0))
        base["n_background"] = n_tgb
        for k, v in tgb_meta.items():
            base[f"tgb_{k}"] = v
        if n_tgb < 20:
            rows = []
            for m in ALL_MODELS:
                rows.append(
                    {
                        **base,
                        "model": m,
                        "converged": False,
                        "runtime_s": 0.0,
                        "auc_roc": float("nan"),
                        "auprc": float("nan"),
                        "prg": float("nan"),
                        "cor": float("nan"),
                        "skip_reason": f"tgb_n={n_tgb}_lt_20",
                    }
                )
            return rows
    else:
        train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
        base["integration_support"] = "full_background_50k"
        base["n_background"] = int((train_df["occ"] == 0).sum())

    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    n_bg = int((y_tr == 0).sum())
    base["n_background"] = n_bg
    pos_weight = float(max(n_bg, 1) / max(n_po, 1))

    mats = prepare_matrices(train_df, pa_env, continuous, categorical)
    Xc_tr, Xk_tr = mats["X_cont_train"], mats["X_cat_train"]
    Xc_te, Xk_te = mats["X_cont_test"], mats["X_cat_test"]

    rows: List[Dict[str, Any]] = []

    # Additive IPP + same-basis GAM
    t0 = time.perf_counter()
    try:
        add_model, scores_add, meta_add = fit_hybrid_ipp(
            Xc_tr,
            Xk_tr,
            y_tr,
            Xc_te,
            Xk_te,
            lambda_s=float(lambda_s),
            lambda_r=lambda_r,
            seed=seed,
            steps=additive_steps,
        )
        mb = _metrics_block(y_pa, scores_add)
        rows.append(
            {
                **base,
                "model": M_ADD,
                "objective": "ipp",
                "converged": bool(meta_add.get("converged", True)),
                "runtime_s": float(meta_add.get("runtime_s", time.perf_counter() - t0)),
                "skip_reason": "",
                **mb,
            }
        )
        t1 = time.perf_counter()
        gam_model = HybridSplineCat(
            Xc_tr.shape[1],
            Xk_tr.shape[1] if getattr(Xk_tr, "ndim", 0) == 2 else 0,
            n_intervals=6,
            degree=3,
            lambda_s=float(lambda_s),
            lambda_r=lambda_r,
        )
        gam_model.fit_bounds(Xc_tr)
        scores_gam, meta_gam = fit_hybrid_gam_ipp(
            gam_model,
            Xc_tr,
            Xk_tr,
            y_tr,
            Xc_te,
            Xk_te,
            B_train=meta_add.get("B_train"),
        )
        mg = _metrics_block(y_pa, scores_gam)
        rows.append(
            {
                **base,
                "model": M_GAM,
                "objective": "ipp",
                "converged": bool(meta_gam.get("converged", True)),
                "runtime_s": float(meta_gam.get("runtime_s", time.perf_counter() - t1)),
                "skip_reason": "",
                **mg,
            }
        )
    except Exception as e:  # noqa: BLE001
        err = str(e)[:300]
        for m in (M_ADD, M_GAM):
            rows.append(
                {
                    **base,
                    "model": m,
                    "objective": "ipp",
                    "converged": False,
                    "runtime_s": 0.0,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "cor": float("nan"),
                    "skip_reason": err,
                }
            )

    # BCE
    t2 = time.perf_counter()
    try:
        _, scores_b, info_b = fit_hybrid_bce(
            Xc_tr,
            Xk_tr,
            y_tr,
            Xc_te,
            Xk_te,
            pos_weight=pos_weight,
            lambda_s=float(lambda_s),
            lambda_r=lambda_r,
            seed=seed,
            steps=additive_steps,
        )
        mbb = _metrics_block(y_pa, scores_b)
        rows.append(
            {
                **base,
                "model": M_BCE,
                "objective": "bce_pos_weight",
                "pos_weight": pos_weight,
                "converged": bool(info_b.get("converged", True)),
                "runtime_s": float(info_b.get("runtime_s", time.perf_counter() - t2)),
                "skip_reason": "",
                **mbb,
            }
        )
    except Exception as e:  # noqa: BLE001
        rows.append(
            {
                **base,
                "model": M_BCE,
                "objective": "bce_pos_weight",
                "pos_weight": pos_weight,
                "converged": False,
                "runtime_s": time.perf_counter() - t2,
                "auc_roc": float("nan"),
                "auprc": float("nan"),
                "prg": float("nan"),
                "cor": float("nan"),
                "skip_reason": str(e)[:300],
            }
        )

    # maxnet 10k / 50k
    for mid, mbg in ((M_MX10, 10000), (M_MX50, 50000)):
        t3 = time.perf_counter()
        try:
            scores_mx, info = fit_maxnet_arm(
                train_df, pa_env, covars, categorical, n_po, mbg, seed
            )
            ok = bool(info.get("ok", False))
            mm = (
                _metrics_block(y_pa, scores_mx)
                if ok and scores_mx is not None
                else {
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "cor": float("nan"),
                }
            )
            rows.append(
                {
                    **base,
                    "model": mid,
                    "objective": "maxnet_default",
                    "maxnet_max_bg": mbg,
                    "maxnet_bg_n": info.get("maxnet_bg_n", ""),
                    "converged": ok,
                    "runtime_s": time.perf_counter() - t3,
                    "skip_reason": "" if ok else str(info.get("error", info))[:300],
                    **mm,
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    **base,
                    "model": mid,
                    "objective": "maxnet_default",
                    "maxnet_max_bg": mbg,
                    "converged": False,
                    "runtime_s": time.perf_counter() - t3,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "cor": float("nan"),
                    "skip_reason": str(e)[:300],
                }
            )

    return rows


def species_paired_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Wide: one row per species×model with random, TGB, delta."""
    recs = []
    for (sp, model), g in metrics.groupby(["species_id", "model"]):
        gr = g[g.background_scheme == "random_50k"]
        gt = g[g.background_scheme == "tgb"]
        if gr.empty or gt.empty:
            continue
        r = gr.iloc[0]
        t = gt.iloc[0]
        auc_r = float(r["auc_roc"])
        auc_t = float(t["auc_roc"])
        recs.append(
            {
                "region": r["region"],
                "species_id": sp,
                "taxon_group": r.get("taxon_group", ""),
                "n_presence": r["n_presence"],
                "lambda_s": r["lambda_s"],
                "lambda_selection": r["lambda_selection"],
                "model": model,
                "auc_random": auc_r,
                "auc_tgb": auc_t,
                "delta_tgb_minus_random": auc_t - auc_r
                if np.isfinite(auc_r) and np.isfinite(auc_t)
                else float("nan"),
                "n_bg_random": r.get("n_background", ""),
                "n_bg_tgb": t.get("n_background", ""),
                "converged_random": bool(r.get("converged", False)),
                "converged_tgb": bool(t.get("converged", False)),
                "skip_random": r.get("skip_reason", ""),
                "skip_tgb": t.get("skip_reason", ""),
            }
        )
    return pd.DataFrame(recs)


def model_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in paired.groupby("model"):
        # intention-to-treat: all species with finite both arms
        both = g[np.isfinite(g["delta_tgb_minus_random"])]
        # maxnet-style: also report completed-only (both converged)
        both_ok = both[both["converged_random"] & both["converged_tgb"]]
        for label, sub in (("finite_pair", both), ("both_converged", both_ok)):
            d = sub["delta_tgb_minus_random"].to_numpy(float)
            mu, lo, hi = _boot_mean_ci(d)
            rows.append(
                {
                    "model": model,
                    "pair_rule": label,
                    "n_paired": int(np.isfinite(d).sum()),
                    "mean_auc_random": float(sub["auc_random"].mean()) if len(sub) else float("nan"),
                    "mean_auc_tgb": float(sub["auc_tgb"].mean()) if len(sub) else float("nan"),
                    "mean_delta_tgb_minus_random": mu,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "median_delta": float(np.nanmedian(d)) if len(d) else float("nan"),
                    "sd_delta": float(np.nanstd(d, ddof=1)) if len(d) > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def write_summary(
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    lambda_map: pd.DataFrame,
) -> None:
    lines = [
        "# TGB Closure v1 — SUMMARY",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`",
        f"**Region / species:** NSW, n = {len(TGB_SPECIES)} pre-registered species",
        f"**λ\\* source:** `{CLOSURE_LAMBDA.relative_to(ROOT)}` (species-level, frozen for both arms)",
        "**Arms:** `random_50k` (full background_50k) vs `tgb` (other PO same taxon_group, max 50k)",
        f"**Models:** {', '.join(ALL_MODELS)}",
        f"**Generated (UTC):** {_now()}",
        "",
        "## Species list and λ\\*",
        "",
        "| species | n_PO | λ\\* | path |",
        "|:--------|-----:|----:|:-----|",
    ]
    for _, r in lambda_map.sort_values("species_id").iterrows():
        lines.append(
            f"| {r['species_id']} | {int(r['n_presence'])} | {float(r['lambda_s']):g} | {r['lambda_path']} |"
        )

    lines += ["", "## Model-level paired ΔAUC (TGB − random)", ""]
    for rule in ("finite_pair", "both_converged"):
        sub = summary[summary.pair_rule == rule]
        lines.append(f"### Pair rule: `{rule}`")
        lines.append("")
        lines.append(
            "| model | n | mean AUC random | mean AUC TGB | mean Δ | 95% CI |"
        )
        lines.append("|:------|--:|----------------:|-------------:|-------:|:-------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['model']} | {int(r['n_paired'])} | "
                f"{r['mean_auc_random']:.4f} | {r['mean_auc_tgb']:.4f} | "
                f"{r['mean_delta_tgb_minus_random']:+.4f} | "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |"
            )
        lines.append("")

    # maxnet failures
    lines.append("## maxnet convergence")
    lines.append("")
    for arm in ("random_50k", "tgb"):
        mx = metrics[(metrics.model == M_MX10) & (metrics.background_scheme == arm)]
        n_ok = int(mx["converged"].astype(bool).sum())
        lines.append(f"- {arm} maxnet@10k: {n_ok}/{len(mx)} converged")
        fails = mx.loc[~mx["converged"].astype(bool), "species_id"].tolist()
        if fails:
            lines.append(f"  - failed: {', '.join(fails)}")
    lines.append("")

    # TGB background sizes
    tgb_rows = metrics[
        (metrics.background_scheme == "tgb") & (metrics.model == M_ADD)
    ][["species_id", "n_background"]].drop_duplicates()
    lines.append("## TGB background size (per species)")
    lines.append("")
    for _, r in tgb_rows.sort_values("species_id").iterrows():
        lines.append(f"- {r['species_id']}: n_bg = {int(r['n_background'])}")
    lines.append("")

    lines.append("## Files")
    lines.append("")
    lines.append("- `metrics.csv` — long format, all models × both arms")
    lines.append("- `species_paired_delta.csv` — species×model random/TGB/Δ")
    lines.append("- `model_summary.csv` — model-level means and bootstrap CIs")
    lines.append("- `manifest.json` — protocol fingerprint")
    lines.append("")

    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", outdir / "SUMMARY.md")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "outputs" / "tgb_closure_v1",
    )
    p.add_argument(
        "--lambda-star",
        type=Path,
        default=CLOSURE_LAMBDA,
        help="Species-level λ* table from methodological closure",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--species",
        default=",".join(TGB_SPECIES),
        help="Comma-separated species ids (default: full TGB set)",
    )
    args = p.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.lambda_star.is_file():
        raise FileNotFoundError(
            f"λ* table not found: {args.lambda_star}. "
            "Run Methodological Closure Stage A first."
        )

    lam_df = pd.read_csv(args.lambda_star)
    species = [s.strip() for s in args.species.split(",") if s.strip()]
    lam_map = lam_df[
        (lam_df["region"] == "NSW") & (lam_df["species_id"].isin(species))
    ].copy()
    missing = set(species) - set(lam_map["species_id"])
    if missing:
        raise RuntimeError(f"λ* missing for species: {sorted(missing)}")

    print(f"Loading NSW… ({len(species)} species)")
    region_data = load_region("NSW")
    all_rows: List[Dict[str, Any]] = []

    for i, sp in enumerate(species, 1):
        row = lam_map.loc[lam_map.species_id == sp].iloc[0]
        lam = float(row["lambda_s"])
        lam_path = str(row["lambda_path"])
        print(f"[{i}/{len(species)}] {sp}  λ*={lam:g} ({lam_path})")
        for arm in ("random_50k", "tgb"):
            t0 = time.perf_counter()
            rows = run_species_arm(
                region_data,
                sp,
                background_scheme=arm,
                lambda_s=lam,
                lambda_path=lam_path,
                seed=args.seed,
            )
            all_rows.extend(rows)
            print(f"  arm={arm}: {len(rows)} model rows in {time.perf_counter()-t0:.1f}s")

    metrics = pd.DataFrame(all_rows)
    metrics_path = outdir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("wrote", metrics_path)

    paired = species_paired_table(metrics)
    paired.to_csv(outdir / "species_paired_delta.csv", index=False)
    summary = model_summary(paired)
    summary.to_csv(outdir / "model_summary.csv", index=False)

    write_summary(metrics, paired, summary, outdir, lam_map)

    # Table S10 style (finite_pair rule, primary models)
    s10 = summary[summary.pair_rule == "finite_pair"].copy()
    s10_out = s10[
        [
            "model",
            "n_paired",
            "mean_auc_random",
            "mean_auc_tgb",
            "mean_delta_tgb_minus_random",
            "ci_lo",
            "ci_hi",
        ]
    ].rename(
        columns={
            "mean_delta_tgb_minus_random": "mean_delta_tgb_minus_random",
            "ci_lo": "ci_lo",
            "ci_hi": "ci_hi",
        }
    )
    s10_out.to_csv(outdir / "TableS10_TGB_sensitivity_closure.csv", index=False)

    manifest = {
        "protocol": PROTOCOL_ID,
        "generated_utc": _now(),
        "region": "NSW",
        "species": species,
        "lambda_star_path": str(args.lambda_star),
        "models": list(ALL_MODELS),
        "background_arms": ["random_50k", "tgb"],
        "seed": args.seed,
        "n_metric_rows": int(len(metrics)),
        "outdir": str(outdir),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("done.")


if __name__ == "__main__":
    main()
