#!/usr/bin/env python
"""Phase 3 NCEAS six-region benchmark runner (PO + 50k BG → independent PA)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.data.categorical import prepare_matrices
from kanmaxent.data.nceas_io import (
    MULTI_GROUP_REGIONS,
    build_po_bg_frame,
    build_tgb_frame,
    load_region,
    pa_labels_and_env_for_species,
    split_species,
    write_region_manifest,
)
from kanmaxent.evaluation.lambda_selector import select_lambda_s_po_cv
from kanmaxent.evaluation.metrics import ranking_metrics
from kanmaxent.nceas_fit import fit_hybrid_bce, fit_hybrid_gam_ipp, fit_hybrid_ipp
from kanmaxent.reference.maxnet_nceas import fit_predict_maxnet_nceas

ALL_REGIONS = ("AWT", "CAN", "NSW", "NZ", "SA", "SWI")


def _status_path(outdir: Path, region: str, spid: str, seed: int) -> Path:
    return outdir / "status" / f"{region}_{spid}_s{seed}.json"


def _load_status(path: Path) -> Optional[dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_species(
    region_data,
    species_id: str,
    *,
    seed: int = 0,
    lambda_s: Optional[float] = None,
    lambda_r: float = 1e-6,
    n_intervals: int = 6,
    degree: int = 3,
    lbfgs_steps: int = 15,
    models: Sequence[str] = ("kan_ipp", "gam_ipp", "kan_bce", "maxnet"),
    tune_lambda: bool = False,
    background_scheme: str = "random_50k",
    maxnet_max_bg: int = 10000,
) -> List[Dict]:
    po_sp = split_species(region_data.po, species_id)
    n_po = len(po_sp)
    y_pa, pa_env, taxon_group = pa_labels_and_env_for_species(region_data, species_id)
    n_pa = int(len(y_pa))
    covars = region_data.continuous + region_data.categorical

    base = {
        "region": region_data.region,
        "taxon_group": taxon_group,
        "species_id": species_id,
        "split_id": "pa_holdout",
        "seed": int(seed),
        "estimand": "global_conditional_density",
        "n_presence": n_po,
        "n_PA_test": n_pa,
        "categorical_encoding": "onehot_l2",
        "weight_decay": 0.0,
        "lambda_r": lambda_r,
        "basis_df": n_intervals + degree,
        "knots": f"G={n_intervals},K={degree}",
        "background_scheme": background_scheme,
    }

    if n_po < 5:
        rows = []
        for m in models:
            rows.append(
                {
                    **base,
                    "model": {
                        "kan_ipp": "additive_kan_ipp",
                        "gam_ipp": "gam_ipp",
                        "kan_bce": "additive_kan_bce",
                        "maxnet": "maxnet",
                    }.get(m, m),
                    "objective": m,
                    "integration_support": "full_background_50k",
                    "n_background": 0,
                    "lambda_s": "",
                    "lambda_selection": "skipped_n_po_lt_5",
                    "converged": False,
                    "runtime_s": 0.0,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "cor": float("nan"),
                    "artifact_path": "",
                    "skip_reason": "n_po<5",
                }
            )
        return rows

    tgb_meta = {}
    if background_scheme == "tgb":
        train_df, tgb_meta = build_tgb_frame(
            region_data.po, species_id, covars, max_bg=50000, seed=seed
        )
        if int(tgb_meta.get("tgb_n_used", 0)) < 20:
            # insufficient TGB → mark skip
            rows = []
            for m in models:
                rows.append(
                    {
                        **base,
                        "model": {
                            "kan_ipp": "additive_kan_ipp",
                            "gam_ipp": "gam_ipp",
                            "kan_bce": "additive_kan_bce",
                            "maxnet": "maxnet",
                        }.get(m, m),
                        "objective": m,
                        "integration_support": "tgb_other_po_same_group",
                        "n_background": int(tgb_meta.get("tgb_n_used", 0)),
                        "lambda_s": "",
                        "lambda_selection": "skipped_tgb_too_small",
                        "converged": False,
                        "runtime_s": 0.0,
                        "auc_roc": float("nan"),
                        "auprc": float("nan"),
                        "prg": float("nan"),
                        "cor": float("nan"),
                        "artifact_path": "",
                        "skip_reason": f"tgb_n={tgb_meta.get('tgb_n_used')}",
                    }
                )
            return rows
        integration_support = "tgb_other_po_same_group"
    else:
        train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
        integration_support = "full_background_50k"

    n_bg = int((train_df["occ"] == 0).sum())
    base["n_background"] = n_bg
    base["integration_support"] = integration_support

    # λs
    lam_path = "frozen_default"
    if lambda_s is None:
        lambda_s = 1e-2
        if tune_lambda and n_po >= 30:

            def fit_eval(po_tr_idx, po_val_idx, lam):
                po_block = train_df.iloc[:n_po]
                bg_block = train_df.iloc[n_po:]
                bg_s = bg_block.sample(
                    n=min(8000, len(bg_block)), random_state=seed
                )
                tr = pd.concat([po_block.iloc[po_tr_idx], bg_s], ignore_index=True)
                val = pd.concat(
                    [
                        po_block.iloc[po_val_idx],
                        bg_s.sample(n=min(2000, len(bg_s)), random_state=seed + 1),
                    ],
                    ignore_index=True,
                )
                prep = prepare_matrices(
                    tr, val, region_data.continuous, region_data.categorical
                )
                _, scores, _ = fit_hybrid_ipp(
                    prep["X_cont_train"],
                    prep["X_cat_train"],
                    tr["occ"].to_numpy(dtype=np.float64),
                    prep["X_cont_test"],
                    prep["X_cat_test"],
                    lambda_s=lam,
                    lambda_r=lambda_r,
                    seed=seed,
                    steps=8,
                )
                return ranking_metrics(val["occ"].to_numpy(), scores)["auc_roc"]

            sel = select_lambda_s_po_cv(
                n_po, n_bg, fit_eval, k_folds=4, min_po_for_cv=30, seed=seed
            )
            lambda_s = float(sel["lambda_s"])
            lam_path = str(sel["path"])
        elif n_po < 30:
            lam_path = "frozen_n_po_lt_30"

    prep = prepare_matrices(
        train_df, pa_env, region_data.continuous, region_data.categorical
    )
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    pos_weight = float(max(n_bg, 1) / max(n_po, 1))
    rows: List[Dict] = []

    # maxnet
    if "maxnet" in models:
        t0 = time.perf_counter()
        tr_mx = train_df[covars + ["occ"]].copy()
        is_bg = tr_mx["occ"].to_numpy() == 0
        if int(is_bg.sum()) > maxnet_max_bg:
            po_mx = tr_mx.loc[~is_bg]
            bg_mx = tr_mx.loc[is_bg].sample(n=maxnet_max_bg, random_state=seed)
            tr_mx = pd.concat([po_mx, bg_mx], ignore_index=True)
        te_mx = pa_env[covars].copy()
        scores, info = fit_predict_maxnet_nceas(
            tr_mx,
            te_mx,
            covars,
            region_data.categorical,
            n_presence=n_po,
            regmult=1.0,
        )
        m = (
            ranking_metrics(y_pa, scores)
            if info.get("ok")
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
                "model": "maxnet",
                "objective": "maxnet_default",
                "lambda_s": "",
                "lambda_selection": f"maxnet_classes_{info.get('classes', '')}",
                "converged": bool(info.get("ok")),
                "runtime_s": round(time.perf_counter() - t0, 4),
                "auc_roc": m["auc_roc"],
                "auprc": m["auprc"],
                "prg": m["prg"],
                "cor": m.get("cor", float("nan")),
                "artifact_path": "",
                "skip_reason": "" if info.get("ok") else str(info.get("error", ""))[:200],
                "maxnet_bg_n": int((tr_mx["occ"] == 0).sum()),
            }
        )

    info_ipp = {}
    model_ipp = None
    if "kan_ipp" in models or "gam_ipp" in models:
        model_ipp, scores_ipp, info_ipp = fit_hybrid_ipp(
            prep["X_cont_train"],
            prep["X_cat_train"],
            y_tr,
            prep["X_cont_test"],
            prep["X_cat_test"],
            n_intervals=n_intervals,
            degree=degree,
            lambda_s=float(lambda_s),
            lambda_r=lambda_r,
            seed=seed,
            steps=lbfgs_steps,
        )
        if "kan_ipp" in models:
            m = ranking_metrics(y_pa, scores_ipp)
            rows.append(
                {
                    **base,
                    "model": "additive_kan_ipp",
                    "objective": "ipp",
                    "lambda_s": float(lambda_s),
                    "lambda_selection": lam_path,
                    "converged": bool(info_ipp.get("converged")),
                    "runtime_s": round(float(info_ipp.get("runtime_s", 0)), 4),
                    "auc_roc": m["auc_roc"],
                    "auprc": m["auprc"],
                    "prg": m["prg"],
                    "cor": m.get("cor", float("nan")),
                    "artifact_path": "",
                    "skip_reason": "",
                }
            )
        if "gam_ipp" in models and model_ipp is not None:
            scores_g, info_g = fit_hybrid_gam_ipp(
                model_ipp,
                prep["X_cont_train"],
                prep["X_cat_train"],
                y_tr,
                prep["X_cont_test"],
                prep["X_cat_test"],
                B_train=info_ipp.get("B_train"),
            )
            m = ranking_metrics(y_pa, scores_g)
            rows.append(
                {
                    **base,
                    "model": "gam_ipp",
                    "objective": "ipp",
                    "lambda_s": float(lambda_s),
                    "lambda_selection": lam_path,
                    "converged": bool(info_g.get("converged")),
                    "runtime_s": round(float(info_g.get("runtime_s", 0)), 4),
                    "auc_roc": m["auc_roc"],
                    "auprc": m["auprc"],
                    "prg": m["prg"],
                    "cor": m.get("cor", float("nan")),
                    "artifact_path": "",
                    "skip_reason": "",
                }
            )

    if "kan_bce" in models:
        _, scores_b, info_b = fit_hybrid_bce(
            prep["X_cont_train"],
            prep["X_cat_train"],
            y_tr,
            prep["X_cont_test"],
            prep["X_cat_test"],
            pos_weight=pos_weight,
            n_intervals=n_intervals,
            degree=degree,
            lambda_s=float(lambda_s),
            lambda_r=lambda_r,
            seed=seed,
            steps=lbfgs_steps,
        )
        m = ranking_metrics(y_pa, scores_b)
        rows.append(
            {
                **base,
                "model": "additive_kan_bce",
                "objective": "bce_pos_weight",
                "lambda_s": float(lambda_s),
                "lambda_selection": lam_path,
                "converged": bool(info_b.get("converged")),
                "runtime_s": round(float(info_b.get("runtime_s", 0)), 4),
                "auc_roc": m["auc_roc"],
                "auprc": m["auprc"],
                "prg": m["prg"],
                "cor": m.get("cor", float("nan")),
                "artifact_path": "",
                "skip_reason": "",
                "pos_weight": pos_weight,
            }
        )

    for r in rows:
        r.update({f"tgb_{k}": v for k, v in tgb_meta.items()})
    return rows


def write_summary(metrics: pd.DataFrame, outdir: Path) -> None:
    lines = ["# NCEAS Phase 3 Full Benchmark SUMMARY", ""]
    lines.append("## By model (mean AUC ± std over species×seed×region)")
    for model, g in metrics.groupby("model"):
        auc = pd.to_numeric(g["auc_roc"], errors="coerce").dropna()
        std = float(auc.std(ddof=1)) if len(auc) > 1 else 0.0
        lines.append(f"- **{model}**: {auc.mean():.4f} ± {std:.4f} (n={len(auc)})")

    lines.append("")
    lines.append("## By region × model (mean AUC)")
    lines.append("")
    try:
        pivot = metrics.pivot_table(
            index="region", columns="model", values="auc_roc", aggfunc="mean"
        )
        lines.append(pivot.round(4).to_markdown())
    except Exception:
        lines.append(str(metrics.groupby(["region", "model"])["auc_roc"].mean()))
    lines.append("")
    # paired deltas
    lines.append("## Paired ΔAUC (species-level means across seeds)")
    wide = metrics.pivot_table(
        index=["region", "species_id"],
        columns="model",
        values="auc_roc",
        aggfunc="mean",
    )
    if "additive_kan_ipp" in wide.columns and "gam_ipp" in wide.columns:
        d = (wide["additive_kan_ipp"] - wide["gam_ipp"]).dropna()
        lines.append(
            f"- KAN−GAM: mean Δ={d.mean():.5f}, |Δ|mean={d.abs().mean():.5f}, r={wide['additive_kan_ipp'].corr(wide['gam_ipp']):.4f}"
        )
    if "additive_kan_ipp" in wide.columns and "maxnet" in wide.columns:
        d = (wide["additive_kan_ipp"] - wide["maxnet"]).dropna()
        lines.append(
            f"- KAN−maxnet: mean Δ={d.mean():.4f}, r={wide['additive_kan_ipp'].corr(wide['maxnet']):.4f}"
        )
    if "additive_kan_ipp" in wide.columns and "additive_kan_bce" in wide.columns:
        d = (wide["additive_kan_ipp"] - wide["additive_kan_bce"]).dropna()
        lines.append(f"- IPP−BCE: mean Δ={d.mean():.4f}")
    lines.append("")
    lines.append("Protocol: independent PA; main background=random_50k; no PA in tuning.")
    (outdir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="NCEAS six-region Phase 3 runner")
    p.add_argument("--regions", default="AWT,CAN,NSW,NZ,SA,SWI")
    p.add_argument("--outdir", default="")
    p.add_argument("--seeds", default="0")
    p.add_argument("--species", default="", help="comma list; default all in region")
    p.add_argument("--models", default="kan_ipp,gam_ipp,kan_bce,maxnet")
    p.add_argument("--lbfgs-steps", type=int, default=15)
    p.add_argument("--no-tune-lambda", action="store_true")
    p.add_argument("--lambda-s", type=float, default=-1.0)
    p.add_argument("--background", default="random_50k", choices=["random_50k", "tgb"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--write-manifests", action="store_true", help="refresh region MANIFEST.json")
    args = p.parse_args()

    regions = tuple(x.strip().upper() for x in args.regions.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag = "tgb" if args.background == "tgb" else "full"
        outdir = ROOT / "outputs" / f"nceas_{tag}_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "status").mkdir(exist_ok=True)
    (outdir / "by_region").mkdir(exist_ok=True)

    print(f"Regions={regions} seeds={seeds} models={models} bg={args.background}")
    print(f"Out={outdir} resume={args.resume}")

    all_rows: List[Dict] = []
    metrics_path = outdir / (
        "metrics_tgb.csv" if args.background == "tgb" else "metrics_all.csv"
    )
    # load partial if resume and file exists
    if args.resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_rows = prev.to_dict(orient="records")
        print(f"Resumed {len(all_rows)} existing rows from {metrics_path}")

    done_keys = {
        (str(r.get("region")), str(r.get("species_id")), int(r.get("seed", 0)), str(r.get("model")))
        for r in all_rows
    }

    for reg in regions:
        if args.write_manifests:
            write_region_manifest(reg, ROOT / "data" / "nceas" / reg.lower())
        data = load_region(reg)
        species = (
            [s.strip() for s in args.species.split(",") if s.strip()]
            if args.species.strip()
            else list(data.species_ids)
        )
        print(f"== {reg}: {len(species)} species ==", flush=True)
        for sp in species:
            for seed in seeds:
                st = _status_path(outdir, reg, sp, seed)
                if args.resume and _load_status(st) and _load_status(st).get("ok"):
                    # still may need rows; if all models present skip
                    if all(
                        (reg, sp, seed, mname) in done_keys
                        for mname in (
                            "additive_kan_ipp",
                            "gam_ipp",
                            "additive_kan_bce",
                            "maxnet",
                        )
                        if mname.replace("additive_", "").replace("kan_", "kan_")
                        or True
                    ):
                        # simpler: if status ok, skip
                        if (reg, sp, seed, "additive_kan_ipp") in done_keys or (
                            reg,
                            sp,
                            seed,
                            "maxnet",
                        ) in done_keys:
                            print(f"  skip {reg}/{sp} s{seed} (resume)", flush=True)
                            continue
                print(f"  {reg}/{sp} seed={seed} ...", flush=True)
                try:
                    rows = run_species(
                        data,
                        sp,
                        seed=seed,
                        lambda_s=None if args.lambda_s < 0 else args.lambda_s,
                        lbfgs_steps=args.lbfgs_steps,
                        models=models,
                        tune_lambda=not args.no_tune_lambda,
                        background_scheme=args.background,
                    )
                except Exception as e:
                    rows = [
                        {
                            "region": reg,
                            "taxon_group": "",
                            "species_id": sp,
                            "split_id": "pa_holdout",
                            "seed": seed,
                            "estimand": "global_conditional_density",
                            "model": "ERROR",
                            "objective": "error",
                            "background_scheme": args.background,
                            "integration_support": "",
                            "n_presence": 0,
                            "n_background": 0,
                            "n_PA_test": 0,
                            "basis_df": "",
                            "knots": "",
                            "lambda_s": "",
                            "lambda_r": "",
                            "weight_decay": 0.0,
                            "lambda_selection": "error",
                            "converged": False,
                            "runtime_s": 0.0,
                            "auc_roc": float("nan"),
                            "auprc": float("nan"),
                            "prg": float("nan"),
                            "cor": float("nan"),
                            "artifact_path": "",
                            "skip_reason": str(e)[:300],
                        }
                    ]
                    _save_status(st, {"ok": False, "error": str(e)})
                else:
                    _save_status(
                        st,
                        {
                            "ok": True,
                            "n_rows": len(rows),
                            "models": [r["model"] for r in rows],
                        },
                    )
                for r in rows:
                    key = (str(r["region"]), str(r["species_id"]), int(r["seed"]), str(r["model"]))
                    if key in done_keys:
                        continue
                    all_rows.append(r)
                    done_keys.add(key)
                    print(
                        f"    {r['model']}: AUC={r.get('auc_roc')} conv={r.get('converged')}",
                        flush=True,
                    )
                # checkpoint metrics after each species
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

        # per-region slice
        reg_df = pd.DataFrame([r for r in all_rows if r.get("region") == reg])
        if len(reg_df):
            reg_df.to_csv(outdir / "by_region" / f"metrics_{reg}.csv", index=False)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(metrics_path, index=False)
    write_summary(metrics, outdir)
    manifest = {
        "run_id": outdir.name,
        "regions": list(regions),
        "seeds": list(seeds),
        "models": list(models),
        "background_scheme": args.background,
        "estimand": "global_conditional_density",
        "integration_support": "full_background_50k"
        if args.background == "random_50k"
        else "tgb_other_po_same_group",
        "lambda_policy": "frozen_or_po4fold",
        "n_rows": len(metrics),
        "created_unix": time.time(),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.groupby(["region", "model"])["auc_roc"].mean().unstack())
    print("Done", outdir)


if __name__ == "__main__":
    main()
