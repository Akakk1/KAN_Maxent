#!/usr/bin/env python
"""Phase 4 Deep KAN interaction ablation (PO + 50k BG → independent PA).

Compares deep2/deep3_kan_ipp vs additive_kan_ipp under the same Phase 3 protocol.
Fair defaults: residual, LBFGS, no-SiLU, warm-start, freeze-edges.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from kanmaxent.evaluation.metrics import ranking_metrics
from kanmaxent.models.deep_kan import fit_deep_kan_ipp
from kanmaxent.nceas_fit import fit_hybrid_ipp

MODEL_ADD = "additive_kan_ipp"
MODEL_D2 = "deep2_kan_ipp"
MODEL_D3 = "deep3_kan_ipp"


def _status_path(outdir: Path, region: str, spid: str, seed: int) -> Path:
    return outdir / "status" / f"{region}_{spid}_s{seed}.json"


def _parse_models(s: str) -> Tuple[str, ...]:
    raw = [x.strip() for x in s.split(",") if x.strip()]
    alias = {
        "additive": MODEL_ADD,
        "add": MODEL_ADD,
        MODEL_ADD: MODEL_ADD,
        "deep2": MODEL_D2,
        "d2": MODEL_D2,
        MODEL_D2: MODEL_D2,
        "deep3": MODEL_D3,
        "d3": MODEL_D3,
        MODEL_D3: MODEL_D3,
    }
    out: List[str] = []
    for x in raw:
        if x not in alias:
            raise ValueError(
                f"unknown model '{x}'; use additive,deep2,deep3 "
                f"(or full names *_kan_ipp)"
            )
        m = alias[x]
        if m not in out:
            out.append(m)
    return tuple(out)


def run_species(
    region_data,
    species_id: str,
    *,
    seed: int = 0,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    n_intervals: int = 6,
    degree: int = 3,
    additive_steps: int = 15,
    deep_steps: int = 80,
    deep_lr: float = 0.05,
    deep_optimizer: str = "lbfgs",
    lbfgs_steps: int = 20,
    residual: bool = True,
    disable_silu: bool = True,
    warm_start: bool = True,
    freeze_edges: bool = False,
    models: Sequence[str] = (MODEL_ADD, MODEL_D2),
    hidden_width: int = 4,
) -> List[Dict]:
    models = tuple(models)
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
        "lambda_s": lambda_s,
        "lambda_r": lambda_r,
        "lambda_selection": "frozen_default",
        "basis_df": n_intervals + degree,
        "knots": f"G={n_intervals},K={degree}",
        "background_scheme": "random_50k",
        "integration_support": "full_background_50k",
        "objective": "ipp",
    }

    if n_po < 5:
        rows = []
        for m in models:
            rows.append(
                {
                    **base,
                    "model": m,
                    "n_background": 0,
                    "converged": False,
                    "runtime_s": 0.0,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "artifact_path": "",
                    "skip_reason": "n_po<5",
                }
            )
        return rows

    train_df = build_po_bg_frame(po_sp, region_data.bg, covars)
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    n_bg = int((y_tr == 0).sum())

    mats = prepare_matrices(
        train_df,
        pa_env,
        region_data.continuous,
        region_data.categorical,
    )
    Xc_tr, Xk_tr = mats["X_cont_train"], mats["X_cat_train"]
    Xc_te, Xk_te = mats["X_cont_test"], mats["X_cat_test"]

    rows: List[Dict] = []
    warm_state = None

    # --- additive baseline (same protocol as Phase 3) ---
    if MODEL_ADD in models or any(m in (MODEL_D2, MODEL_D3) for m in models):
        # always fit additive when deep needs warm-start or additive is requested
        need_add_row = MODEL_ADD in models
        try:
            add_model, scores_add, meta_add = fit_hybrid_ipp(
                Xc_tr,
                Xk_tr,
                y_tr,
                Xc_te,
                Xk_te,
                n_intervals=n_intervals,
                degree=degree,
                lambda_s=lambda_s,
                lambda_r=lambda_r,
                seed=seed,
                steps=additive_steps,
            )
            m_add = ranking_metrics(y_pa, scores_add)
            # reuse additive state for deep warm-start (avoid second full LBFGS)
            warm_state = {
                "spline_coeffs": add_model.spline.coeffs.detach().cpu().numpy().copy(),
                "beta_cat": None
                if add_model.beta_cat is None
                else add_model.beta_cat.detach().cpu().numpy().copy(),
            }
            if need_add_row:
                rows.append(
                    {
                        **base,
                        "model": MODEL_ADD,
                        "n_background": n_bg,
                        "converged": bool(meta_add.get("converged", True)),
                        "runtime_s": float(meta_add.get("runtime_s", 0.0)),
                        "auc_roc": m_add.get("auc_roc", float("nan")),
                        "auprc": m_add.get("auprc", float("nan")),
                        "prg": m_add.get("prg", float("nan")),
                        "artifact_path": "",
                        "skip_reason": "",
                        "optimizer": "lbfgs",
                        "train_steps": additive_steps,
                    }
                )
        except Exception as e:  # noqa: BLE001
            if need_add_row:
                rows.append(
                    {
                        **base,
                        "model": MODEL_ADD,
                        "n_background": n_bg,
                        "converged": False,
                        "runtime_s": 0.0,
                        "auc_roc": float("nan"),
                        "auprc": float("nan"),
                        "prg": float("nan"),
                        "artifact_path": "",
                        "skip_reason": str(e)[:300],
                    }
                )

    def _fit_deep(model_name: str, depth: int) -> None:
        try:
            _, scores_d, meta_d = fit_deep_kan_ipp(
                Xc_tr,
                Xk_tr,
                y_tr,
                Xc_te,
                Xk_te,
                n_intervals=n_intervals,
                degree=degree,
                lambda_s=lambda_s,
                lambda_r=lambda_r,
                seed=seed,
                steps=deep_steps,
                lr=deep_lr,
                freeze_grid=True,
                residual=residual,
                disable_silu=disable_silu,
                warm_start_additive=warm_start and warm_state is None,
                warm_start_state=warm_state if warm_start else None,
                optimizer=deep_optimizer,
                lbfgs_steps=lbfgs_steps,
                freeze_edges_after_warmstart=freeze_edges,
                adaptive_budget=True,
                depth=depth,
                hidden_width=hidden_width,
            )
            m_d = ranking_metrics(y_pa, scores_d)
            rows.append(
                {
                    **base,
                    "model": model_name,
                    "n_background": n_bg,
                    "converged": bool(meta_d.get("converged", True)),
                    "runtime_s": float(meta_d.get("runtime_s", 0.0)),
                    "auc_roc": m_d.get("auc_roc", float("nan")),
                    "auprc": m_d.get("auprc", float("nan")),
                    "prg": m_d.get("prg", float("nan")),
                    "artifact_path": "",
                    "skip_reason": "",
                    "optimizer": meta_d.get("optimizer", deep_optimizer),
                    "train_steps": meta_d.get("steps_lbfgs", lbfgs_steps),
                    "steps_adam": meta_d.get("steps_adam", 0),
                    "final_loss": meta_d.get("final_loss", float("nan")),
                    "deep_lr": deep_lr,
                    "residual": residual,
                    "disable_silu": disable_silu,
                    "warm_start": warm_start,
                    "freeze_edges": freeze_edges,
                    "depth": depth,
                    "hidden_width": hidden_width if depth == 3 else None,
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    **base,
                    "model": model_name,
                    "n_background": n_bg,
                    "converged": False,
                    "runtime_s": 0.0,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "artifact_path": "",
                    "skip_reason": str(e)[:300],
                    "depth": depth,
                    "hidden_width": hidden_width if depth == 3 else None,
                }
            )

    if MODEL_D2 in models:
        _fit_deep(MODEL_D2, 2)
    if MODEL_D3 in models:
        _fit_deep(MODEL_D3, 3)

    return rows


def paired_summary(
    metrics: pd.DataFrame,
    deep_model: str = MODEL_D2,
) -> pd.DataFrame:
    """Species-level ΔAUC = deep − additive."""
    a = metrics[metrics.model == MODEL_ADD].set_index(
        ["region", "species_id", "seed"]
    )["auc_roc"]
    d = metrics[metrics.model == deep_model].set_index(
        ["region", "species_id", "seed"]
    )["auc_roc"]
    common = a.index.intersection(d.index)
    if len(common) == 0:
        return pd.DataFrame(
            columns=[
                "region",
                "species_id",
                "seed",
                "auc_additive",
                "auc_deep",
                "delta_auc",
                "deep_model",
            ]
        )
    out = pd.DataFrame(
        {
            "auc_additive": a.loc[common].values,
            "auc_deep": d.loc[common].values,
            "delta_auc": (d.loc[common] - a.loc[common]).values,
            "deep_model": deep_model,
        },
        index=common,
    ).reset_index()
    # keep legacy column for deep2 readers
    if deep_model == MODEL_D2:
        out["auc_deep2"] = out["auc_deep"]
    elif deep_model == MODEL_D3:
        out["auc_deep3"] = out["auc_deep"]
    return out


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def write_summary(
    metrics: pd.DataFrame,
    paired_by_model: Dict[str, pd.DataFrame],
    outdir: Path,
    *,
    hidden_width: int = 4,
) -> None:
    has_d3 = MODEL_D3 in metrics["model"].values if len(metrics) else False
    has_d2 = MODEL_D2 in metrics["model"].values if len(metrics) else False
    title = "Phase 4 Deep KAN Ablation SUMMARY"
    if has_d3 and has_d2:
        arch = (
            "residual η = Σφ + mixer(φ); deep2: Φ:P→1; "
            f"deep3: [P, h={hidden_width}, 1] (grid frozen, no SiLU)"
        )
    elif has_d3:
        arch = f"residual η = Σφ + Ψ(Φ(φ)); deep3 [P, h={hidden_width}, 1]"
    else:
        arch = "residual η = Σφ + Φ(φ); deep2 KANLayer P→1 (grid frozen)"

    lines = [
        f"# {title}",
        "",
        f"**Run:** `{outdir.name}`",
        f"**Architecture:** {arch}",
        "**Protocol:** PO + random_50k → independent PA (same as Phase 3)",
        "",
        "## Mean AUC",
        "",
    ]
    for m, sub in metrics.groupby("model"):
        auc = sub["auc_roc"].dropna()
        lines.append(f"- **{m}**: {auc.mean():.4f} ± {auc.std():.4f} (n={len(auc)})")
    lines.append("")

    for deep_m, paired in paired_by_model.items():
        short = "deep2" if deep_m == MODEL_D2 else ("deep3" if deep_m == MODEL_D3 else deep_m)
        lines.append(f"## Paired ΔAUC ({short} − additive)")
        lines.append("")
        if len(paired):
            mean, lo, hi = bootstrap_mean_ci(paired["delta_auc"].values)
            n_pos = int((paired["delta_auc"] > 0.01).sum())
            n_neg = int((paired["delta_auc"] < -0.01).sum())
            lines.append(
                f"- mean Δ = **{mean:+.4f}** [{lo:+.4f}, {hi:+.4f}] (bootstrap 95% CI)"
            )
            lines.append(f"- median Δ = {paired['delta_auc'].median():+.4f}")
            lines.append(f"- n species = {len(paired)}")
            lines.append(f"- n with Δ > 0.01: **{n_pos}**; Δ < −0.01: {n_neg}")
            lines.append("")
            lines.append("### Per species")
            lines.append("")
            auc_col = "auc_deep"
            lines.append(f"| species | AUC add | AUC {short} | Δ |")
            lines.append("|:--------|--------:|----------:|--:|")
            for _, r in paired.sort_values("delta_auc", ascending=False).iterrows():
                lines.append(
                    f"| {r['species_id']} | {r['auc_additive']:.4f} | "
                    f"{r[auc_col]:.4f} | {r['delta_auc']:+.4f} |"
                )
        else:
            lines.append("- (no paired rows)")
        lines.append("")

    if has_d2 and has_d3 and MODEL_ADD in metrics["model"].values:
        # optional deep3 − deep2 table
        a2 = metrics[metrics.model == MODEL_D2].set_index(
            ["region", "species_id", "seed"]
        )["auc_roc"]
        a3 = metrics[metrics.model == MODEL_D3].set_index(
            ["region", "species_id", "seed"]
        )["auc_roc"]
        common = a2.index.intersection(a3.index)
        if len(common):
            d32 = (a3.loc[common] - a2.loc[common]).values
            mean, lo, hi = bootstrap_mean_ci(d32)
            lines.append("## Paired ΔAUC (deep3 − deep2)")
            lines.append("")
            lines.append(
                f"- mean Δ = **{mean:+.4f}** [{lo:+.4f}, {hi:+.4f}] (bootstrap 95% CI)"
            )
            lines.append("")

    lines.append("## Note")
    lines.append("")
    lines.append(
        "- Deep-3 is an **appendix null** under the fair residual protocol "
        "(not a novelty rescue)."
    )
    lines.append("")
    lines.append("*Phase 4 Deep KAN interaction ablation.*")
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 4 Deep KAN ablation (deep2/deep3)")
    p.add_argument("--regions", default="CAN")
    p.add_argument("--species", default="", help="comma list; default all")
    p.add_argument("--seeds", default="0")
    p.add_argument("--outdir", default="")
    p.add_argument("--lambda-s", type=float, default=1e-2)
    p.add_argument("--deep-steps", type=int, default=80, help="Adam steps (if used)")
    p.add_argument("--deep-lr", type=float, default=0.05)
    p.add_argument(
        "--deep-optimizer",
        default="lbfgs",
        choices=["lbfgs", "adam", "adam_then_lbfgs"],
        help="fair default: lbfgs (same family as additive)",
    )
    p.add_argument(
        "--lbfgs-steps",
        type=int,
        default=12,
        help="outer LBFGS steps for mixer (default 12; bulk fair)",
    )
    p.add_argument("--no-residual", action="store_true", help="use eta=mixer(phi) only")
    p.add_argument("--silu", action="store_true", help="enable pykan SiLU base")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument(
        "--freeze-edges",
        action="store_true",
        default=True,
        help="after warm-start, only train mixer (default on for fair bulk)",
    )
    p.add_argument(
        "--no-freeze-edges",
        action="store_true",
        help="jointly fine-tune edges + mixer after warm-start",
    )
    p.add_argument(
        "--models",
        default="additive,deep2",
        help="comma list: additive,deep2,deep3 (default additive,deep2)",
    )
    p.add_argument(
        "--hidden-width",
        type=int,
        default=4,
        help="Deep-3 hidden width h in [P,h,1] (default 4)",
    )
    p.add_argument("--additive-steps", type=int, default=15)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    models = _parse_models(args.models)
    regions = tuple(x.strip().upper() for x in args.regions.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    if args.outdir:
        outdir = Path(args.outdir)
        if not outdir.is_absolute():
            # allow outputs/... relative to ROOT; strip duplicate outputs/
            s = str(outdir).replace("\\", "/")
            if s.startswith("outputs/"):
                outdir = ROOT / s
            else:
                outdir = ROOT / outdir
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = ROOT / "outputs" / f"deepkan_ablation_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "status").mkdir(exist_ok=True)

    metrics_path = outdir / "metrics.csv"
    all_rows: List[Dict] = []
    if args.resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_rows = prev.to_dict(orient="records")
        print(f"Resumed {len(all_rows)} rows")

    done = {
        (str(r.get("region")), str(r.get("species_id")), int(r.get("seed", 0)), str(r.get("model")))
        for r in all_rows
    }

    print(
        f"Regions={regions} seeds={seeds} models={models} "
        f"h={args.hidden_width} out={outdir}",
        flush=True,
    )

    for reg in regions:
        data = load_region(reg)
        species = (
            [s.strip() for s in args.species.split(",") if s.strip()]
            if args.species.strip()
            else list(data.species_ids)
        )
        print(f"== {reg}: {len(species)} species ==", flush=True)
        for sp in species:
            for seed in seeds:
                if all((reg, sp, seed, m) in done for m in models):
                    print(f"  skip {reg}/{sp} s{seed}", flush=True)
                    continue
                print(f"  {reg}/{sp} seed={seed} ...", flush=True)
                t0 = time.perf_counter()
                try:
                    rows = run_species(
                        data,
                        sp,
                        seed=seed,
                        lambda_s=args.lambda_s,
                        additive_steps=args.additive_steps,
                        deep_steps=args.deep_steps,
                        deep_lr=args.deep_lr,
                        deep_optimizer=args.deep_optimizer,
                        lbfgs_steps=args.lbfgs_steps,
                        residual=not args.no_residual,
                        disable_silu=not args.silu,
                        warm_start=not args.no_warm_start,
                        freeze_edges=(not args.no_freeze_edges),
                        models=models,
                        hidden_width=args.hidden_width,
                    )
                except Exception as e:  # noqa: BLE001
                    rows = [
                        {
                            "region": reg,
                            "species_id": sp,
                            "seed": seed,
                            "model": "ERROR",
                            "skip_reason": str(e)[:300],
                            "auc_roc": float("nan"),
                            "converged": False,
                        }
                    ]
                    (_status_path(outdir, reg, sp, seed)).write_text(
                        json.dumps({"ok": False, "error": str(e)}), encoding="utf-8"
                    )
                else:
                    (_status_path(outdir, reg, sp, seed)).write_text(
                        json.dumps(
                            {
                                "ok": True,
                                "models": [r["model"] for r in rows],
                                "runtime_s": time.perf_counter() - t0,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                for r in rows:
                    key = (
                        str(r["region"]),
                        str(r["species_id"]),
                        int(r["seed"]),
                        str(r["model"]),
                    )
                    if key in done:
                        continue
                    all_rows.append(r)
                    done.add(key)
                    print(
                        f"    {r['model']}: AUC={r.get('auc_roc')} "
                        f"conv={r.get('converged')} t={r.get('runtime_s')}",
                        flush=True,
                    )
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(metrics_path, index=False)

    paired_by_model: Dict[str, pd.DataFrame] = {}
    for dm in (MODEL_D2, MODEL_D3):
        if dm in models and len(metrics) and (metrics.model == dm).any():
            pr = paired_summary(metrics, deep_model=dm)
            paired_by_model[dm] = pr
            suffix = "deep2" if dm == MODEL_D2 else "deep3"
            pr.to_csv(outdir / f"paired_delta_auc_{suffix}.csv", index=False)
            if dm == MODEL_D2:
                # legacy filename
                pr.to_csv(outdir / "paired_delta_auc.csv", index=False)

    # combined multi-model paired if both present
    if MODEL_D2 in paired_by_model and MODEL_D3 in paired_by_model:
        p2 = paired_by_model[MODEL_D2].set_index(["region", "species_id", "seed"])
        p3 = paired_by_model[MODEL_D3].set_index(["region", "species_id", "seed"])
        common = p2.index.intersection(p3.index)
        comb = pd.DataFrame(
            {
                "auc_additive": p2.loc[common, "auc_additive"].values,
                "auc_deep2": p2.loc[common, "auc_deep"].values,
                "auc_deep3": p3.loc[common, "auc_deep"].values,
                "delta_deep2": p2.loc[common, "delta_auc"].values,
                "delta_deep3": p3.loc[common, "delta_auc"].values,
            },
            index=common,
        ).reset_index()
        comb["delta_d3_minus_d2"] = comb["auc_deep3"] - comb["auc_deep2"]
        comb.to_csv(outdir / "paired_delta_auc_all.csv", index=False)

    write_summary(
        metrics, paired_by_model, outdir, hidden_width=args.hidden_width
    )

    freeze_edges = not args.no_freeze_edges
    manifest = {
        "run_id": outdir.name,
        "regions": list(regions),
        "seeds": list(seeds),
        "models": list(models),
        "architecture": (
            "residual sum(phi)+mixer(phi)" if not args.no_residual else "mixer(phi)"
        ),
        "deep3_hidden_width": args.hidden_width if MODEL_D3 in models else None,
        "backend": "pykan.KANLayer",
        "freeze_grid": True,
        "disable_silu": not args.silu,
        "warm_start_additive": not args.no_warm_start,
        "freeze_edges": freeze_edges,
        "deep_optimizer": args.deep_optimizer,
        "lbfgs_steps": args.lbfgs_steps,
        "protocol": "PO_random50k_to_PA",
        "lambda_s": args.lambda_s,
        "deep_steps": args.deep_steps,
        "deep_lr": args.deep_lr,
        "n_rows": len(metrics),
        "created_unix": time.time(),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for dm, pr in paired_by_model.items():
        print(f"--- {dm} ---")
        print(pr.to_string(index=False) if len(pr) else "no paired")
        if len(pr):
            mean, lo, hi = bootstrap_mean_ci(pr["delta_auc"].values)
            print(f"mean ΔAUC={mean:+.4f} [{lo:+.4f},{hi:+.4f}]")
    print("Done", outdir)


if __name__ == "__main__":
    main()
