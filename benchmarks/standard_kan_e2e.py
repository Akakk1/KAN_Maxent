#!/usr/bin/env python
"""End-to-end standard KAN–IPP pilot (CAN protocol).

See ``docs/standard_kan_e2e_can20_protocol.md``.

Compares ``standard_kan_ipp`` (KAN on scaled continuous x) to ``additive_kan_ipp``
under Phase-3 PO + random 50k → independent PA evaluation.

This is **not** the fair residual Deep hybrid experiment.
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
from kanmaxent.models.standard_kan_ipp import fit_standard_kan_ipp
from kanmaxent.nceas_fit import fit_hybrid_ipp

MODEL_ADD = "additive_kan_ipp"
MODEL_SKAN = "standard_kan_ipp"


def _status_path(outdir: Path, region: str, spid: str, seed: int) -> Path:
    return outdir / "status" / f"{region}_{spid}_s{seed}.json"


def run_species(
    region_data,
    species_id: str,
    *,
    seed: int = 0,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    lambda_kan: float = 1e-4,
    n_intervals: int = 6,
    degree: int = 3,
    additive_steps: int = 15,
    hidden_width: int = 4,
    adam_steps: int = 150,
    adam_lr: float = 0.03,
    lbfgs_steps: int = 10,
    models: Sequence[str] = (MODEL_ADD, MODEL_SKAN),
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
        "lambda_s": lambda_s,
        "lambda_r": lambda_r,
        "lambda_kan": lambda_kan,
        "lambda_selection": "frozen_default",
        "basis_df": n_intervals + degree,
        "knots": f"G={n_intervals},K={degree}",
        "background_scheme": "random_50k",
        "integration_support": "full_background_50k",
        "objective": "ipp",
        "protocol": "standard_kan_e2e_can20",
    }

    if n_po < 5:
        return [
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
            for m in models
        ]

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

    if MODEL_ADD in models:
        try:
            _, scores_add, meta_add = fit_hybrid_ipp(
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
                    "final_loss": float("nan"),
                    "hidden_width": None,
                    "kan_input": "n/a_additive_edges",
                }
            )
        except Exception as e:  # noqa: BLE001
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
                    "hidden_width": None,
                    "kan_input": "n/a_additive_edges",
                }
            )

    if MODEL_SKAN in models:
        try:
            _, scores_s, meta_s = fit_standard_kan_ipp(
                Xc_tr,
                Xk_tr,
                y_tr,
                Xc_te,
                Xk_te,
                hidden_width=hidden_width,
                n_intervals=n_intervals,
                degree=degree,
                lambda_kan=lambda_kan,
                lambda_r=lambda_r,
                seed=seed,
                adam_steps=adam_steps,
                adam_lr=adam_lr,
                lbfgs_steps=lbfgs_steps,
                freeze_grid=True,
                disable_silu=False,
            )
            m_s = ranking_metrics(y_pa, scores_s)
            rows.append(
                {
                    **base,
                    "model": MODEL_SKAN,
                    "n_background": n_bg,
                    "converged": bool(meta_s.get("converged", True)),
                    "runtime_s": float(meta_s.get("runtime_s", 0.0)),
                    "auc_roc": m_s.get("auc_roc", float("nan")),
                    "auprc": m_s.get("auprc", float("nan")),
                    "prg": m_s.get("prg", float("nan")),
                    "artifact_path": "",
                    "skip_reason": "",
                    "optimizer": meta_s.get("optimizer", "adam_then_lbfgs"),
                    "steps_adam": meta_s.get("steps_adam", adam_steps),
                    "steps_lbfgs": meta_s.get("steps_lbfgs", lbfgs_steps),
                    "final_loss": meta_s.get("final_loss", float("nan")),
                    "hidden_width": hidden_width,
                    "kan_input": "continuous_z",
                    "architecture": f"pykan_[{Xc_tr.shape[1]},{hidden_width},1]",
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    **base,
                    "model": MODEL_SKAN,
                    "n_background": n_bg,
                    "converged": False,
                    "runtime_s": 0.0,
                    "auc_roc": float("nan"),
                    "auprc": float("nan"),
                    "prg": float("nan"),
                    "artifact_path": "",
                    "skip_reason": str(e)[:300],
                    "hidden_width": hidden_width,
                    "kan_input": "continuous_z",
                }
            )

    return rows


def paired_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    a = metrics[metrics.model == MODEL_ADD].set_index(["region", "species_id", "seed"])
    s = metrics[metrics.model == MODEL_SKAN].set_index(["region", "species_id", "seed"])
    common = a.index.intersection(s.index)
    if len(common) == 0:
        return pd.DataFrame(
            columns=[
                "region",
                "species_id",
                "seed",
                "auc_additive",
                "auc_standard_kan",
                "delta_auc",
                "converged_additive",
                "converged_standard_kan",
            ]
        )
    out = pd.DataFrame(
        {
            "auc_additive": a.loc[common, "auc_roc"].values,
            "auc_standard_kan": s.loc[common, "auc_roc"].values,
            "converged_additive": a.loc[common, "converged"].values,
            "converged_standard_kan": s.loc[common, "converged"].values,
        },
        index=common,
    )
    out["delta_auc"] = out["auc_standard_kan"] - out["auc_additive"]
    return out.reset_index()


def _boot_mean_ci(x: np.ndarray, B: int = 1000, seed: int = 0) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(B)])
    return float(x.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def write_summary(
    outdir: Path,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    hidden_width: int,
    seeds: Sequence[int],
    species: Sequence[str],
) -> None:
    ok = paired[np.isfinite(paired.delta_auc) & paired.converged_standard_kan.astype(bool)]
    mu, lo, hi = _boot_mean_ci(ok.delta_auc.values if len(ok) else np.array([]))
    n_fail = int((~paired.converged_standard_kan.astype(bool)).sum()) if len(paired) else 0
    n_pos = int((ok.delta_auc > 0.01).sum()) if len(ok) else 0
    n_neg = int((ok.delta_auc < -0.01).sum()) if len(ok) else 0

    lines = [
        "# Standard KAN end-to-end CAN pilot — SUMMARY",
        "",
        f"**Protocol:** `docs/standard_kan_e2e_can20_protocol.md`",
        f"**Run dir:** `{outdir.name}`",
        f"**Architecture:** η = KAN_[P, h={hidden_width}, 1](T(x)) + X_cat @ β",
        f"**KAN input:** scaled continuous covariates (not additive φ edges)",
        f"**Seeds:** {list(seeds)}",
        f"**Species requested:** {len(species)}",
        "",
        "## Formula",
        "",
        "```",
        "eta = KAN(T(x)) + c^T beta",
        "T = per-feature quantile affine map into [-1, 1] fitted on PO+BG only",
        "```",
        "",
        "## Paired ΔAUC (standard_kan − additive)",
        "",
        f"- Complete-case species-seed rows: **{len(ok)}** (failures excluded from mean: **{n_fail}**)",
        f"- Mean ΔAUC: **{mu:+.4f}** (bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}])",
        f"- Median ΔAUC: **{(float(ok.delta_auc.median()) if len(ok) else float('nan')):+.4f}**",
        f"- n(Δ > 0.01): {n_pos}; n(Δ < −0.01): {n_neg}",
        "",
        "This experiment does **not** replace the fair residual Deep protocol.",
        "",
    ]
    (outdir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", default="CAN")
    p.add_argument("--species", default="", help="comma list; default all in region")
    p.add_argument("--seeds", default="0")
    p.add_argument("--outdir", default="")
    p.add_argument("--hidden-width", type=int, default=4)
    p.add_argument("--adam-steps", type=int, default=150)
    p.add_argument("--adam-lr", type=float, default=0.03)
    p.add_argument("--lbfgs-steps", type=int, default=10)
    p.add_argument("--additive-steps", type=int, default=15)
    p.add_argument("--lambda-kan", type=float, default=1e-4)
    p.add_argument(
        "--models",
        default="additive,standard_kan",
        help="additive,standard_kan (or full names)",
    )
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    alias = {
        "additive": MODEL_ADD,
        "add": MODEL_ADD,
        MODEL_ADD: MODEL_ADD,
        "standard_kan": MODEL_SKAN,
        "skan": MODEL_SKAN,
        "e2e": MODEL_SKAN,
        MODEL_SKAN: MODEL_SKAN,
    }
    models: List[str] = []
    for x in args.models.split(","):
        x = x.strip()
        if not x:
            continue
        if x not in alias:
            raise SystemExit(f"unknown model {x!r}; use additive,standard_kan")
        m = alias[x]
        if m not in models:
            models.append(m)

    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    outdir = Path(
        args.outdir
        if args.outdir
        else ROOT / "outputs" / f"standard_kan_e2e_{'_'.join(r.lower() for r in regions)}_{stamp}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "status").mkdir(exist_ok=True)

    manifest = {
        "protocol": "standard_kan_e2e_can20",
        "doc": "docs/standard_kan_e2e_can20_protocol.md",
        "formula": "eta = KAN_[P,h,1](T(x)) + X_cat @ beta",
        "kan_input": "continuous_z",
        "residual_additive_skip": False,
        "warm_start_additive": False,
        "regions": regions,
        "seeds": seeds,
        "models": models,
        "hidden_width": args.hidden_width,
        "adam_steps": args.adam_steps,
        "adam_lr": args.adam_lr,
        "lbfgs_steps": args.lbfgs_steps,
        "lambda_kan": args.lambda_kan,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_rows: List[Dict] = []
    metrics_path = outdir / "metrics.csv"
    if args.resume and metrics_path.is_file():
        prev = pd.read_csv(metrics_path)
        all_rows = prev.to_dict(orient="records")
        done = {
            (r["region"], r["species_id"], int(r["seed"]), r["model"]) for r in all_rows
        }
    else:
        done = set()

    t_run = time.perf_counter()
    species_all: List[str] = []

    for reg in regions:
        print(f"Loading region {reg}…", flush=True)
        region_data = load_region(reg)
        if args.species.strip():
            species = [s.strip() for s in args.species.split(",") if s.strip()]
        else:
            # stable order from region species list / po
            if hasattr(region_data, "species_ids") and region_data.species_ids:
                species = list(region_data.species_ids)
            else:
                sp_col = "spid" if "spid" in region_data.po.columns else "species"
                species = sorted(region_data.po[sp_col].astype(str).unique().tolist())
        species_all.extend(species)

        for seed in seeds:
            for sp in species:
                need = [m for m in models if (reg, sp, seed, m) not in done]
                if not need:
                    print(f"  skip {reg}/{sp} seed={seed} (done)", flush=True)
                    continue
                print(f"  fit {reg}/{sp} seed={seed} models={need}…", flush=True)
                t0 = time.perf_counter()
                rows = run_species(
                    region_data,
                    sp,
                    seed=seed,
                    lambda_kan=args.lambda_kan,
                    hidden_width=args.hidden_width,
                    adam_steps=args.adam_steps,
                    adam_lr=args.adam_lr,
                    lbfgs_steps=args.lbfgs_steps,
                    additive_steps=args.additive_steps,
                    models=need,
                )
                for r in rows:
                    all_rows.append(r)
                    done.add((r["region"], r["species_id"], int(r["seed"]), r["model"]))
                st = {
                    "region": reg,
                    "species_id": sp,
                    "seed": seed,
                    "runtime_s": time.perf_counter() - t0,
                    "rows": [
                        {
                            "model": r["model"],
                            "auc_roc": r.get("auc_roc"),
                            "converged": r.get("converged"),
                            "skip_reason": r.get("skip_reason"),
                            "final_loss": r.get("final_loss"),
                        }
                        for r in rows
                    ],
                }
                _status_path(outdir, reg, sp, seed).write_text(
                    json.dumps(st, indent=2), encoding="utf-8"
                )
                # checkpoint metrics each species
                pd.DataFrame(all_rows).to_csv(metrics_path, index=False)
                for r in rows:
                    auc = r.get("auc_roc")
                    print(
                        f"    {r['model']}: AUC={auc} conv={r.get('converged')} "
                        f"t={r.get('runtime_s', 0):.1f}s {r.get('skip_reason', '')}",
                        flush=True,
                    )

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(metrics_path, index=False)
    paired = paired_delta(metrics)
    paired.to_csv(outdir / "paired_delta_auc.csv", index=False)
    write_summary(
        outdir,
        metrics,
        paired,
        hidden_width=args.hidden_width,
        seeds=seeds,
        species=species_all,
    )
    manifest["runtime_s"] = time.perf_counter() - t_run
    manifest["n_metric_rows"] = len(metrics)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done → {outdir}", flush=True)
    if len(paired):
        ok = paired[np.isfinite(paired.delta_auc)]
        if len(ok):
            print(
                f"Mean ΔAUC (standard − additive) = {ok.delta_auc.mean():+.4f} "
                f"(n={len(ok)})",
                flush=True,
            )


if __name__ == "__main__":
    main()
