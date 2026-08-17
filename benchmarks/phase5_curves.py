#!/usr/bin/env python
"""Phase 5: response curves φ_p, shape metrics, CPU-first device policy.

Modes:
  pilot   — Ginkgo P0/P1 + one NCEAS species P2 timing (CPU warm-start)
  ginkgo  — full Ginkgo curves (+ optional bootstrap)
  nceas   — sampled NCEAS species curves
  plot    — build summary figures from exported CSVs
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.data.categorical import prepare_matrices
from kanmaxent.data.ginkgo_io import env_columns, load_ginkgo
from kanmaxent.data.nceas_io import (
    build_po_bg_frame,
    load_region,
    pa_labels_and_env_for_species,
    split_species,
)
from kanmaxent.data.preprocess import FoldPreprocessor
from kanmaxent.interpret.curves import export_component_curves
from kanmaxent.interpret.device import parse_device_policy
from kanmaxent.interpret.shape_metrics import curve_peak_agreement
from kanmaxent.nceas_fit import fit_hybrid_gam_ipp, fit_hybrid_ipp
from kanmaxent.trainers import fit_ipp_kan, make_shared_kan_shell
from kanmaxent.utils import as_numpy_f64


def _outdir(name: str) -> Path:
    """Resolve output directory under repo outputs/ (avoid outputs/outputs/)."""
    p = Path(name)
    if not p.is_absolute():
        # strip leading "outputs/" if user already passed it
        parts = p.parts
        if parts and parts[0] == "outputs":
            p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        p = ROOT / "outputs" / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def fit_ginkgo_full(
    *,
    seed: int = 0,
    lambda_s: float = 1e-2,
    n_intervals: int = 6,
    degree: int = 3,
    steps: int = 25,
    device_policy=None,
) -> Tuple[object, object, np.ndarray, List[str], FoldPreprocessor, dict]:
    """Fit additive KAN-IPP on full Ginkgo table (mechanism curves)."""
    t0 = time.perf_counter()
    pol = device_policy or parse_device_policy("cpu")
    fit_dev = pol.resolve_fit_device()
    # force float64 CPU path for LBFGS stability
    assert fit_dev.type == "cpu" or not pol.prefer_cpu_lbfgs

    df = load_ginkgo()
    env = env_columns(df)
    y = df["label"].to_numpy(dtype=np.float64)
    X_raw = df[env].to_numpy(dtype=np.float64)
    prep = FoldPreprocessor().fit(X_raw)
    X = prep.transform(X_raw)

    model, res = fit_ipp_kan(
        X,
        y,
        X,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        seed=seed,
        steps=steps,
    )
    meta = {
        "runtime_s": time.perf_counter() - t0,
        "converged": res.converged,
        "fit_device": str(fit_dev),
        "n_rows": int(len(y)),
        "n_presence": int(y.sum()),
        "env": env,
        "device_policy": pol.to_manifest(),
        "trainer_runtime_s": res.runtime_s,
    }
    return model, None, X, env, prep, meta


def fit_nceas_species(
    region: str,
    species_id: str,
    *,
    seed: int = 0,
    lambda_s: float = 1e-2,
    lbfgs_steps: int = 15,
    device_policy=None,
) -> Tuple[object, object, dict, dict]:
    """Fit additive KAN + same-basis GAM; return models and matrices meta."""
    t0 = time.perf_counter()
    pol = device_policy or parse_device_policy("cpu")
    fit_dev = pol.resolve_fit_device()

    data = load_region(region)
    po_sp = split_species(data.po, species_id)
    n_po = len(po_sp)
    if n_po < 5:
        raise ValueError(f"n_po<5 for {region}/{species_id}")
    y_pa, pa_env, taxon = pa_labels_and_env_for_species(data, species_id)
    covars = data.continuous + data.categorical
    train_df = build_po_bg_frame(po_sp, data.bg, covars)
    y_tr = train_df["occ"].to_numpy(dtype=np.float64)
    mats = prepare_matrices(train_df, pa_env, data.continuous, data.categorical)
    Xc_tr = mats["X_cont_train"]
    Xk_tr = mats["X_cat_train"]
    Xc_te = mats["X_cont_test"]
    Xk_te = mats["X_cat_test"]

    model, scores, info = fit_hybrid_ipp(
        Xc_tr,
        Xk_tr,
        y_tr,
        Xc_te,
        Xk_te,
        lambda_s=lambda_s,
        seed=seed,
        steps=lbfgs_steps,
    )
    # GAM on same hybrid shell basis; load spline coeffs for φ export
    scores_g, info_g = fit_hybrid_gam_ipp(
        model, Xc_tr, Xk_tr, y_tr, Xc_te, Xk_te, B_train=info.get("B_train")
    )
    from kanmaxent.nceas_fit import HybridSplineCat

    gam_model = HybridSplineCat(
        model.n_continuous,
        model.n_cat_oh,
        n_intervals=model.spline.n_intervals,
        degree=model.spline.degree,
        lambda_s=model.lambda_s,
        lambda_r=model.lambda_r,
    )
    gam_model.fit_bounds(Xc_tr)
    beta = np.asarray(info_g["beta"], dtype=np.float64)
    n_b = model.spline.n_basis_per_feature
    P = model.n_continuous
    with torch.no_grad():
        for p in range(P):
            gam_model.spline.coeffs[p].copy_(
                torch.as_tensor(beta[p * n_b : (p + 1) * n_b], dtype=torch.float64)
            )
        if gam_model.beta_cat is not None and model.n_cat_oh > 0:
            gam_model.beta_cat.copy_(
                torch.as_tensor(beta[P * n_b : P * n_b + model.n_cat_oh], dtype=torch.float64)
            )

    meta = {
        "region": region,
        "species_id": species_id,
        "taxon_group": taxon,
        "n_po": n_po,
        "n_bg": int((y_tr == 0).sum()),
        "n_PA": int(len(y_pa)),
        "runtime_s": time.perf_counter() - t0,
        "fit_device": str(fit_dev),
        "device_policy": pol.to_manifest(),
        "kan_converged": bool(info.get("converged", True)),
        "gam_converged": bool(info_g.get("converged", True)),
        "continuous": list(data.continuous),
        "categorical": list(data.categorical),
        "scaler_mean": mats["scaler"].mean_.tolist()
        if hasattr(mats["scaler"], "mean_")
        else None,
        "scaler_scale": mats["scaler"].scale_.tolist()
        if hasattr(mats["scaler"], "scale_")
        else None,
    }
    pack = {
        "Xc_tr": Xc_tr,
        "Xk_tr": Xk_tr,
        "y_tr": y_tr,
        "env_cont": list(data.continuous),
        "scaler": mats["scaler"],
        "gam_model": gam_model,
    }
    return model, pack, meta, {"scores_kan": scores, "scores_gam": scores_g}


def export_additive_curves(
    model,
    env_cont: Sequence[str],
    Xc_tr: np.ndarray,
    out_dir: Path,
    *,
    scaler_mean=None,
    scaler_scale=None,
    n_grid: int = 100,
) -> Path:
    """Export φ for continuous edges of HybridSplineCat or AdditiveSplineKAN."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spline = model.spline if hasattr(model, "spline") else model
    X_ref = torch.as_tensor(Xc_tr, dtype=torch.float64)
    qlo = np.quantile(Xc_tr, 0.01, axis=0)
    qhi = np.quantile(Xc_tr, 0.99, axis=0)
    return export_component_curves(
        spline,
        env_cont,
        out_dir,
        n_grid=n_grid,
        X_ref=X_ref,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        train_q=(qlo, qhi),
    )


def select_nceas_species(metrics_csv: Path, per_region_high: int = 2, per_region_mid: int = 1) -> pd.DataFrame:
    m = pd.read_csv(metrics_csv)
    m = m[(m["model"] == "additive_kan_ipp") & m["auc_roc"].notna()].copy()
    rows = []
    for reg, sub in m.groupby("region"):
        sub = sub.sort_values("auc_roc", ascending=False)
        high = sub.head(per_region_high)
        for _, r in high.iterrows():
            rows.append({**r.to_dict(), "sample_reason": "high_auc"})
        mid = sub.iloc[len(sub) // 3 : 2 * len(sub) // 3]
        if len(mid) and per_region_mid > 0:
            pick = mid.sample(n=min(per_region_mid, len(mid)), random_state=0)
            for _, r in pick.iterrows():
                if r["species_id"] in set(high["species_id"]):
                    continue
                rows.append({**r.to_dict(), "sample_reason": "mid_auc"})
    return pd.DataFrame(rows)


def run_pilot(args) -> None:
    out = _outdir("phase5_cpu_pilot")
    lines = [
        "# Phase 5 CPU pilot TIMING",
        "",
        f"**When:** {datetime.now(timezone.utc).isoformat()}",
        f"**Device request:** {args.device}",
        f"**prefer_cpu_lbfgs:** {not args.allow_gpu_lbfgs}",
        "",
    ]
    pol = parse_device_policy(
        args.device, prefer_cpu_lbfgs=not args.allow_gpu_lbfgs
    )

    # P0 Ginkgo fit + export no bootstrap
    t0 = time.perf_counter()
    model, _, X, env, prep, meta = fit_ginkgo_full(seed=0, device_policy=pol)
    export_additive_curves(
        model,
        env,
        X,
        out / "ginkgo_p0",
        scaler_mean=prep.mean_,
        scaler_scale=prep.scale_,
    )
    p0 = time.perf_counter() - t0
    lines += [
        "## P0 Ginkgo fit + full φ export (no bootstrap)",
        f"- wall_s: **{p0:.2f}**",
        f"- trainer_s: {meta.get('trainer_runtime_s')}",
        f"- fit_device: {meta.get('fit_device')}",
        f"- n_presence: {meta.get('n_presence')} / n_rows: {meta.get('n_rows')}",
        "",
    ]
    print(f"P0 done in {p0:.2f}s", flush=True)

    # P1 bootstrap B times: refit on resampled rows
    B = args.pilot_boot
    t1 = time.perf_counter()
    df = load_ginkgo()
    env = env_columns(df)
    y = df["label"].to_numpy(dtype=np.float64)
    X_raw = df[env].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(0)
    boot_times = []
    for b in range(B):
        tb = time.perf_counter()
        idx = rng.integers(0, len(y), size=len(y))
        prep = FoldPreprocessor().fit(X_raw[idx])
        Xs = prep.transform(X_raw[idx])
        ys = y[idx]
        fit_ipp_kan(Xs, ys, Xs, seed=b, steps=15, lambda_s=1e-2)
        boot_times.append(time.perf_counter() - tb)
    p1 = time.perf_counter() - t1
    lines += [
        f"## P1 Ginkgo bootstrap B={B} (refit each)",
        f"- wall_s: **{p1:.2f}**",
        f"- mean_per_boot_s: **{float(np.mean(boot_times)):.2f}**",
        f"- extrapolate B=50: ~{float(np.mean(boot_times))*50:.1f}s",
        f"- extrapolate B=200: ~{float(np.mean(boot_times))*200:.1f}s",
        "",
    ]
    print(f"P1 done in {p1:.2f}s (mean {np.mean(boot_times):.2f}s/boot)", flush=True)

    # P2 NCEAS one species
    t2 = time.perf_counter()
    model2, pack, meta2, _ = fit_nceas_species(
        "CAN", "can02", seed=0, device_policy=pol, lbfgs_steps=15
    )
    export_additive_curves(
        model2,
        pack["env_cont"],
        pack["Xc_tr"],
        out / "nceas_can02_p2",
        scaler_mean=meta2.get("scaler_mean"),
        scaler_scale=meta2.get("scaler_scale"),
    )
    # B boots
    boot_n = []
    rng = np.random.default_rng(1)
    Xc = pack["Xc_tr"]
    Xk = pack["Xk_tr"]
    ytr = pack["y_tr"]
    for b in range(B):
        tb = time.perf_counter()
        idx = rng.integers(0, len(ytr), size=len(ytr))
        fit_hybrid_ipp(
            Xc[idx],
            Xk[idx] if Xk is not None and len(Xk) else Xk,
            ytr[idx],
            Xc[idx],
            Xk[idx] if Xk is not None and len(Xk) else Xk,
            seed=b,
            steps=12,
        )
        boot_n.append(time.perf_counter() - tb)
    p2 = time.perf_counter() - t2
    lines += [
        f"## P2 NCEAS CAN/can02 fit + export + B={B} bootstrap refits",
        f"- wall_s (incl fit+export+boots): **{p2:.2f}**",
        f"- single fit_s: {meta2.get('runtime_s'):.2f}",
        f"- mean_per_boot_s: **{float(np.mean(boot_n)):.2f}**",
        f"- extrapolate B=50: ~{float(np.mean(boot_n))*50 + float(meta2.get('runtime_s',0)):.1f}s per species",
        f"- extrapolate 20 spp × B=50 (serial): ~{(float(np.mean(boot_n))*50 + 15)*20/60:.1f} min",
        f"- extrapolate 20 spp × B=50 (4-way parallel): ~{(float(np.mean(boot_n))*50 + 15)*20/4/60:.1f} min",
        "",
    ]
    print(f"P2 done in {p2:.2f}s", flush=True)

    # decision
    per_sp_b50 = float(np.mean(boot_n)) * 50 + float(meta2.get("runtime_s", 15))
    lines += [
        "## Decision (Plan_Phase5 §4.3)",
        "",
    ]
    if per_sp_b50 < 5 * 60:
        lines.append(
            f"- Per-species B=50 ≈ **{per_sp_b50:.0f}s** < 5 min → **use pure CPU** for Phase 5 bulk; "
            "do **not** enable GPU LBFGS."
        )
        decision = "cpu_only"
    elif per_sp_b50 < 10 * 60:
        lines.append(
            f"- Per-species B=50 ≈ **{per_sp_b50:.0f}s** moderate → **CPU + species parallel** "
            "(MAX_JOBS≤4); GPU not required."
        )
        decision = "cpu_parallel"
    else:
        lines.append(
            f"- Per-species B=50 ≈ **{per_sp_b50:.0f}s** slow → consider reducing B or "
            "parallel species; GPU LBFGS still **not preferred** (stability)."
        )
        decision = "cpu_parallel_reduce_B"
    lines += [
        f"- **decision:** `{decision}`",
        f"- device_policy events: {pol.to_manifest()}",
        "",
        "*Generated by benchmarks/phase5_curves.py --mode pilot*",
    ]
    (out / "TIMING.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "device_policy.json").write_text(
        json.dumps(pol.to_manifest(), indent=2), encoding="utf-8"
    )
    print("Wrote", out / "TIMING.md")
    print("\n".join(lines))


def run_ginkgo(args) -> None:
    out = _outdir(args.outdir or f"phase5_curves_{datetime.now(timezone.utc).strftime('%Y%m%d')}")
    gdir = out / "ginkgo"
    pol = parse_device_policy(args.device, prefer_cpu_lbfgs=not args.allow_gpu_lbfgs)
    model, _, X, env, prep, meta = fit_ginkgo_full(seed=args.seed, device_policy=pol)
    export_additive_curves(
        model, env, X, gdir / "kan_ipp", scaler_mean=prep.mean_, scaler_scale=prep.scale_
    )
    # same-basis GAM (scipy L-BFGS-B on shared design)
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    from kanmaxent.models.bspline import second_difference_penalty_matrix

    y = load_ginkgo()["label"].to_numpy(dtype=np.float64)
    B = model.full_design_matrix(torch.as_tensor(X, dtype=torch.float64))
    Bnp = as_numpy_f64(B)
    n_b = model.n_basis_per_feature
    P = model.n_features
    Ds = [second_difference_penalty_matrix(n_b) for _ in range(P)]
    counts = y
    w = np.ones(len(y))

    def nll(beta):
        eta = Bnp @ beta
        n = counts.sum()
        loss = -float(np.dot(counts, eta)) + float(n * logsumexp(eta + np.log(w)))
        loss += 1e-6 * float(np.dot(beta, beta))
        for p in range(P):
            D = Ds[p]
            if D.size:
                c = beta[p * n_b : (p + 1) * n_b]
                Dc = D @ c
                loss += 1e-2 * float(np.dot(Dc, Dc))
        return loss

    beta0 = np.zeros(Bnp.shape[1])
    res = minimize(nll, beta0, method="L-BFGS-B", options={"maxiter": 400})
    gam_model = make_shared_kan_shell(len(env), X, lambda_s=1e-2)
    with torch.no_grad():
        for p in range(P):
            gam_model.coeffs[p].copy_(
                torch.as_tensor(res.x[p * n_b : (p + 1) * n_b], dtype=torch.float64)
            )
    export_additive_curves(
        gam_model,
        env,
        X,
        gdir / "gam_ipp",
        scaler_mean=prep.mean_,
        scaler_scale=prep.scale_,
    )

    # agreement table
    agr = []
    for name in env:
        ka = pd.read_csv(gdir / "kan_ipp" / f"phi_{name}.csv")
        ga = pd.read_csv(gdir / "gam_ipp" / f"phi_{name}.csv")
        r = curve_peak_agreement(ka["x_scaled"].values, ka["phi"].values, ga["phi"].values)
        r["feature"] = name
        agr.append(r)
    pd.DataFrame(agr).to_csv(gdir / "kan_vs_gam_agreement.csv", index=False)
    (gdir / "fit_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("Ginkgo curves →", gdir)


def run_nceas(args) -> None:
    out = _outdir(args.outdir or f"phase5_curves_{datetime.now(timezone.utc).strftime('%Y%m%d')}")
    metrics = ROOT / "outputs/nceas_full_20260714_phase3/metrics_all.csv"
    sample = select_nceas_species(metrics)
    if args.species:
        want = set(s.strip() for s in args.species.split(",") if s.strip())
        sample = sample[sample["species_id"].isin(want)]
    sample.to_csv(out / "species_manifest.csv", index=False)
    pol = parse_device_policy(args.device, prefer_cpu_lbfgs=not args.allow_gpu_lbfgs)
    print(f"NCEAS sample n={len(sample)}", flush=True)

    rows_meta = []
    for _, r in sample.iterrows():
        reg, sp = r["region"], r["species_id"]
        sp_dir = out / "nceas" / reg / sp
        print(f"  {reg}/{sp} ...", flush=True)
        try:
            model, pack, meta, _ = fit_nceas_species(
                reg, sp, seed=args.seed, device_policy=pol, lbfgs_steps=args.lbfgs_steps
            )
            export_additive_curves(
                model,
                pack["env_cont"],
                pack["Xc_tr"],
                sp_dir / "kan_ipp",
                scaler_mean=meta.get("scaler_mean"),
                scaler_scale=meta.get("scaler_scale"),
            )
            export_additive_curves(
                pack["gam_model"],
                pack["env_cont"],
                pack["Xc_tr"],
                sp_dir / "gam_ipp",
                scaler_mean=meta.get("scaler_mean"),
                scaler_scale=meta.get("scaler_scale"),
            )
            # per-feature KAN vs GAM agreement
            agr_rows = []
            for name in pack["env_cont"]:
                ka = pd.read_csv(sp_dir / "kan_ipp" / f"phi_{name}.csv")
                ga = pd.read_csv(sp_dir / "gam_ipp" / f"phi_{name}.csv")
                rr = curve_peak_agreement(
                    ka["x_scaled"].values, ka["phi"].values, ga["phi"].values
                )
                rr["feature"] = name
                rr["region"] = reg
                rr["species_id"] = sp
                agr_rows.append(rr)
            pd.DataFrame(agr_rows).to_csv(
                sp_dir / "kan_vs_gam_agreement.csv", index=False
            )
            meta["sample_reason"] = r.get("sample_reason", "")
            meta["auc_phase3"] = float(r.get("auc_roc", float("nan")))
            meta["mean_kan_gam_r"] = float(
                np.nanmean([a["pearson_r"] for a in agr_rows])
            )
            (sp_dir / "fit_meta.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )
            rows_meta.append(meta)
            print(
                f"    ok t={meta['runtime_s']:.1f}s mean r(KAN,GAM)={meta['mean_kan_gam_r']:.4f}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"    FAIL {e}", flush=True)
            rows_meta.append(
                {"region": reg, "species_id": sp, "error": str(e)[:300]}
            )
    pd.DataFrame(rows_meta).to_csv(out / "nceas_fit_meta.csv", index=False)
    print("Done NCEAS →", out)


def run_plot(args) -> None:
    """Simple multi-panel figures from exported CSVs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.outdir) if args.outdir else _outdir(
        f"phase5_curves_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    )
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ginkgo: pick bio13, bio11, bio6 if present
    gkan = out / "ginkgo" / "kan_ipp"
    ggam = out / "ginkgo" / "gam_ipp"
    if gkan.is_dir():
        feats = []
        for cand in ["bio13", "bio11", "bio6", "bio12", "bio5"]:
            if (gkan / f"phi_{cand}.csv").is_file():
                feats.append(cand)
        if not feats:
            feats = [p.stem.replace("phi_", "") for p in sorted(gkan.glob("phi_*.csv"))[:3]]
        n = len(feats)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
        for i, name in enumerate(feats):
            ax = axes[0, i]
            ka = pd.read_csv(gkan / f"phi_{name}.csv")
            x = ka["x_raw"] if "x_raw" in ka.columns else ka["x_scaled"]
            ax.plot(x, ka["phi"], label="KAN-IPP", color="C0", lw=2)
            if ggam.is_dir() and (ggam / f"phi_{name}.csv").is_file():
                ga = pd.read_csv(ggam / f"phi_{name}.csv")
                xg = ga["x_raw"] if "x_raw" in ga.columns else ga["x_scaled"]
                ax.plot(xg, ga["phi"], label="GAM-IPP", color="C1", ls="--", lw=1.8)
            if "in_train_support" in ka.columns:
                # light shade outside support
                pass
            ax.set_title(name)
            ax.set_xlabel("covariate")
            ax.set_ylabel(r"$\phi$ (centered)")
            ax.legend(fontsize=8)
            ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        fig.suptitle("Ginkgo biloba — additive response curves", y=1.02)
        fig.tight_layout()
        fig.savefig(fig_dir / "FigG1_ginkgo_curves.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / "FigG1_ginkgo_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Wrote FigG1", fig_dir)

    # NCEAS: one panel per region — first species, first continuous var; KAN + GAM
    nroot = out / "nceas"
    if nroot.is_dir():
        regs = sorted([p.name for p in nroot.iterdir() if p.is_dir()])
        if regs:
            fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
            axes = axes.ravel()
            for i, reg in enumerate(regs[:6]):
                ax = axes[i]
                spp = sorted([p for p in (nroot / reg).iterdir() if p.is_dir()])
                if not spp:
                    ax.set_visible(False)
                    continue
                sp = spp[0]
                kdir = sp / "kan_ipp"
                gdir = sp / "gam_ipp"
                phis = sorted(kdir.glob("phi_*.csv")) if kdir.is_dir() else []
                if not phis:
                    ax.set_title(f"{reg}/no data")
                    continue
                feat = phis[0].stem.replace("phi_", "")
                dfp = pd.read_csv(phis[0])
                x = dfp["x_raw"] if "x_raw" in dfp.columns else dfp["x_scaled"]
                ax.plot(x, dfp["phi"], color="C0", lw=2, label="KAN-IPP")
                gpath = gdir / phis[0].name
                if gpath.is_file():
                    ga = pd.read_csv(gpath)
                    xg = ga["x_raw"] if "x_raw" in ga.columns else ga["x_scaled"]
                    ax.plot(xg, ga["phi"], color="C1", ls="--", lw=1.8, label="GAM-IPP")
                ax.set_title(f"{reg}/{sp.name}\n{feat}")
                ax.axhline(0, color="k", lw=0.4, alpha=0.4)
                ax.legend(fontsize=7)
            fig.suptitle("NCEAS sample — example φ curves (KAN vs GAM)")
            fig.tight_layout()
            fig.savefig(fig_dir / "FigN1_nceas_sample_curves.pdf", bbox_inches="tight")
            fig.savefig(fig_dir / "FigN1_nceas_sample_curves.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("Wrote FigN1", fig_dir)


def run_enhance(args) -> None:
    """Re-export NCEAS GAM curves if missing, build agreement table, bootstrap CI subset."""
    out = _outdir(args.outdir or "phase5_curves_20260715")
    pol = parse_device_policy(args.device, prefer_cpu_lbfgs=not args.allow_gpu_lbfgs)
    B = int(args.bootstrap)
    n_boot_spp = int(args.boot_species)

    # --- re-fit/export any species missing gam_ipp ---
    nroot = out / "nceas"
    species_dirs = sorted([p for p in nroot.glob("*/*") if p.is_dir()]) if nroot.is_dir() else []
    print(f"Enhance: {len(species_dirs)} NCEAS species dirs", flush=True)
    agr_all = []
    for sp_dir in species_dirs:
        reg, sp = sp_dir.parent.name, sp_dir.name
        need = not (sp_dir / "gam_ipp").is_dir() or not list((sp_dir / "gam_ipp").glob("phi_*.csv"))
        if need:
            print(f"  re-export GAM {reg}/{sp} ...", flush=True)
            model, pack, meta, _ = fit_nceas_species(
                reg, sp, seed=args.seed, device_policy=pol, lbfgs_steps=args.lbfgs_steps
            )
            export_additive_curves(
                model,
                pack["env_cont"],
                pack["Xc_tr"],
                sp_dir / "kan_ipp",
                scaler_mean=meta.get("scaler_mean"),
                scaler_scale=meta.get("scaler_scale"),
            )
            export_additive_curves(
                pack["gam_model"],
                pack["env_cont"],
                pack["Xc_tr"],
                sp_dir / "gam_ipp",
                scaler_mean=meta.get("scaler_mean"),
                scaler_scale=meta.get("scaler_scale"),
            )
            meta["sample_reason"] = meta.get("sample_reason", "reexport")
            (sp_dir / "fit_meta.json").write_text(
                json.dumps(meta, indent=2, default=str), encoding="utf-8"
            )
        # agreement
        for fp in sorted((sp_dir / "kan_ipp").glob("phi_*.csv")):
            name = fp.stem.replace("phi_", "")
            gfp = sp_dir / "gam_ipp" / fp.name
            if not gfp.is_file():
                continue
            ka = pd.read_csv(fp)
            ga = pd.read_csv(gfp)
            rr = curve_peak_agreement(
                ka["x_scaled"].values, ka["phi"].values, ga["phi"].values
            )
            rr.update({"region": reg, "species_id": sp, "feature": name})
            agr_all.append(rr)
            # shape metrics from kan
            from kanmaxent.interpret.shape_metrics import shape_metrics

            sm = shape_metrics(ka["x_scaled"].values, ka["phi"].values)
            sm.update({"region": reg, "species_id": sp, "feature": name, "model": "kan_ipp"})
            agr_all[-1].update({f"kan_{k}": v for k, v in sm.items() if k not in ("n",)})

    agr_df = pd.DataFrame(agr_all)
    agr_df.to_csv(out / "nceas_kan_vs_gam_agreement_all.csv", index=False)
    if len(agr_df):
        summary = (
            agr_df.groupby("region")["pearson_r"]
            .agg(["count", "mean", "median", "min"])
            .reset_index()
        )
        summary.to_csv(out / "nceas_kan_vs_gam_agreement_by_region.csv", index=False)
        print(
            f"Agreement: global mean r={agr_df['pearson_r'].mean():.4f} "
            f"median={agr_df['pearson_r'].median():.4f} n={len(agr_df)}",
            flush=True,
        )

    # --- bootstrap CI: Ginkgo + first n_boot_spp NCEAS species ---
    boot_root = out / "bootstrap"
    boot_root.mkdir(exist_ok=True)

    # Ginkgo B
    print(f"Bootstrap Ginkgo B={B} ...", flush=True)
    t0 = time.perf_counter()
    df = load_ginkgo()
    env = env_columns(df)
    y = df["label"].to_numpy(dtype=np.float64)
    X_raw = df[env].to_numpy(dtype=np.float64)
    prep0 = FoldPreprocessor().fit(X_raw)
    X0 = prep0.transform(X_raw)
    model0, _ = fit_ipp_kan(X0, y, X0, seed=0, steps=20, lambda_s=1e-2)
    # fixed grid from full fit
    from kanmaxent.interpret.curves import component_phi_on_grid

    rng = np.random.default_rng(0)
    boot_phi = {n: [] for n in env}
    for b in range(B):
        idx = rng.integers(0, len(y), size=len(y))
        prep = FoldPreprocessor().fit(X_raw[idx])
        Xs = prep.transform(X_raw[idx])
        ys = y[idx]
        mb, _ = fit_ipp_kan(Xs, ys, Xs, seed=b, steps=12, lambda_s=1e-2)
        Xref = torch.as_tensor(Xs, dtype=torch.float64)
        for p, name in enumerate(env):
            _, phi_b = component_phi_on_grid(mb, p, n_grid=80, X_ref=Xref)
            boot_phi[name].append(phi_b)
        if (b + 1) % 10 == 0:
            print(f"  ginkgo boot {b+1}/{B}", flush=True)
    gdir = boot_root / "ginkgo"
    gdir.mkdir(exist_ok=True)
    Xref0 = torch.as_tensor(X0, dtype=torch.float64)
    for p, name in enumerate(env):
        arr = np.stack(boot_phi[name], axis=0)
        lo, mid, hi = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
        xs, phi_point = component_phi_on_grid(model0, p, n_grid=80, X_ref=Xref0)
        x_raw = xs * prep0.scale_[p] + prep0.mean_[p]
        out_arr = np.column_stack([xs, x_raw, phi_point, lo, mid, hi])
        np.savetxt(
            gdir / f"phi_{name}_boot.csv",
            out_arr,
            delimiter=",",
            header="x_scaled,x_raw,phi,phi_lo,phi_med,phi_hi",
            comments="",
        )
    print(f"  Ginkgo bootstrap done in {time.perf_counter()-t0:.1f}s", flush=True)

    # NCEAS subset
    man = out / "species_manifest.csv"
    if man.is_file() and n_boot_spp > 0:
        sample = pd.read_csv(man).head(n_boot_spp)
        print(f"Bootstrap NCEAS n={len(sample)} spp B={B} ...", flush=True)
        for _, r in sample.iterrows():
            reg, sp = r["region"], r["species_id"]
            print(f"  boot {reg}/{sp} ...", flush=True)
            tb = time.perf_counter()
            model, pack, meta, _ = fit_nceas_species(
                reg, sp, seed=args.seed, device_policy=pol, lbfgs_steps=12
            )
            Xc = pack["Xc_tr"]
            Xk = pack["Xk_tr"]
            ytr = pack["y_tr"]
            env_cont = pack["env_cont"]
            # point curves already exist; bootstrap continuous phi
            boot_c = {n: [] for n in env_cont}
            rng = np.random.default_rng(abs(hash(sp)) % (2**31))
            for b in range(B):
                idx = rng.integers(0, len(ytr), size=len(ytr))
                if Xk is not None and getattr(Xk, "ndim", 0) == 2 and Xk.shape[1] > 0:
                    Xk_b = Xk[idx]
                else:
                    Xk_b = np.zeros((len(idx), 0), dtype=np.float64)
                mb, _, _ = fit_hybrid_ipp(
                    Xc[idx],
                    Xk_b,
                    ytr[idx],
                    Xc[idx],
                    Xk_b,
                    seed=b,
                    steps=10,
                )
                Xref = torch.as_tensor(Xc[idx], dtype=torch.float64)
                for p, name in enumerate(env_cont):
                    _, phi_b = component_phi_on_grid(mb.spline, p, n_grid=60, X_ref=Xref)
                    boot_c[name].append(phi_b)
            bdir = boot_root / "nceas" / reg / sp
            bdir.mkdir(parents=True, exist_ok=True)
            Xref0 = torch.as_tensor(Xc, dtype=torch.float64)
            for p, name in enumerate(env_cont):
                arr = np.stack(boot_c[name], axis=0)
                lo, mid, hi = np.quantile(arr, [0.025, 0.5, 0.975], axis=0)
                xs, phi_point = component_phi_on_grid(model.spline, p, n_grid=60, X_ref=Xref0)
                sm = meta.get("scaler_mean")
                ss = meta.get("scaler_scale")
                if sm is not None and ss is not None:
                    x_raw = xs * ss[p] + sm[p]
                    out_arr = np.column_stack([xs, x_raw, phi_point, lo, mid, hi])
                    hdr = "x_scaled,x_raw,phi,phi_lo,phi_med,phi_hi"
                else:
                    out_arr = np.column_stack([xs, phi_point, lo, mid, hi])
                    hdr = "x_scaled,phi,phi_lo,phi_med,phi_hi"
                np.savetxt(
                    bdir / f"phi_{name}_boot.csv",
                    out_arr,
                    delimiter=",",
                    header=hdr,
                    comments="",
                )
            print(f"    done in {time.perf_counter()-tb:.1f}s", flush=True)

    # replot with GAM
    class A:
        outdir = str(out)
        device = args.device
        allow_gpu_lbfgs = args.allow_gpu_lbfgs

    run_plot(A())
    # SUMMARY snippet
    lines = [
        "# Phase 5 curves — quality enhance",
        "",
        f"- NCEAS agreement rows: {len(agr_df)}",
        f"- mean pearson r(KAN,GAM): {agr_df['pearson_r'].mean():.4f}" if len(agr_df) else "- no agr",
        f"- bootstrap B={B}, ginkgo=yes, nceas_spp={n_boot_spp}",
        f"- FigN1 now overlays KAN + GAM when gam_ipp present",
        "",
    ]
    (out / "ENHANCE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", out / "ENHANCE_SUMMARY.md")


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 5 response curves")
    p.add_argument(
        "--mode",
        choices=["pilot", "ginkgo", "nceas", "plot", "enhance", "all"],
        default="pilot",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--allow-gpu-lbfgs", action="store_true", help="do NOT force CPU for LBFGS")
    p.add_argument("--outdir", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pilot-boot", type=int, default=10, help="bootstrap count for pilot")
    p.add_argument("--bootstrap", type=int, default=50, help="B for enhance bootstrap CI")
    p.add_argument(
        "--boot-species",
        type=int,
        default=6,
        help="NCEAS species count for bootstrap CI in enhance mode",
    )
    p.add_argument("--lbfgs-steps", type=int, default=15)
    p.add_argument("--species", default="", help="comma species filter for nceas")
    args = p.parse_args()

    if args.mode in ("pilot", "all"):
        run_pilot(args)
    if args.mode in ("ginkgo", "all"):
        run_ginkgo(args)
    if args.mode in ("nceas", "all"):
        run_nceas(args)
    if args.mode in ("enhance",):
        run_enhance(args)
    if args.mode in ("plot", "all"):
        if not args.outdir and args.mode == "all":
            cands = sorted((ROOT / "outputs").glob("phase5_curves_*"))
            if cands:
                args.outdir = str(cands[-1])
        run_plot(args)


if __name__ == "__main__":
    main()
