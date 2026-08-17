#!/usr/bin/env python
"""Merge per-region deepkan fair runs and write paired multi-seed SUMMARY."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(np.mean(x)), float(lo), float(hi)


def paired_from_metrics(m: pd.DataFrame) -> pd.DataFrame:
    a = m[m.model == "additive_kan_ipp"].set_index(["region", "species_id", "seed"])[
        "auc_roc"
    ]
    d = m[m.model == "deep2_kan_ipp"].set_index(["region", "species_id", "seed"])[
        "auc_roc"
    ]
    common = a.index.intersection(d.index)
    out = pd.DataFrame(
        {
            "auc_additive": a.loc[common].values,
            "auc_deep2": d.loc[common].values,
            "delta_auc": (d.loc[common] - a.loc[common]).values,
        },
        index=common,
    ).reset_index()
    return out[np.isfinite(out["delta_auc"])]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base",
        default="outputs/deepkan_fair_six_3seed_20260715",
        help="base dir containing by_region/{REG}/metrics.csv",
    )
    args = p.parse_args()
    base = Path(args.base)
    frames = []
    for reg_dir in sorted((base / "by_region").glob("*")):
        mp = reg_dir / "metrics.csv"
        if mp.is_file():
            df = pd.read_csv(mp)
            frames.append(df)
            print(f"  {reg_dir.name}: {len(df)} rows")
    if not frames:
        raise SystemExit("no metrics found")
    m = pd.concat(frames, ignore_index=True)
    m.to_csv(base / "metrics_all.csv", index=False)

    paired = paired_from_metrics(m)
    paired.to_csv(base / "paired_delta_auc_all.csv", index=False)

    # species-level mean across seeds
    sp = (
        paired.groupby(["region", "species_id"], as_index=False)
        .agg(
            n_seed=("seed", "nunique"),
            auc_additive=("auc_additive", "mean"),
            auc_deep2=("auc_deep2", "mean"),
            delta_auc=("delta_auc", "mean"),
            delta_sd=("delta_auc", "std"),
        )
    )
    sp.to_csv(base / "paired_delta_auc_species_mean.csv", index=False)

    mean, lo, hi = bootstrap_mean_ci(sp["delta_auc"].values)
    n_pos = int((sp["delta_auc"] > 0.01).sum())
    n_neg = int((sp["delta_auc"] < -0.01).sum())

    lines = [
        "# Phase 4 Fair Deep-2 — Six-region multi-seed SUMMARY",
        "",
        f"**Run:** `{base.name}`",
        "**Protocol:** residual + warm-start additive + LBFGS + disable SiLU",
        f"**Scale:** {m['region'].nunique()} regions, "
        f"{sp['species_id'].nunique()} species, "
        f"seeds={sorted(m['seed'].unique().tolist())}, "
        f"{len(m)} metric rows, {len(sp)} species-level paired means",
        "",
        "## Global (species-level mean over seeds)",
        "",
        f"- mean ΔAUC = **{mean:+.4f}** [{lo:+.4f}, {hi:+.4f}]",
        f"- median Δ = {sp['delta_auc'].median():+.4f}",
        f"- n(Δ>0.01) = {n_pos}/{len(sp)}; n(Δ<−0.01) = {n_neg}/{len(sp)}",
        "",
        "## By region (species means)",
        "",
        "| region | n_sp | mean Δ | CI_lo | CI_hi | n(Δ>0.01) | mean seeds |",
        "|:-------|-----:|-------:|------:|------:|----------:|-----------:|",
    ]
    for reg, sub in sp.groupby("region"):
        mm, ll, hh = bootstrap_mean_ci(sub["delta_auc"].values)
        lines.append(
            f"| {reg} | {len(sub)} | {mm:+.4f} | {ll:+.4f} | {hh:+.4f} | "
            f"{int((sub.delta_auc > 0.01).sum())} | {sub.n_seed.mean():.2f} |"
        )

    # seed stability: within-species SD of delta
    if paired["seed"].nunique() > 1:
        seed_sd = paired.groupby(["region", "species_id"])["delta_auc"].std()
        lines += [
            "",
            "## Multi-seed stability",
            "",
            f"- mean within-species SD(Δ) = {seed_sd.mean():.4f}",
            f"- median within-species SD(Δ) = {seed_sd.median():.4f}",
        ]

    lines += [
        "",
        "## Mean AUC by model (all rows)",
        "",
    ]
    for model, sub in m.groupby("model"):
        auc = sub["auc_roc"].dropna()
        lines.append(f"- **{model}**: {auc.mean():.4f} ± {auc.std():.4f} (n={len(auc)})")

    lines += [
        "",
        "## Decision",
        "",
    ]
    if np.isfinite(lo) and lo > 0:
        lines.append("- Global CI above 0 → positive interaction signal.")
    elif np.isfinite(hi) and hi < 0:
        lines.append("- Global CI below 0 → deep systematically worse (unlikely after fair).")
    else:
        lines.append(
            "- **No systematic gain:** global CI includes 0 (or mean ≈ 0). "
            "Supports additivity as a testable hypothesis under fair IPP + residual Deep-2."
        )
    lines.append("")
    lines.append("*Fair protocol: residual / warm-start / LBFGS / no-SiLU.*")

    (base / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "run_id": base.name,
        "protocol": "fair_residual_lbfgs_nosilu_warmstart",
        "n_rows": len(m),
        "n_species": int(sp["species_id"].nunique()),
        "regions": sorted(m["region"].unique().tolist()),
        "seeds": sorted(int(s) for s in m["seed"].unique()),
        "mean_delta_species": mean,
        "ci": [lo, hi],
        "n_delta_gt_0.01": n_pos,
        "created_unix": time.time(),
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n".join(lines[:25]))
    print("Wrote", base / "SUMMARY.md")


if __name__ == "__main__":
    main()
