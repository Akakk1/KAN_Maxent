#!/usr/bin/env python
"""CLI: Phase 1 Ginkgo 5-fold spatial CV full contrast."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.evaluation.spatial_cv import run_ginkgo_cv


def main() -> None:
    p = argparse.ArgumentParser(description="Ginkgo readonly spatial CV (Phase 1)")
    p.add_argument(
        "--data",
        default=str(ROOT / "data/ginkgo/ginkgo_training_with_coords.csv"),
        help="Path to Ginkgo CSV (read-only copy)",
    )
    p.add_argument(
        "--outdir",
        default="",
        help="Output directory (must not exist or be empty). Default: outputs/ginkgo_cv_<timestamp>",
    )
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    p.add_argument("--folds", default="", help="Comma-separated folds; default all")
    p.add_argument("--lambda-s", type=float, default=1e-2)
    p.add_argument("--lambda-r", type=float, default=1e-6)
    p.add_argument("--n-intervals", type=int, default=6)
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--lbfgs-steps", type=int, default=35)
    p.add_argument(
        "--models",
        default="kan_ipp,gam_ipp,kan_bce,maxnet",
        help="Comma-separated: kan_ipp,gam_ipp,kan_bce,maxnet",
    )
    p.add_argument("--no-curves", action="store_true")
    p.add_argument("--skip-sha", action="store_true")
    args = p.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    folds = (
        tuple(int(x) for x in args.folds.split(",") if x.strip() != "")
        if args.folds.strip()
        else None
    )
    models = tuple(x.strip() for x in args.models.split(",") if x.strip())

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = ROOT / "outputs" / f"ginkgo_cv_{ts}"

    print(f"Data: {args.data}")
    print(f"Out:  {outdir}")
    print(f"Seeds={seeds} models={models} lambda_s={args.lambda_s}")

    metrics = run_ginkgo_cv(
        data_path=args.data,
        outdir=outdir,
        seeds=seeds,
        folds=folds,
        n_intervals=args.n_intervals,
        degree=args.degree,
        lambda_s=args.lambda_s,
        lambda_r=args.lambda_r,
        lbfgs_steps=args.lbfgs_steps,
        models=models,
        export_curves=not args.no_curves,
        check_sha256=not args.skip_sha,
    )
    print(metrics.groupby("model")["auc_roc"].agg(["mean", "std", "count"]))
    print(f"Wrote {outdir / 'metrics.csv'}")
    print(f"Summary: {outdir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
