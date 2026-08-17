#!/usr/bin/env python
"""Compare our maxnet AUCs to Valavi OSF MaxNet predictions (multi-region)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_pa_for_species(region: str, species_id: str, group: str | None = None) -> pd.DataFrame:
    from kanmaxent.data.nceas_io import load_region, species_group

    data = load_region(region)
    if group is None:
        group = species_group(data.po, species_id)
    if data.pa_by_group:
        return data.pa_by_group[group]
    assert data.pa is not None
    return data.pa


def valavi_auc(pa: pd.DataFrame, pred_path: Path, species_id: str) -> dict:
    pred = pd.read_csv(pred_path)
    m = pa[["siteid", species_id]].merge(
        pred[["siteid", "prediction"]], on="siteid", how="inner"
    )
    y = m[species_id].to_numpy(dtype=float)
    s = m["prediction"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2 or len(m) == 0:
        return {
            "species_id": species_id,
            "valavi_auc": float("nan"),
            "n_eval": len(m),
        }
    return {
        "species_id": species_id,
        "valavi_auc": float(roc_auc_score(y, s)),
        "valavi_auprc": float(average_precision_score(y, s)),
        "n_eval": len(m),
        "n_pos": int(y.sum()),
    }


def compare_region(
    region: str,
    our_metrics: pd.DataFrame,
    pred_dir: Path,
    outdir: Path,
) -> dict:
    from kanmaxent.data.nceas_io import load_region, species_group

    data = load_region(region)
    our = our_metrics.loc[
        (our_metrics["region"] == region) & (our_metrics["model"] == "maxnet"),
        ["species_id", "auc_roc"],
    ].rename(columns={"auc_roc": "ours_auc"})
    rows = []
    missing = []
    for sp in sorted(our["species_id"].unique()):
        f = pred_dir / f"{sp}_maxnet.csv"
        if not f.is_file():
            missing.append(sp)
            rows.append({"species_id": sp, "valavi_auc": float("nan")})
            continue
        g = species_group(data.po, sp)
        pa = data.pa_by_group[g] if data.pa_by_group else data.pa
        assert pa is not None
        rows.append(valavi_auc(pa, f, sp))
    val = pd.DataFrame(rows)
    cmp = our.merge(val, on="species_id")
    cmp["region"] = region
    cmp["delta_auc"] = cmp["ours_auc"] - cmp["valavi_auc"]
    mask = cmp["ours_auc"].notna() & cmp["valavi_auc"].notna()
    if mask.sum() >= 3:
        r_p, p_p = pearsonr(cmp.loc[mask, "ours_auc"], cmp.loc[mask, "valavi_auc"])
        r_s, p_s = spearmanr(cmp.loc[mask, "ours_auc"], cmp.loc[mask, "valavi_auc"])
    else:
        r_p = p_p = r_s = p_s = float("nan")
    summary = {
        "region": region,
        "n_species_compared": int(mask.sum()),
        "n_missing_pred_files": len(missing),
        "missing": missing,
        "pearson_r": float(r_p) if np.isfinite(r_p) else None,
        "pearson_p": float(p_p) if np.isfinite(p_p) else None,
        "spearman_rho": float(r_s) if np.isfinite(r_s) else None,
        "valavi_mean_auc": float(cmp.loc[mask, "valavi_auc"].mean()) if mask.any() else None,
        "ours_mean_auc": float(cmp.loc[mask, "ours_auc"].mean()) if mask.any() else None,
        "mean_abs_delta": float(cmp.loc[mask, "delta_auc"].abs().mean()) if mask.any() else None,
        "dod_r_gt_0.90": bool(np.isfinite(r_p) and r_p > 0.90),
    }
    cmp.to_csv(outdir / f"valavi_maxnet_comparison_{region}.csv", index=False)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--our-metrics",
        default=str(ROOT / "outputs/nceas_full_20260714_phase3/metrics_all.csv"),
    )
    p.add_argument(
        "--valavi-maxnet-dir",
        default=str(ROOT / "data/valavi_osf/Models_prediction/MaxNet"),
    )
    p.add_argument(
        "--outdir",
        default=str(ROOT / "outputs/nceas_full_20260714_phase3"),
    )
    p.add_argument("--regions", default="AWT,CAN,NSW,NZ,SA,SWI")
    args = p.parse_args()

    our = pd.read_csv(args.our_metrics)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = Path(args.valavi_maxnet_dir)
    regions = [r.strip().upper() for r in args.regions.split(",") if r.strip()]
    summaries = []
    all_cmp = []
    for reg in regions:
        if reg not in our["region"].unique():
            print(f"skip {reg}: not in metrics")
            continue
        s = compare_region(reg, our, pred_dir, outdir)
        summaries.append(s)
        print(reg, "r=", s.get("pearson_r"), "n=", s.get("n_species_compared"), "pass=", s.get("dod_r_gt_0.90"))
        f = outdir / f"valavi_maxnet_comparison_{reg}.csv"
        if f.is_file():
            all_cmp.append(pd.read_csv(f))
    if all_cmp:
        pd.concat(all_cmp, ignore_index=True).to_csv(
            outdir / "valavi_maxnet_comparison_all.csv", index=False
        )
    (outdir / "valavi_comparison_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
