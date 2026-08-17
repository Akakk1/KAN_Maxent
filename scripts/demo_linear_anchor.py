#!/usr/bin/env python
"""Minimal demo: linear-degeneration KAN-IPP vs profile Poisson (E.3 metrics)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kanmaxent.data.synthetic import make_linear_scenario
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.bspline import second_difference_penalty_matrix
from kanmaxent.reference.poisson_glm import (
    fit_profile_poisson,
    normalized_relative_intensity,
)
from kanmaxent.train_utils import fit_additive_kan_ipp
from kanmaxent.utils import as_numpy_f64


def main() -> None:
    data = make_linear_scenario(n_points=400, n_features=2, n_presence=80, seed=0)
    X_np = as_numpy_f64(data["X"])
    counts_np = as_numpy_f64(data["counts"])
    w_np = as_numpy_f64(data["weights"])

    model = AdditiveSplineKAN(
        n_features=2, linear_mode=True, lambda_s=0.0, lambda_r=1e-8, dtype=torch.float64
    )
    model.fit_bounds(X_np)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    counts = torch.as_tensor(counts_np, dtype=torch.float64)
    weights = torch.as_tensor(w_np, dtype=torch.float64)
    fit_additive_kan_ipp(model, X, counts, weights, steps=40, lr=0.5)

    with torch.no_grad():
        eta_kan = as_numpy_f64(model(X, center=True, center_weights=weights))

    X_design = as_numpy_f64(model.full_design_matrix(X))
    D_blocks = [
        second_difference_penalty_matrix(model.n_basis_per_feature)
        for _ in range(model.n_features)
    ]
    _, eta_pois, _ = fit_profile_poisson(
        X_design,
        counts_np,
        w_np,
        lambda_s=0.0,
        lambda_r=1e-8,
        D_blocks=D_blocks,
        n_basis_per_feature=model.n_basis_per_feature,
        n_features=model.n_features,
    )
    eta_pois = as_numpy_f64(eta_pois)

    d_k = as_numpy_f64(normalized_relative_intensity(eta_kan, w_np))
    d_p = as_numpy_f64(normalized_relative_intensity(eta_pois, w_np))
    rho, _ = spearmanr(d_k, d_p)
    mask = d_p > float(np.max(d_p)) * 1e-6
    max_rel = float(np.max(np.abs(d_k[mask] - d_p[mask]) / d_p[mask]))

    print("=== Linear degeneration trust anchor (E.3) ===")
    print(f"Spearman rho (KAN-IPP vs profile Poisson densities): {rho:.6f}")
    print(f"Max relative density error: {max_rel:.4%}")
    print("PASS" if float(rho) > 0.99 and max_rel < 0.01 else "FAIL")


if __name__ == "__main__":
    main()
