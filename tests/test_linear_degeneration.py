"""E.3: linear degeneration trust anchor (G=1, K=1) vs profile Poisson."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from kanmaxent.data.synthetic import make_linear_scenario
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.bspline import second_difference_penalty_matrix
from kanmaxent.reference.poisson_glm import (
    fit_profile_poisson,
    normalized_relative_intensity,
)
from kanmaxent.train_utils import fit_additive_kan_ipp
from kanmaxent.utils import as_numpy_f64


def test_e3_linear_degeneration_matches_profile_poisson():
    """E.3: Spearman ρ > 0.99 and max relative error of density < 1%."""
    data = make_linear_scenario(
        n_points=400, n_features=2, n_presence=80, seed=0, uneven=True
    )
    X_np = as_numpy_f64(data["X"])
    counts_np = as_numpy_f64(data["counts"])
    w_np = as_numpy_f64(data["weights"])

    # --- KAN linear mode ---
    model = AdditiveSplineKAN(
        n_features=2,
        linear_mode=True,
        lambda_s=0.0,  # linear: no smoothness DoF (n_basis=2)
        lambda_r=1e-8,
        dtype=torch.float64,
    )
    model.fit_bounds(X_np)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    counts = torch.as_tensor(counts_np, dtype=torch.float64)
    weights = torch.as_tensor(w_np, dtype=torch.float64)

    fit_additive_kan_ipp(model, X, counts, weights, steps=40, lr=0.5)
    with torch.no_grad():
        eta_kan = as_numpy_f64(model(X, center=True, center_weights=weights))

    # --- Profile Poisson on the same design matrix ---
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
        maxiter=800,
    )
    eta_pois = as_numpy_f64(eta_pois)

    dens_kan = as_numpy_f64(normalized_relative_intensity(eta_kan, w_np))
    dens_pois = as_numpy_f64(normalized_relative_intensity(eta_pois, w_np))

    rho, _ = spearmanr(dens_kan, dens_pois)
    mask = dens_pois > float(np.max(dens_pois)) * 1e-6
    rel_err = np.abs(dens_kan[mask] - dens_pois[mask]) / dens_pois[mask]
    max_rel = float(np.max(rel_err)) if rel_err.size else 0.0

    assert float(rho) > 0.99, f"Spearman ρ={rho:.6f} <= 0.99"
    assert max_rel < 0.01, f"max relative density error={max_rel:.4%} >= 1%"
