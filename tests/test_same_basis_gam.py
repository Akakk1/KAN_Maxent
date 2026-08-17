"""E.4 same-basis KAN vs GAM; E.6 penalty parity."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from kanmaxent.data.synthetic import make_nonlinear_scenario
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.bspline import second_difference_penalty_matrix
from kanmaxent.reference.gam_ipp import fit_gam_ipp
from kanmaxent.reference.poisson_glm import normalized_relative_intensity
from kanmaxent.train_utils import fit_additive_kan_ipp
from kanmaxent.utils import as_numpy_f64


def _fit_pair(lambda_s: float, seed: int = 1):
    data = make_nonlinear_scenario(n_points=400, n_features=2, n_presence=90, seed=seed)
    X_np = as_numpy_f64(data["X"])
    counts_np = as_numpy_f64(data["counts"])
    w_np = as_numpy_f64(data["weights"])

    model = AdditiveSplineKAN(
        n_features=2,
        n_intervals=6,
        degree=3,
        lambda_s=lambda_s,
        lambda_r=1e-6,
        dtype=torch.float64,
    )
    model.fit_bounds(X_np)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    counts = torch.as_tensor(counts_np, dtype=torch.float64)
    weights = torch.as_tensor(w_np, dtype=torch.float64)

    fit_additive_kan_ipp(model, X, counts, weights, steps=80, lr=0.25)

    with torch.no_grad():
        eta_kan = as_numpy_f64(model(X, center=True, center_weights=weights))
        coeffs_kan = as_numpy_f64(model.coeffs).ravel()

    X_design = as_numpy_f64(model.full_design_matrix(X))
    D_blocks = [
        second_difference_penalty_matrix(model.n_basis_per_feature)
        for _ in range(model.n_features)
    ]
    beta_gam, eta_gam, _ = fit_gam_ipp(
        X_design,
        counts_np,
        w_np,
        lambda_s=lambda_s,
        lambda_r=1e-6,
        D_blocks=D_blocks,
        n_basis_per_feature=model.n_basis_per_feature,
        n_features=model.n_features,
        maxiter=1000,
    )
    beta_gam = as_numpy_f64(beta_gam)
    eta_gam = as_numpy_f64(eta_gam)

    dens_kan = as_numpy_f64(normalized_relative_intensity(eta_kan, w_np))
    dens_gam = as_numpy_f64(normalized_relative_intensity(eta_gam, w_np))
    return {
        "dens_kan": dens_kan,
        "dens_gam": dens_gam,
        "eta_kan": eta_kan,
        "eta_gam": eta_gam,
        "coeffs_kan": coeffs_kan,
        "beta_gam": beta_gam,
        "lambda_s": lambda_s,
        "n_basis": int(model.n_basis_per_feature),
        "n_features": int(model.n_features),
        "w": w_np,
    }


def test_e4_same_basis_kan_matches_gam():
    """E.4: same design/knots/λ → densities and coeffs agree within tolerance."""
    out = _fit_pair(lambda_s=1e-2, seed=1)
    rho, _ = spearmanr(out["dens_kan"], out["dens_gam"])
    assert float(rho) > 0.99, f"density Spearman ρ={rho:.6f}"

    ek = out["eta_kan"] - float(np.mean(out["eta_kan"]))
    eg = out["eta_gam"] - float(np.mean(out["eta_gam"]))
    corr = float(np.corrcoef(ek, eg)[0, 1])
    assert corr > 0.99, f"eta corr={corr:.6f}"

    ck = out["coeffs_kan"].copy()
    cg = out["beta_gam"].copy()
    n_b = out["n_basis"]
    for p in range(out["n_features"]):
        sl = slice(p * n_b, (p + 1) * n_b)
        ck[sl] = ck[sl] - float(np.mean(ck[sl]))
        cg[sl] = cg[sl] - float(np.mean(cg[sl]))
    coeff_corr = float(np.corrcoef(ck, cg)[0, 1])
    assert coeff_corr > 0.95, f"coeff corr after demean={coeff_corr:.6f}"


def _roughness(coeffs: np.ndarray, n_features: int, n_basis: int) -> float:
    D = np.asarray(second_difference_penalty_matrix(n_basis), dtype=np.float64)
    if D.size == 0:
        return 0.0
    total = 0.0
    c_all = np.asarray(coeffs, dtype=np.float64).ravel()
    for p in range(n_features):
        c = c_all[p * n_basis : (p + 1) * n_basis]
        Dc = D.dot(c)
        total += float(np.dot(Dc, Dc))
    return total


def test_e6_penalty_parity():
    """E.6: increasing λs smooths both KAN and GAM in the same direction/magnitude."""
    low = _fit_pair(lambda_s=1e-4, seed=5)
    high = _fit_pair(lambda_s=1.0, seed=5)

    n_b = low["n_basis"]
    n_f = low["n_features"]

    r_kan_low = _roughness(low["coeffs_kan"], n_f, n_b)
    r_kan_high = _roughness(high["coeffs_kan"], n_f, n_b)
    r_gam_low = _roughness(low["beta_gam"], n_f, n_b)
    r_gam_high = _roughness(high["beta_gam"], n_f, n_b)

    assert r_kan_high <= r_kan_low * 1.05 + 1e-8, (
        f"KAN roughness did not decrease: {r_kan_low} → {r_kan_high}"
    )
    assert r_gam_high <= r_gam_low * 1.05 + 1e-8, (
        f"GAM roughness did not decrease: {r_gam_low} → {r_gam_high}"
    )

    ratio_kan = (r_kan_high + 1e-12) / (r_kan_low + 1e-12)
    ratio_gam = (r_gam_high + 1e-12) / (r_gam_low + 1e-12)
    assert ratio_kan < 0.9 or r_kan_low < 1e-6
    assert ratio_gam < 0.9 or r_gam_low < 1e-6
    if r_kan_low > 1e-6 and r_gam_low > 1e-6:
        log_diff = abs(float(np.log(ratio_kan + 1e-12) - np.log(ratio_gam + 1e-12)))
        assert log_diff < 2.0, f"penalty response mismatch log_diff={log_diff}"
