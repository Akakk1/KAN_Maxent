"""Profile Poisson / weighted intensity reference for IPP equivalence tests.

On a shared discrete grid, the profile Poisson negative log-likelihood
(without intercept penalty) yields the same relative intensity as conditional
IPP after profiling the global intercept (v6.1 appendix A).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize


def profile_poisson_nll(
    beta: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    lambda_s: float = 0.0,
    lambda_r: float = 0.0,
    D_blocks: Optional[list] = None,
    n_basis_per_feature: int = 0,
    n_features: int = 0,
) -> float:
    """L_P = sum_j w_j exp(eta_j) - sum_j c_j eta_j + Omega (no intercept)."""
    eta = X @ beta
    # numerically stable: max-shift for exp sum
    m = float(np.max(eta)) if eta.size else 0.0
    integ = float(np.sum(weights * np.exp(eta - m)) * np.exp(m))
    lin = float(np.dot(counts, eta))
    loss = integ - lin
    if lambda_r > 0:
        loss += lambda_r * float(np.dot(beta, beta))
    if lambda_s > 0 and D_blocks is not None:
        for p in range(n_features):
            sl = slice(p * n_basis_per_feature, (p + 1) * n_basis_per_feature)
            c = beta[sl]
            D = D_blocks[p]
            if D.size:
                Dc = D @ c
                loss += lambda_s * float(np.dot(Dc, Dc))
    return loss


def profile_poisson_grad(
    beta: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    lambda_s: float = 0.0,
    lambda_r: float = 0.0,
    D_blocks: Optional[list] = None,
    n_basis_per_feature: int = 0,
    n_features: int = 0,
) -> np.ndarray:
    eta = X @ beta
    m = float(np.max(eta)) if eta.size else 0.0
    e = np.exp(eta - m)
    # d/d beta [sum w exp(eta)] = X.T @ (w * exp(eta))
    wexp = weights * e * np.exp(m)
    g = X.T @ wexp - X.T @ counts
    if lambda_r > 0:
        g = g + 2.0 * lambda_r * beta
    if lambda_s > 0 and D_blocks is not None:
        for p in range(n_features):
            sl = slice(p * n_basis_per_feature, (p + 1) * n_basis_per_feature)
            c = beta[sl]
            D = D_blocks[p]
            if D.size:
                # d/dc ||Dc||^2 = 2 D.T D c
                g[sl] = g[sl] + 2.0 * lambda_s * (D.T @ (D @ c))
    return g


def fit_profile_poisson(
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    *,
    lambda_s: float = 0.0,
    lambda_r: float = 0.0,
    D_blocks: Optional[list] = None,
    n_basis_per_feature: int = 0,
    n_features: int = 0,
    beta0: Optional[np.ndarray] = None,
    maxiter: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit eta = X beta by minimizing profile Poisson NLL.

    Returns
    -------
    beta, eta, nll
    """
    X = np.asarray(X, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    p = X.shape[1]
    if beta0 is None:
        beta0 = np.zeros(p, dtype=np.float64)

    def fun(b):
        return profile_poisson_nll(
            b, X, counts, weights, lambda_s, lambda_r, D_blocks, n_basis_per_feature, n_features
        )

    def jac(b):
        return profile_poisson_grad(
            b, X, counts, weights, lambda_s, lambda_r, D_blocks, n_basis_per_feature, n_features
        )

    res = minimize(
        fun,
        beta0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": maxiter, "ftol": 1e-12},
    )
    beta = res.x
    eta = X @ beta
    # Center eta (remove unidentifiable shift for comparison with conditional IPP)
    # Under conditional IPP, only relative eta matters; for intensity comparison
    # we compare normalized densities.
    return beta, eta, float(res.fun)


def normalized_relative_intensity(eta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """p_j = w_j exp(eta_j) / sum_k w_k exp(eta_k)."""
    eta = np.asarray(eta, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    m = float(np.max(eta))
    unnorm = weights * np.exp(eta - m)
    return unnorm / unnorm.sum()
