"""Same-basis GAM/IPP reference: penalized coefficients on shared design matrix.

Uses the conditional IPP objective (not Poisson) with identical second-difference
and ridge penalties as AdditiveSplineKAN, solved by L-BFGS-B.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def ipp_nll_numpy(
    beta: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    lambda_s: float,
    lambda_r: float,
    D_blocks: List[np.ndarray],
    n_basis_per_feature: int,
    n_features: int,
    per_presence: bool = False,
) -> float:
    eta = X @ beta
    # Center components blockwise for parity with model forward(center=True)
    # Here X is already the full design; centering eta is equivalent for the
    # conditional likelihood (global shift cancels). Component-wise centering
    # is applied when building predictions for coefficient comparison.
    n = float(counts.sum())
    lin = float(np.dot(counts, eta))
    log_z = float(logsumexp(eta + np.log(weights)))
    loss = -lin + n * log_z
    if lambda_r > 0:
        loss += lambda_r * float(np.dot(beta, beta))
    if lambda_s > 0:
        for p in range(n_features):
            sl = slice(p * n_basis_per_feature, (p + 1) * n_basis_per_feature)
            c = beta[sl]
            D = D_blocks[p]
            if D.size:
                Dc = D @ c
                loss += lambda_s * float(np.dot(Dc, Dc))
    if per_presence:
        loss = loss / n
    return loss


def ipp_grad_numpy(
    beta: np.ndarray,
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    lambda_s: float,
    lambda_r: float,
    D_blocks: List[np.ndarray],
    n_basis_per_feature: int,
    n_features: int,
    per_presence: bool = False,
) -> np.ndarray:
    eta = X @ beta
    n = float(counts.sum())
    # d log_z / d eta_j = w_j exp(eta_j) / Z
    m = float(np.max(eta + np.log(weights)))
    unnorm = np.exp(eta + np.log(weights) - m)
    Z = unnorm.sum()
    p = unnorm / Z
    # dL/d eta = -c + n * p
    d_eta = -counts + n * p
    g = X.T @ d_eta
    if lambda_r > 0:
        g = g + 2.0 * lambda_r * beta
    if lambda_s > 0:
        for feat in range(n_features):
            sl = slice(feat * n_basis_per_feature, (feat + 1) * n_basis_per_feature)
            c = beta[sl]
            D = D_blocks[feat]
            if D.size:
                g[sl] = g[sl] + 2.0 * lambda_s * (D.T @ (D @ c))
    if per_presence:
        g = g / n
    return g


def fit_gam_ipp(
    X: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
    *,
    lambda_s: float,
    lambda_r: float,
    D_blocks: List[np.ndarray],
    n_basis_per_feature: int,
    n_features: int,
    beta0: Optional[np.ndarray] = None,
    maxiter: int = 800,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit same-basis GAM under IPP + penalties. Returns beta, eta, nll."""
    X = np.asarray(X, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    pdim = X.shape[1]
    if beta0 is None:
        beta0 = np.zeros(pdim, dtype=np.float64)

    def fun(b):
        return ipp_nll_numpy(
            b, X, counts, weights, lambda_s, lambda_r, D_blocks, n_basis_per_feature, n_features
        )

    def jac(b):
        return ipp_grad_numpy(
            b, X, counts, weights, lambda_s, lambda_r, D_blocks, n_basis_per_feature, n_features
        )

    res = minimize(
        fun,
        beta0,
        method="L-BFGS-B",
        jac=jac,
        options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-10},
    )
    beta = res.x
    eta = X @ beta
    return beta, eta, float(res.fun)
