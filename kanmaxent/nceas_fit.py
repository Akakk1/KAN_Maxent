"""NCEAS-protocol fitting: continuous B-spline + categorical linear terms."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import logsumexp

from kanmaxent.losses.bce import bce_with_logits_nll
from kanmaxent.losses.ipp import ipp_nll
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.bspline import second_difference_penalty_matrix
from kanmaxent.utils import as_numpy_f64


class HybridSplineCat(nn.Module):
    """eta = sum_p phi_p(x_p) + X_cat @ beta_cat; bias = 0."""

    def __init__(
        self,
        n_continuous: int,
        n_cat_oh: int,
        *,
        n_intervals: int = 6,
        degree: int = 3,
        lambda_s: float = 1e-2,
        lambda_r: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_continuous = n_continuous
        self.n_cat_oh = n_cat_oh
        self.lambda_s = float(lambda_s)
        self.lambda_r = float(lambda_r)
        self.spline = AdditiveSplineKAN(
            n_features=n_continuous,
            n_intervals=n_intervals,
            degree=degree,
            lambda_s=lambda_s,
            lambda_r=lambda_r,
            dtype=torch.float64,
        )
        if n_cat_oh > 0:
            self.beta_cat = nn.Parameter(torch.zeros(n_cat_oh, dtype=torch.float64))
        else:
            self.register_parameter("beta_cat", None)

    def fit_bounds(self, X_cont: np.ndarray) -> "HybridSplineCat":
        self.spline.fit_bounds(X_cont)
        return self

    def spline_design(self, X_cont: torch.Tensor) -> torch.Tensor:
        return self.spline.full_design_matrix(X_cont)

    def eta_from_parts(
        self,
        B_spline: torch.Tensor,
        X_cat: Optional[torch.Tensor],
        *,
        center_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_b = self.spline.n_basis_per_feature
        P = self.n_continuous
        phis = []
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            phis.append(B_spline[:, sl] @ self.spline.coeffs[p])
        phis_t = torch.stack(phis, dim=1)
        if center_weights is None:
            w = torch.ones(B_spline.shape[0], dtype=B_spline.dtype, device=B_spline.device)
        else:
            w = center_weights
        w = w / w.sum()
        means = (w[:, None] * phis_t).sum(dim=0)
        eta = (phis_t - means).sum(dim=1)
        if self.beta_cat is not None and X_cat is not None and X_cat.shape[1] > 0:
            eta = eta + X_cat @ self.beta_cat
        return eta

    def penalty(self) -> torch.Tensor:
        # smoothness only on spline; ridge on spline + cat
        pen = self.lambda_s * self.spline.smoothness_penalty()
        pen = pen + self.lambda_r * self.spline.ridge_penalty()
        if self.beta_cat is not None:
            pen = pen + self.lambda_r * (self.beta_cat ** 2).sum()
        return pen


def _init_hybrid(model: HybridSplineCat, seed: int) -> None:
    rng = np.random.default_rng(seed)
    model.spline.set_coeffs_from_numpy(
        rng.normal(0, 0.01, size=tuple(model.spline.coeffs.shape))
    )
    if model.beta_cat is not None:
        with torch.no_grad():
            model.beta_cat.copy_(
                torch.as_tensor(rng.normal(0, 0.01, size=(model.n_cat_oh,)), dtype=torch.float64)
            )


def fit_hybrid_ipp(
    X_cont_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    X_cont_test: np.ndarray,
    X_cat_test: np.ndarray,
    *,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    seed: int = 0,
    steps: int = 25,
    lr: float = 0.25,
) -> Tuple[HybridSplineCat, np.ndarray, Dict]:
    import time

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cat = X_cat_train.shape[1] if X_cat_train.ndim == 2 else 0
    model = HybridSplineCat(
        X_cont_train.shape[1],
        n_cat,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
    )
    model.fit_bounds(X_cont_train)
    _init_hybrid(model, seed)

    Xc_tr = torch.as_tensor(X_cont_train, dtype=torch.float64)
    Xc_te = torch.as_tensor(X_cont_test, dtype=torch.float64)
    Xk_tr = torch.as_tensor(X_cat_train, dtype=torch.float64) if n_cat else None
    Xk_te = torch.as_tensor(X_cat_test, dtype=torch.float64) if n_cat else None
    B_tr = model.spline_design(Xc_tr).detach()
    B_te = model.spline_design(Xc_te).detach()
    counts = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(len(y_train), dtype=torch.float64)

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=15,
        line_search_fn="strong_wolfe",
        history_size=30,
        tolerance_grad=1e-8,
    )

    def closure():
        opt.zero_grad()
        eta = model.eta_from_parts(B_tr, Xk_tr, center_weights=weights)
        loss = ipp_nll(eta, counts, weights, model.penalty(), per_presence=False)
        loss.backward()
        return loss

    for _ in range(steps):
        loss = opt.step(closure)
        if not torch.isfinite(loss):
            break

    scores = _predict_test(model, B_tr, B_te, Xk_tr, Xk_te, weights)

    return model, scores, {
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(np.isfinite(scores).all()),
        "lambda_s": lambda_s,
        "B_train": B_tr,
    }



def _predict_test(
    model: HybridSplineCat,
    B_tr: torch.Tensor,
    B_te: torch.Tensor,
    Xk_tr: Optional[torch.Tensor],
    Xk_te: Optional[torch.Tensor],
    w_tr: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        n_b = model.spline.n_basis_per_feature
        P = model.n_continuous
        phis_tr, phis_te = [], []
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            phis_tr.append(B_tr[:, sl] @ model.spline.coeffs[p])
            phis_te.append(B_te[:, sl] @ model.spline.coeffs[p])
        ptr = torch.stack(phis_tr, dim=1)
        pte = torch.stack(phis_te, dim=1)
        ww = w_tr / w_tr.sum()
        means = (ww[:, None] * ptr).sum(dim=0)
        eta = (pte - means).sum(dim=1)
        if model.beta_cat is not None and Xk_te is not None and Xk_te.shape[1] > 0:
            eta = eta + Xk_te @ model.beta_cat
    return as_numpy_f64(eta)


def fit_hybrid_bce(
    X_cont_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    X_cont_test: np.ndarray,
    X_cat_test: np.ndarray,
    *,
    pos_weight: float,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    seed: int = 0,
    steps: int = 25,
    lr: float = 0.25,
) -> Tuple[HybridSplineCat, np.ndarray, Dict]:
    import time

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    n_cat = X_cat_train.shape[1] if X_cat_train.ndim == 2 else 0
    model = HybridSplineCat(
        X_cont_train.shape[1],
        n_cat,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
    )
    model.fit_bounds(X_cont_train)
    _init_hybrid(model, seed)

    Xc_tr = torch.as_tensor(X_cont_train, dtype=torch.float64)
    Xc_te = torch.as_tensor(X_cont_test, dtype=torch.float64)
    Xk_tr = torch.as_tensor(X_cat_train, dtype=torch.float64) if n_cat else None
    Xk_te = torch.as_tensor(X_cat_test, dtype=torch.float64) if n_cat else None
    B_tr = model.spline_design(Xc_tr).detach()
    B_te = model.spline_design(Xc_te).detach()
    y = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(len(y_train), dtype=torch.float64)

    opt = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=15, line_search_fn="strong_wolfe", history_size=30
    )

    def closure():
        opt.zero_grad()
        logits = model.eta_from_parts(B_tr, Xk_tr, center_weights=weights)
        pen = model.penalty() / max(len(y), 1)
        loss = bce_with_logits_nll(logits, y, pen, pos_weight=pos_weight, reduction="mean")
        loss.backward()
        return loss

    for _ in range(steps):
        loss = opt.step(closure)
        if not torch.isfinite(loss):
            break

    scores = _predict_test(model, B_tr, B_te, Xk_tr, Xk_te, weights)
    return model, scores, {
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(np.isfinite(scores).all()),
        "lambda_s": lambda_s,
        "pos_weight": pos_weight,
    }


def fit_hybrid_gam_ipp(
    model: HybridSplineCat,
    X_cont_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    X_cont_test: np.ndarray,
    X_cat_test: np.ndarray,
    *,
    B_train: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, Dict]:
    """Same basis GAM: joint L-BFGS-B on spline+cat coeffs with IPP + penalties."""
    import time

    t0 = time.perf_counter()
    Xc_tr = torch.as_tensor(X_cont_train, dtype=torch.float64)
    Xc_te = torch.as_tensor(X_cont_test, dtype=torch.float64)
    if B_train is None:
        B_tr = as_numpy_f64(model.spline_design(Xc_tr))
    else:
        B_tr = as_numpy_f64(B_train)
    B_te = as_numpy_f64(model.spline_design(Xc_te))
    Xk_tr = np.asarray(X_cat_train, dtype=np.float64)
    Xk_te = np.asarray(X_cat_test, dtype=np.float64)
    if Xk_tr.size:
        X_tr = np.hstack([B_tr, Xk_tr])
        X_te = np.hstack([B_te, Xk_te])
    else:
        X_tr, X_te = B_tr, B_te

    n_spline = B_tr.shape[1]
    n_cat = Xk_tr.shape[1] if Xk_tr.ndim == 2 else 0
    n_b = model.spline.n_basis_per_feature
    P = model.n_continuous
    D_blocks = [second_difference_penalty_matrix(n_b) for _ in range(P)]
    counts = np.asarray(y_train, dtype=np.float64)
    weights = np.ones(len(y_train), dtype=np.float64)
    lam_s, lam_r = model.lambda_s, model.lambda_r

    def nll(beta):
        eta = X_tr @ beta
        n = counts.sum()
        loss = -float(np.dot(counts, eta)) + float(n * logsumexp(eta + np.log(weights)))
        # ridge all
        loss += lam_r * float(np.dot(beta, beta))
        # smoothness spline only
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            D = D_blocks[p]
            if D.size:
                c = beta[sl]
                Dc = D @ c
                loss += lam_s * float(np.dot(Dc, Dc))
        return loss

    def grad(beta):
        eta = X_tr @ beta
        n = float(counts.sum())
        m = float(np.max(eta + np.log(weights)))
        un = np.exp(eta + np.log(weights) - m)
        p_mass = un / un.sum()
        d_eta = -counts + n * p_mass
        g = X_tr.T @ d_eta
        g = g + 2.0 * lam_r * beta
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            D = D_blocks[p]
            if D.size:
                c = beta[sl]
                g[sl] = g[sl] + 2.0 * lam_s * (D.T @ (D @ c))
        return g

    beta0 = np.zeros(X_tr.shape[1], dtype=np.float64)
    res = minimize(nll, beta0, method="L-BFGS-B", jac=grad, options={"maxiter": 400, "ftol": 1e-12})
    beta = res.x
    eta_te = X_te @ beta
    eta_tr = X_tr @ beta
    eta_te = eta_te - float(np.mean(eta_tr))
    return as_numpy_f64(eta_te), {
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(res.success or np.isfinite(eta_te).all()),
        "nll": float(res.fun),
        "beta": beta,
    }
