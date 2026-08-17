"""Phase 1 trainers: IPP-KAN, BCE-KAN, same-basis GAM (shared knots)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from kanmaxent.losses.bce import bce_with_logits_nll
from kanmaxent.losses.ipp import ipp_nll
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.bspline import second_difference_penalty_matrix
from kanmaxent.reference.gam_ipp import fit_gam_ipp
from kanmaxent.utils import as_numpy_f64


@dataclass
class TrainResult:
    model_name: str
    scores_test: np.ndarray
    eta_train: Optional[np.ndarray]
    converged: bool
    runtime_s: float
    extras: Dict


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _init_coeffs(model: AdditiveSplineKAN, seed: int, scale: float = 0.01) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, scale, size=tuple(model.coeffs.shape))
    model.set_coeffs_from_numpy(noise.astype(np.float64))


def make_shared_kan_shell(
    n_features: int,
    X_train_scaled: np.ndarray,
    *,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
) -> AdditiveSplineKAN:
    """Create AdditiveSplineKAN and fit_bounds on train (shared basis source)."""
    model = AdditiveSplineKAN(
        n_features=n_features,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
        dtype=torch.float64,
    )
    model.fit_bounds(X_train_scaled)
    return model


def _eta_from_design(
    model: AdditiveSplineKAN,
    B: Tensor,
    *,
    center: bool,
    center_weights: Optional[Tensor] = None,
) -> Tensor:
    """eta = sum_p B_p @ c_p with optional weighted centering of each phi_p.

    B is full design (N, P * n_basis) matching model.full_design_matrix.
    """
    n_b = model.n_basis_per_feature
    P = model.n_features
    phis = []
    for p in range(P):
        sl = slice(p * n_b, (p + 1) * n_b)
        phis.append(B[:, sl] @ model.coeffs[p])
    phis_t = torch.stack(phis, dim=1)
    if center:
        if center_weights is None:
            w = torch.ones(B.shape[0], dtype=B.dtype, device=B.device)
        else:
            w = center_weights.to(dtype=B.dtype, device=B.device)
        w = w / w.sum()
        means = (w[:, None] * phis_t).sum(dim=0)
        phis_t = phis_t - means
    return phis_t.sum(dim=1)


def _predict_test_centered(
    model: AdditiveSplineKAN,
    B_train: Tensor,
    B_test: Tensor,
    w_train: Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        n_b = model.n_basis_per_feature
        P = model.n_features
        phis_tr = []
        phis_te = []
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            phis_tr.append(B_train[:, sl] @ model.coeffs[p])
            phis_te.append(B_test[:, sl] @ model.coeffs[p])
        phis_tr_t = torch.stack(phis_tr, dim=1)
        phis_te_t = torch.stack(phis_te, dim=1)
        ww = w_train / w_train.sum()
        means = (ww[:, None] * phis_tr_t).sum(dim=0)
        eta_tr = (phis_tr_t - means).sum(dim=1)
        eta_te = (phis_te_t - means).sum(dim=1)
    return as_numpy_f64(eta_tr), as_numpy_f64(eta_te)


def fit_ipp_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    seed: int = 0,
    steps: int = 40,
    lr: float = 0.3,
) -> Tuple[AdditiveSplineKAN, TrainResult]:
    import time

    t0 = time.perf_counter()
    _seed_all(seed)
    model = make_shared_kan_shell(
        X_train.shape[1],
        X_train,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
    )
    _init_coeffs(model, seed)

    Xtr = torch.as_tensor(X_train, dtype=torch.float64)
    Xte = torch.as_tensor(X_test, dtype=torch.float64)
    # Cache design matrices once (shared basis with GAM)
    B_tr = model.full_design_matrix(Xtr).detach()
    B_te = model.full_design_matrix(Xte).detach()
    counts = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(X_train.shape[0], dtype=torch.float64)

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
        history_size=40,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
    )

    def closure():
        opt.zero_grad()
        eta = _eta_from_design(model, B_tr, center=True, center_weights=weights)
        pen = model.penalty()
        loss = ipp_nll(eta, counts, weights, pen, per_presence=False)
        loss.backward()
        return loss

    last = None
    for _ in range(steps):
        last = opt.step(closure)
        if last is not None and not torch.isfinite(last):
            break

    eta_tr, eta_te = _predict_test_centered(model, B_tr, B_te, weights)
    ok = bool(np.isfinite(eta_te).all() and np.isfinite(eta_tr).all())
    res = TrainResult(
        model_name="additive_kan_ipp",
        scores_test=eta_te,
        eta_train=eta_tr,
        converged=ok,
        runtime_s=float(time.perf_counter() - t0),
        extras={"lambda_s": lambda_s, "lambda_r": lambda_r, "seed": seed, "B_train": B_tr},
    )
    return model, res


def fit_bce_kan(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    seed: int = 0,
    steps: int = 40,
    lr: float = 0.3,
) -> Tuple[AdditiveSplineKAN, TrainResult]:
    import time

    t0 = time.perf_counter()
    _seed_all(seed)
    model = make_shared_kan_shell(
        X_train.shape[1],
        X_train,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
    )
    _init_coeffs(model, seed)

    Xtr = torch.as_tensor(X_train, dtype=torch.float64)
    Xte = torch.as_tensor(X_test, dtype=torch.float64)
    B_tr = model.full_design_matrix(Xtr).detach()
    B_te = model.full_design_matrix(Xte).detach()
    y = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(X_train.shape[0], dtype=torch.float64)

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
        history_size=40,
        tolerance_grad=1e-9,
        tolerance_change=1e-11,
    )

    def closure():
        opt.zero_grad()
        logits = _eta_from_design(model, B_tr, center=True, center_weights=weights)
        pen = model.penalty()
        loss = bce_with_logits_nll(logits, y, pen / max(len(y), 1), reduction="mean")
        loss.backward()
        return loss

    for _ in range(steps):
        loss = opt.step(closure)
        if not torch.isfinite(loss):
            break

    eta_tr, eta_te = _predict_test_centered(model, B_tr, B_te, weights)
    ok = bool(np.isfinite(eta_te).all())
    res = TrainResult(
        model_name="additive_kan_bce",
        scores_test=eta_te,
        eta_train=eta_tr,
        converged=ok,
        runtime_s=float(time.perf_counter() - t0),
        extras={"lambda_s": lambda_s, "lambda_r": lambda_r, "seed": seed},
    )
    return model, res


def fit_gam_ipp_shared(
    basis_model: AdditiveSplineKAN,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    lambda_s: Optional[float] = None,
    lambda_r: Optional[float] = None,
    B_train: Optional[Tensor] = None,
    B_test: Optional[Tensor] = None,
) -> TrainResult:
    """GAM/IPP on the same design matrix / knots as basis_model."""
    import time

    t0 = time.perf_counter()
    lambda_s = float(basis_model.lambda_s if lambda_s is None else lambda_s)
    lambda_r = float(basis_model.lambda_r if lambda_r is None else lambda_r)

    Xtr = torch.as_tensor(X_train, dtype=torch.float64)
    Xte = torch.as_tensor(X_test, dtype=torch.float64)
    if B_train is None:
        X_design_tr = as_numpy_f64(basis_model.full_design_matrix(Xtr))
    else:
        X_design_tr = as_numpy_f64(B_train)
    if B_test is None:
        X_design_te = as_numpy_f64(basis_model.full_design_matrix(Xte))
    else:
        X_design_te = as_numpy_f64(B_test)

    counts = np.asarray(y_train, dtype=np.float64)
    weights = np.ones(X_train.shape[0], dtype=np.float64)
    D_blocks: List[np.ndarray] = [
        second_difference_penalty_matrix(basis_model.n_basis_per_feature)
        for _ in range(basis_model.n_features)
    ]

    beta, eta_tr, nll = fit_gam_ipp(
        X_design_tr,
        counts,
        weights,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
        D_blocks=D_blocks,
        n_basis_per_feature=basis_model.n_basis_per_feature,
        n_features=basis_model.n_features,
        maxiter=600,
    )
    eta_te = X_design_te @ beta
    eta_te = eta_te - float(np.mean(eta_tr))
    ok = bool(np.isfinite(eta_te).all() and np.isfinite(nll))
    return TrainResult(
        model_name="gam_ipp",
        scores_test=as_numpy_f64(eta_te),
        eta_train=as_numpy_f64(eta_tr),
        converged=ok,
        runtime_s=float(time.perf_counter() - t0),
        extras={
            "lambda_s": lambda_s,
            "lambda_r": lambda_r,
            "nll": float(nll),
            "beta": beta,
        },
    )
