"""Minimal optimizers for Phase 0 math tests (not a full training pipeline)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from kanmaxent.losses.ipp import ipp_nll
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN


def fit_additive_kan_ipp(
    model: AdditiveSplineKAN,
    X: Tensor,
    counts: Tensor,
    weights: Tensor,
    *,
    steps: int = 800,
    lr: float = 0.1,
    per_presence: bool = False,
    verbose: bool = False,
) -> AdditiveSplineKAN:
    """Fit AdditiveSplineKAN with full-domain IPP NLL + explicit penalties.

    Uses LBFGS on coefficients. Does NOT use Adam weight_decay (explicit ridge
    via model.penalty() only — see docs/source_compatibility.md).
    """
    X = X.to(dtype=model.coeffs.dtype, device=model.coeffs.device)
    counts = counts.to(dtype=model.coeffs.dtype, device=model.coeffs.device)
    weights = weights.to(dtype=model.coeffs.dtype, device=model.coeffs.device)

    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=20,
        line_search_fn="strong_wolfe",
        history_size=50,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
    )

    def closure():
        opt.zero_grad()
        eta = model(X, center=True, center_weights=weights)
        pen = model.penalty()
        loss = ipp_nll(eta, counts, weights, pen, per_presence=per_presence)
        loss.backward()
        return loss

    for step in range(steps):
        loss = opt.step(closure)
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"step {step}: loss={float(loss):.6f}")
        # Early stop if grad small
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += float(p.grad.detach().norm().item())
        if grad_norm < 1e-9:
            break

    return model
