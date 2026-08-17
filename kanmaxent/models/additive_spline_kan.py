"""Additive B-spline KAN: eta(x) = sum_p phi_p(x_p), bias fixed at 0."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kanmaxent.models.bspline import (
    bspline_design_matrix,
    make_open_uniform_knots,
    n_basis,
    second_difference_penalty_matrix,
)


class AdditiveSplineKAN(nn.Module):
    """Reference additive B-spline model (auditable design matrix + coeffs).

    Parameters
    ----------
    n_features
        Number of environmental covariates P.
    n_intervals
        G: number of grid intervals per feature.
    degree
        K: B-spline degree (plan: K=3 cubic).
    x_min, x_max
        Per-feature bounds, shape (P,). Fitted from training data if omitted
        and `fit_bounds` is called / provided at init via data.
    linear_mode
        If True, force G=1, K=1 (linear degeneration for trust-anchor tests).
    lambda_s, lambda_r
        Smoothness (second-difference) and ridge penalties.
    dtype
        Default float64 for numerical tests; float32 ok for larger runs.
    """

    def __init__(
        self,
        n_features: int,
        n_intervals: int = 8,
        degree: int = 3,
        *,
        x_min: Optional[Sequence[float]] = None,
        x_max: Optional[Sequence[float]] = None,
        linear_mode: bool = False,
        lambda_s: float = 1e-2,
        lambda_r: float = 1e-6,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if linear_mode:
            n_intervals, degree = 1, 1

        self.n_features = int(n_features)
        self.n_intervals = int(n_intervals)
        self.degree = int(degree)
        self.linear_mode = bool(linear_mode)
        self.lambda_s = float(lambda_s)
        self.lambda_r = float(lambda_r)
        self._dtype = dtype

        self.n_basis_per_feature = n_basis(self.n_intervals, self.degree)
        # Coefficients: (P, n_basis)
        self.coeffs = nn.Parameter(
            torch.zeros(self.n_features, self.n_basis_per_feature, dtype=dtype)
        )

        if x_min is None:
            x_min = [0.0] * self.n_features
        if x_max is None:
            x_max = [1.0] * self.n_features
        x_min_t = torch.as_tensor(x_min, dtype=dtype)
        x_max_t = torch.as_tensor(x_max, dtype=dtype)
        if x_min_t.numel() != self.n_features or x_max_t.numel() != self.n_features:
            raise ValueError("x_min/x_max must have length n_features")
        self.register_buffer("x_min", x_min_t)
        self.register_buffer("x_max", x_max_t)

        self._rebuild_knots()
        # Precompute second-diff matrices (shared structure per feature)
        D = second_difference_penalty_matrix(self.n_basis_per_feature)
        self.register_buffer(
            "D_smooth",
            torch.as_tensor(D, dtype=dtype) if D.size else torch.zeros(0, self.n_basis_per_feature, dtype=dtype),
        )

    def _rebuild_knots(self) -> None:
        knots_list = []
        for p in range(self.n_features):
            lo = float(self.x_min[p].item())
            hi = float(self.x_max[p].item())
            if hi <= lo:
                hi = lo + 1.0
            knots_list.append(
                make_open_uniform_knots(lo, hi, self.n_intervals, self.degree)
            )
        # (P, n_knots)
        knots_arr = np.stack(knots_list, axis=0)
        if hasattr(self, "knots"):
            self.knots.copy_(torch.as_tensor(knots_arr, dtype=self._dtype, device=self.coeffs.device))
        else:
            self.register_buffer("knots", torch.as_tensor(knots_arr, dtype=self._dtype))

    @torch.no_grad()
    def fit_bounds(self, X: Union[np.ndarray, Tensor], eps: float = 1e-8) -> "AdditiveSplineKAN":
        """Set per-feature knots from training data bounds (call per fold)."""
        if isinstance(X, Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim != 2 or X_np.shape[1] != self.n_features:
            raise ValueError(f"X must be (N, {self.n_features})")
        lo = X_np.min(axis=0)
        hi = X_np.max(axis=0)
        span = np.maximum(hi - lo, eps)
        # Small pad to avoid boundary degeneracy
        lo = lo - 0.01 * span
        hi = hi + 0.01 * span
        self.x_min.copy_(torch.as_tensor(lo, dtype=self._dtype, device=self.x_min.device))
        self.x_max.copy_(torch.as_tensor(hi, dtype=self._dtype, device=self.x_max.device))
        self._rebuild_knots()
        return self

    def design_matrix_feature(self, x_p: Tensor, feature: int) -> Tensor:
        """B-spline design for one feature: (N, n_basis)."""
        knots = self.knots[feature].detach().cpu().numpy()
        B = bspline_design_matrix(
            x_p.detach().cpu().numpy(),
            knots,
            self.degree,
            as_torch=True,
            dtype=self.coeffs.dtype,
            device=self.coeffs.device,
        )
        return B

    def full_design_matrix(self, X: Tensor) -> Tensor:
        """Block-diagonal stacked design: (N, P * n_basis)."""
        blocks = [self.design_matrix_feature(X[:, p], p) for p in range(self.n_features)]
        return torch.cat(blocks, dim=1)

    def component_functions(self, X: Tensor) -> Tensor:
        """Return phi_p(x_p) for each feature: shape (N, P)."""
        N = X.shape[0]
        phis = []
        for p in range(self.n_features):
            B = self.design_matrix_feature(X[:, p], p)
            phi = B @ self.coeffs[p]
            phis.append(phi)
        return torch.stack(phis, dim=1)

    def forward(
        self,
        X: Tensor,
        *,
        center: bool = True,
        center_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict eta(x) = sum_p phi_p(x_p), optional weighted centering.

        Centering enforces weighted mean of each phi_p = 0 under the reference
        distribution given by center_weights (default: uniform over rows).
        Global bias remains 0 (unidentifiable under conditional IPP).
        """
        phis = self.component_functions(X)  # (N, P)
        if center:
            if center_weights is None:
                w = torch.ones(X.shape[0], dtype=X.dtype, device=X.device)
            else:
                w = center_weights.to(dtype=X.dtype, device=X.device)
            w = w / w.sum()
            means = (w[:, None] * phis).sum(dim=0)  # (P,)
            phis = phis - means
        eta = phis.sum(dim=1)
        return eta

    def smoothness_penalty(self) -> Tensor:
        """Second-difference penalty sum_p ||D c_p||^2."""
        if self.D_smooth.numel() == 0:
            return torch.zeros((), dtype=self.coeffs.dtype, device=self.coeffs.device)
        # D: (n-2, n_basis), coeffs: (P, n_basis)
        Dc = self.coeffs @ self.D_smooth.T  # (P, n-2)
        return (Dc ** 2).sum()

    def ridge_penalty(self) -> Tensor:
        return (self.coeffs ** 2).sum()

    def penalty(self) -> Tensor:
        """Omega = lambda_s * smooth + lambda_r * ridge."""
        return self.lambda_s * self.smoothness_penalty() + self.lambda_r * self.ridge_penalty()

    def export_state(self) -> Dict[str, object]:
        """Export knots, coeffs, basis metadata for audit / GAM parity."""
        return {
            "n_features": self.n_features,
            "n_intervals": self.n_intervals,
            "degree": self.degree,
            "n_basis_per_feature": self.n_basis_per_feature,
            "linear_mode": self.linear_mode,
            "lambda_s": self.lambda_s,
            "lambda_r": self.lambda_r,
            "x_min": self.x_min.detach().cpu().numpy().copy(),
            "x_max": self.x_max.detach().cpu().numpy().copy(),
            "knots": self.knots.detach().cpu().numpy().copy(),
            "coeffs": self.coeffs.detach().cpu().numpy().copy(),
        }

    def set_coeffs_from_numpy(self, coeffs: np.ndarray) -> None:
        with torch.no_grad():
            self.coeffs.copy_(
                torch.as_tensor(coeffs, dtype=self.coeffs.dtype, device=self.coeffs.device)
            )
