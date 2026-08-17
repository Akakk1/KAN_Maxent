"""Explicit B-spline design matrices (Cox–de Boor), auditable for GAM parity."""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import Tensor

ArrayLike = Union[np.ndarray, Tensor, Sequence[float]]


def make_open_uniform_knots(
    x_min: float,
    x_max: float,
    n_intervals: int,
    degree: int,
) -> np.ndarray:
    """Open uniform knot vector for G intervals and degree K.

    n_basis = G + K. Interior knots: G - 1 unique points (G=1 → none).
    """
    if n_intervals < 1:
        raise ValueError("n_intervals (G) must be >= 1")
    if degree < 1:
        raise ValueError("degree (K) must be >= 1")
    if x_max <= x_min:
        raise ValueError("x_max must be > x_min")

    if n_intervals == 1:
        interior = np.array([], dtype=np.float64)
    else:
        interior = np.linspace(x_min, x_max, n_intervals + 1)[1:-1]

    knots = np.concatenate(
        [
            np.full(degree + 1, x_min, dtype=np.float64),
            interior,
            np.full(degree + 1, x_max, dtype=np.float64),
        ]
    )
    return knots


def n_basis(n_intervals: int, degree: int) -> int:
    """Number of B-spline basis functions: G + K."""
    return n_intervals + degree


def _cox_de_boor(x: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate all B-spline bases at x. Returns (n_points, n_basis)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    knots = np.asarray(knots, dtype=np.float64).ravel()
    t_lo = knots[degree]
    t_hi = knots[len(knots) - degree - 1]
    # Clamp for open uniform extrapolation at boundaries
    xc = np.clip(x, t_lo, t_hi)

    n_spans = len(knots) - 1
    # Degree 0 indicator bases on knot spans
    B = np.zeros((len(xc), n_spans), dtype=np.float64)
    for i in range(n_spans):
        left, right = knots[i], knots[i + 1]
        if right <= left:
            continue
        if right >= t_hi:
            B[:, i] = ((xc >= left) & (xc <= right)).astype(np.float64)
        else:
            B[:, i] = ((xc >= left) & (xc < right)).astype(np.float64)

    for d in range(1, degree + 1):
        B_next = np.zeros((len(xc), n_spans - d), dtype=np.float64)
        for i in range(n_spans - d):
            denom1 = knots[i + d] - knots[i]
            denom2 = knots[i + d + 1] - knots[i + 1]
            t1 = 0.0
            t2 = 0.0
            if denom1 > 0:
                t1 = ((xc - knots[i]) / denom1) * B[:, i]
            if denom2 > 0:
                t2 = ((knots[i + d + 1] - xc) / denom2) * B[:, i + 1]
            B_next[:, i] = t1 + t2
        B = B_next

    return B


def bspline_design_matrix(
    x: ArrayLike,
    knots: ArrayLike,
    degree: int,
    *,
    as_torch: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> Union[np.ndarray, Tensor]:
    """Return design matrix B with shape (n_points, n_basis)."""
    x_np = np.asarray(x, dtype=np.float64).ravel()
    knots_np = np.asarray(knots, dtype=np.float64).ravel()
    B = _cox_de_boor(x_np, knots_np, degree)
    if as_torch:
        return torch.as_tensor(B, dtype=dtype, device=device)
    return B


def second_difference_penalty_matrix(n_basis_dim: int) -> np.ndarray:
    """D such that ||D c||^2 = sum (c_{k+1} - 2 c_k + c_{k-1})^2.

    Shape (n-2, n) for n >= 3; empty (0, n) otherwise.
    """
    n = n_basis_dim
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D
