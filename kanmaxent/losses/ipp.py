"""Discrete inhomogeneous Poisson process (IPP) conditional likelihood.

Negative conditional log-likelihood (unscaled, v6.1 §2.2):

    L = -sum_j c_j * eta_j + n * log(sum_j w_j * exp(eta_j)) + Omega

Implementation uses logsumexp(eta + log_w). Batch-softmax is NOT a valid
default partition estimator (see batch_softmax_nll_counterexample).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def log_partition(eta: Tensor, log_w: Tensor) -> Tensor:
    """Stable log Z = log sum_j w_j exp(eta_j) = logsumexp(eta + log_w)."""
    if eta.shape != log_w.shape:
        raise ValueError(f"eta and log_w shapes must match, got {eta.shape} vs {log_w.shape}")
    return torch.logsumexp(eta + log_w, dim=-1)


def weighted_integral_exp(eta: Tensor, weights: Tensor) -> Tensor:
    """Direct (possibly unstable) Z = sum_j w_j exp(eta_j). For tests only."""
    return (weights * torch.exp(eta)).sum(dim=-1)


def ipp_nll(
    eta: Tensor,
    counts: Tensor,
    weights: Tensor,
    penalty: Optional[Tensor] = None,
    *,
    per_presence: bool = False,
) -> Tensor:
    """Unscaled (or per-presence) negative IPP conditional log-likelihood.

    Parameters
    ----------
    eta
        Log relative intensity at M support points, shape (M,) or (B, M).
    counts
        Presence counts c_j at support points, same shape as eta.
    weights
        Area / integration weights w_j > 0, same shape as eta (or broadcastable).
    penalty
        Optional scalar Omega(theta). If per_presence=True, must already be
        scaled consistently (caller responsibility) or will be divided by n.
    per_presence
        If True, divide the entire loss (including penalty) by n = sum c_j.

    Returns
    -------
    Scalar (or batch) negative log-likelihood.
    """
    eta = eta.reshape(-1) if eta.ndim == 0 else eta
    if eta.ndim == 1:
        return _ipp_nll_1d(eta, counts, weights, penalty, per_presence=per_presence)

    # Batched: (B, M)
    losses = []
    for b in range(eta.shape[0]):
        pen_b = penalty[b] if penalty is not None and penalty.ndim > 0 else penalty
        losses.append(
            _ipp_nll_1d(eta[b], counts[b], weights[b], pen_b, per_presence=per_presence)
        )
    return torch.stack(losses)


def _ipp_nll_1d(
    eta: Tensor,
    counts: Tensor,
    weights: Tensor,
    penalty: Optional[Tensor],
    *,
    per_presence: bool,
) -> Tensor:
    counts = counts.to(dtype=eta.dtype, device=eta.device)
    weights = weights.to(dtype=eta.dtype, device=eta.device)
    if torch.any(weights <= 0):
        raise ValueError("All integration weights must be strictly positive")

    n = counts.sum()
    # Presence linear term; if n==0, loss reduces to penalty only (degenerate)
    lin = (counts * eta).sum()
    log_w = torch.log(weights)
    log_z = log_partition(eta, log_w)
    loss = -lin + n * log_z

    if penalty is not None:
        loss = loss + penalty

    if per_presence:
        if n <= 0:
            raise ValueError("per_presence=True requires sum(counts) > 0")
        loss = loss / n

    return loss


def batch_softmax_nll_counterexample(
    eta: Tensor,
    counts: Tensor,
    weights: Tensor,
    batch_index: Tensor,
    n_batches: int,
) -> Tensor:
    """Incorrect batch-softmax style partition (for anti-tests only).

    Splits support points into batches and averages per-batch
    n_b * logsumexp_b terms instead of one global log-partition. This does
    not equal the full IPP conditional likelihood under non-uniform weights
    or uneven batch composition.
    """
    counts = counts.to(dtype=eta.dtype, device=eta.device)
    weights = weights.to(dtype=eta.dtype, device=eta.device)
    n = counts.sum()
    lin = (counts * eta).sum()

    # Approximate Z by mean of batch Z's (common mistaken pattern)
    log_z_terms = []
    for b in range(n_batches):
        mask = batch_index == b
        if not torch.any(mask):
            continue
        eta_b = eta[mask]
        w_b = weights[mask]
        # Wrong: treat batch as if it were the full domain
        log_z_b = torch.logsumexp(eta_b + torch.log(w_b), dim=-1)
        log_z_terms.append(log_z_b)

    if not log_z_terms:
        raise ValueError("empty batch_index")
    # Another wrong variant: average batch log-partitions
    log_z_approx = torch.stack(log_z_terms).mean()
    return -lin + n * log_z_approx
