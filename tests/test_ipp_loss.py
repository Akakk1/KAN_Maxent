"""E.1 finite-diff vs autograd; E.2 logsumexp vs direct integral; E.5 batch-softmax."""

from __future__ import annotations

import numpy as np
import torch

from kanmaxent.data.synthetic import make_linear_scenario, make_uneven_batch_scenario
from kanmaxent.losses.ipp import (
    batch_softmax_nll_counterexample,
    ipp_nll,
    log_partition,
    weighted_integral_exp,
)


def test_e1_finite_difference_matches_autograd():
    """E.1: finite-diff gradient of IPP NLL matches autograd (rel < 1e-5 or abs < 1e-7)."""
    data = make_linear_scenario(n_points=80, n_features=2, n_presence=30, seed=3)
    eta = torch.tensor(data["eta_true"], dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(data["counts"], dtype=torch.float64)
    weights = torch.tensor(data["weights"], dtype=torch.float64)

    loss = ipp_nll(eta, counts, weights, penalty=None)
    loss.backward()
    g_auto = eta.grad.detach().clone()

    # Central finite differences
    eps = 1e-6
    g_fd = torch.zeros_like(eta)
    with torch.no_grad():
        for j in range(eta.numel()):
            e_p = eta.detach().clone()
            e_m = eta.detach().clone()
            e_p[j] += eps
            e_m[j] -= eps
            lp = ipp_nll(e_p, counts, weights)
            lm = ipp_nll(e_m, counts, weights)
            g_fd[j] = (lp - lm) / (2 * eps)

    abs_err = (g_auto - g_fd).abs()
    rel_err = abs_err / (g_auto.abs() + 1e-12)
    ok = (rel_err < 1e-5) | (abs_err < 1e-7)
    assert torch.all(ok), (
        f"gradient mismatch: max abs={abs_err.max():.3e}, max rel={rel_err.max():.3e}"
    )


def test_e2_logsumexp_matches_direct_integral():
    """E.2: logsumexp(eta+log_w) vs log(sum w exp(eta)), rel err < 1e-7."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        M = 100
        eta = torch.tensor(rng.normal(0, 1.5, size=M), dtype=torch.float64)
        w = torch.tensor(rng.lognormal(0, 0.5, size=M), dtype=torch.float64)
        log_z = log_partition(eta, torch.log(w))
        z_direct = weighted_integral_exp(eta, w)
        log_z_direct = torch.log(z_direct)
        rel = (log_z - log_z_direct).abs() / (log_z_direct.abs() + 1e-12)
        # Also check exp domain relative error on Z
        rel_z = (torch.exp(log_z) - z_direct).abs() / (z_direct.abs() + 1e-12)
        assert float(rel) < 1e-7 or float(rel_z) < 1e-7


def test_e2_with_large_eta_stable():
    """logsumexp remains finite for large eta where naive exp may overflow."""
    eta = torch.tensor([100.0, 101.0, 99.0], dtype=torch.float64)
    w = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float64)
    log_z = log_partition(eta, torch.log(w))
    assert torch.isfinite(log_z)
    # Cross-check with max-shifted direct sum
    m = eta.max()
    z = (w * torch.exp(eta - m)).sum() * torch.exp(m)
    assert torch.isclose(torch.exp(log_z), z, rtol=1e-10)


def test_e5_batch_softmax_counterexample():
    """E.5: batch-softmax partition is biased vs full logsumexp under uneven weights."""
    data = make_uneven_batch_scenario(n_points=300, n_presence=60, n_batches=5, seed=2)
    eta = torch.tensor(data["eta_true"], dtype=torch.float64, requires_grad=True)
    counts = torch.tensor(data["counts"], dtype=torch.float64)
    weights = torch.tensor(data["weights"], dtype=torch.float64)
    batch_index = torch.tensor(data["batch_index"], dtype=torch.long)
    n_batches = int(data["n_batches"][0])

    loss_full = ipp_nll(eta, counts, weights)
    loss_full.backward()
    g_full = eta.grad.detach().clone()
    eta.grad = None

    eta2 = eta.detach().clone().requires_grad_(True)
    loss_batch = batch_softmax_nll_counterexample(
        eta2, counts, weights, batch_index, n_batches
    )
    loss_batch.backward()
    g_batch = eta2.grad.detach().clone()

    # Loss values differ
    rel_loss = abs(float(loss_full - loss_batch)) / (abs(float(loss_full)) + 1e-12)
    # Gradient cosine similarity < 0.999 or relative L2 difference material
    cos = float(
        torch.nn.functional.cosine_similarity(
            g_full.reshape(1, -1), g_batch.reshape(1, -1)
        )
    )
    rel_g = float((g_full - g_batch).norm() / (g_full.norm() + 1e-12))

    assert rel_loss > 1e-4 or cos < 0.999 or rel_g > 1e-3, (
        f"batch-softmax unexpectedly close: rel_loss={rel_loss}, cos={cos}, rel_g={rel_g}"
    )
    # Record that full path is the reference
    assert torch.isfinite(loss_full)
