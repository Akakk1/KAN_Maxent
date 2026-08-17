"""Smoke tests for Deep-2 / Deep-3 KAN (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("kan")

from kanmaxent.models.deep_kan import DeepKanHybrid, fit_deep_kan_ipp, init_deep_kan


def test_deep_kan_forward_shape():
    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    model = DeepKanHybrid(p, n_cat_oh=0, n_intervals=4, degree=3)
    model.fit_bounds(X)
    init_deep_kan(model, 0)
    B = model.spline_design(torch.as_tensor(X, dtype=torch.float64))
    eta = model.eta_from_parts(B, None)
    assert eta.shape == (n,)
    assert torch.isfinite(eta).all()
    pen = model.penalty()
    assert torch.isfinite(pen)
    m = model.export_manifest()
    assert m["architecture"].startswith("deep2")
    assert m["pykan_entropy_penalty"] is False
    assert m["freeze_grid"] is True
    assert m["depth"] == 2


def test_deep3_forward_shape():
    rng = np.random.default_rng(0)
    n, p = 200, 4
    X = rng.normal(size=(n, p))
    model = DeepKanHybrid(
        p, n_cat_oh=0, n_intervals=4, degree=3, depth=3, hidden_width=3, residual=True
    )
    model.fit_bounds(X)
    init_deep_kan(model, 0)
    B = model.spline_design(torch.as_tensor(X, dtype=torch.float64))
    eta = model.eta_from_parts(B, None)
    assert eta.shape == (n,)
    assert torch.isfinite(eta).all()
    assert torch.isfinite(model.penalty())
    m = model.export_manifest()
    assert m["depth"] == 3
    assert m["hidden_width"] == 3
    assert m["architecture"].startswith("deep3")


def test_deep_kan_fit_reduces_loss_or_finite():
    rng = np.random.default_rng(1)
    n, p = 800, 3
    X = rng.normal(size=(n, p))
    # synthetic: interaction y depends on x0*x1
    logit = 0.8 * X[:, 0] * X[:, 1] - 0.3 * X[:, 2]
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < prob * 0.15).astype(np.float64)
    # ensure some presences
    y[:20] = 1.0
    X_te = rng.normal(size=(100, p))
    Xc = np.zeros((n, 0))
    Xc_te = np.zeros((100, 0))
    model, scores, meta = fit_deep_kan_ipp(
        X,
        Xc,
        y,
        X_te,
        Xc_te,
        n_intervals=4,
        degree=3,
        lambda_s=1e-2,
        seed=0,
        optimizer="lbfgs",
        lbfgs_steps=8,
        residual=True,
        disable_silu=True,
        warm_start_additive=True,
        adaptive_budget=False,
    )
    assert scores.shape == (100,)
    assert meta["converged"]
    assert np.isfinite(meta["final_loss"])
    assert meta["residual"] is True
    assert meta["disable_silu"] is True
    assert meta["depth"] == 2


def test_deep3_fit_finite():
    rng = np.random.default_rng(2)
    n, p = 600, 3
    X = rng.normal(size=(n, p))
    logit = 0.5 * X[:, 0] * X[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit)) * 0.2).astype(np.float64)
    y[:15] = 1.0
    X_te = rng.normal(size=(80, p))
    Xc = np.zeros((n, 0))
    Xc_te = np.zeros((80, 0))
    _, scores, meta = fit_deep_kan_ipp(
        X,
        Xc,
        y,
        X_te,
        Xc_te,
        n_intervals=4,
        degree=3,
        lambda_s=1e-2,
        seed=0,
        optimizer="lbfgs",
        lbfgs_steps=6,
        residual=True,
        disable_silu=True,
        warm_start_additive=True,
        adaptive_budget=False,
        depth=3,
        hidden_width=3,
    )
    assert scores.shape == (80,)
    assert meta["converged"]
    assert meta["depth"] == 3
    assert meta["hidden_width"] == 3
    assert np.isfinite(meta["final_loss"])


def test_deep_kan_manifest_disable_silu():
    model = DeepKanHybrid(3, 0, disable_silu=True, residual=True)
    m = model.export_manifest()
    assert m["pykan_base_activation"] == "Identity"
    assert m["residual"] is True
