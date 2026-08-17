"""Synthetic presence-only scenarios for Phase 0 math tests."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _sample_points(
    rng: np.random.Generator,
    n: int,
    n_features: int,
    uneven: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return X (n, P) and positive weights w (n,)."""
    if uneven:
        # Cluster mass in a corner of the domain
        X = rng.beta(0.4, 2.0, size=(n, n_features))
        # Heterogeneous cell area weights
        w = rng.lognormal(mean=0.0, sigma=0.8, size=n)
    else:
        X = rng.uniform(0.0, 1.0, size=(n, n_features))
        w = np.ones(n, dtype=np.float64)
    w = np.maximum(w, 1e-6)
    return X.astype(np.float64), w.astype(np.float64)


def make_linear_scenario(
    n_points: int = 400,
    n_features: int = 2,
    n_presence: int = 80,
    seed: int = 0,
    uneven: bool = True,
) -> Dict[str, np.ndarray]:
    """Scenario 1: true log-intensity is linear in features (E.3)."""
    rng = np.random.default_rng(seed)
    X, w = _sample_points(rng, n_points, n_features, uneven)
    # True slopes
    beta_true = np.linspace(0.8, 1.4, n_features)
    eta_true = X @ beta_true
    # Sample presence indices proportional to w * exp(eta)
    m = eta_true.max()
    logits = np.log(w) + eta_true - m
    logits = logits - logits.max()
    prob = np.exp(logits)
    prob = prob / prob.sum()
    # Multinomial presence counts (allows aggregation)
    counts = rng.multinomial(n_presence, prob).astype(np.float64)
    return {
        "X": X,
        "weights": w,
        "counts": counts,
        "eta_true": eta_true,
        "beta_true": beta_true,
        "scenario": np.array(["linear"]),
    }


def make_nonlinear_scenario(
    n_points: int = 500,
    n_features: int = 2,
    n_presence: int = 100,
    seed: int = 1,
) -> Dict[str, np.ndarray]:
    """Scenario 2: smooth nonlinear additive truth (E.4 / E.6)."""
    rng = np.random.default_rng(seed)
    X, w = _sample_points(rng, n_points, n_features, uneven=True)
    # phi1 = sin(2 pi x), phi2 = (x-0.5)^2
    eta_true = np.sin(2 * np.pi * X[:, 0]) + 2.0 * (X[:, 1] - 0.5) ** 2
    m = eta_true.max()
    logits = np.log(w) + eta_true - m
    logits = logits - logits.max()
    prob = np.exp(logits)
    prob = prob / prob.sum()
    counts = rng.multinomial(n_presence, prob).astype(np.float64)
    return {
        "X": X,
        "weights": w,
        "counts": counts,
        "eta_true": eta_true,
        "scenario": np.array(["nonlinear"]),
    }


def make_uneven_batch_scenario(
    n_points: int = 300,
    n_features: int = 2,
    n_presence: int = 60,
    n_batches: int = 5,
    seed: int = 2,
) -> Dict[str, np.ndarray]:
    """Scenario 3: extreme uneven weights + batch ids (E.5)."""
    data = make_linear_scenario(
        n_points=n_points,
        n_features=n_features,
        n_presence=n_presence,
        seed=seed,
        uneven=True,
    )
    rng = np.random.default_rng(seed + 99)
    # Biased batch assignment: early batches get high-weight points
    order = np.argsort(-data["weights"])
    batch_index = np.zeros(n_points, dtype=np.int64)
    # Uneven batch sizes
    cuts = np.linspace(0, n_points, n_batches + 1).astype(int)
    for b in range(n_batches):
        idx = order[cuts[b] : cuts[b + 1]]
        batch_index[idx] = b
    data["batch_index"] = batch_index
    data["n_batches"] = np.array([n_batches])
    return data
