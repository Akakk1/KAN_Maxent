"""Unit tests for Phase 5 shape metrics and device policy."""

from __future__ import annotations

import numpy as np
import torch

from kanmaxent.interpret.device import parse_device_policy
from kanmaxent.interpret.shape_metrics import curve_peak_agreement, shape_metrics


def test_monotonic_increasing():
    x = np.linspace(0, 1, 50)
    phi = 2 * x - 0.5
    m = shape_metrics(x, phi)
    assert m["monotonicity"] > 0.99
    assert m["n_local_extrema"] == 0
    assert abs(m["argmax_x"] - 1.0) < 1e-6


def test_unimodal():
    x = np.linspace(-2, 2, 201)
    phi = np.exp(-(x**2))
    m = shape_metrics(x, phi)
    assert m["n_local_extrema"] >= 1
    assert abs(m["argmax_x"]) < 0.05


def test_peak_agreement():
    x = np.linspace(0, 1, 100)
    a = np.sin(2 * np.pi * x)
    b = a + 0.01
    r = curve_peak_agreement(x, a, b)
    assert r["pearson_r"] > 0.99


def test_device_policy_prefer_cpu_lbfgs():
    pol = parse_device_policy("cuda", prefer_cpu_lbfgs=True)
    dev = pol.resolve_fit_device()
    assert dev.type == "cpu"
    assert any("prefer_cpu_lbfgs" in e for e in pol.fallback_events)
