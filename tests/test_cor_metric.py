"""COR metric (Pearson / point-biserial with PA labels)."""

from __future__ import annotations

import numpy as np

from kanmaxent.evaluation.metrics import cor_pa, ranking_metrics


def test_cor_perfect_ranking():
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    s = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.7])
    c = cor_pa(y, s)
    assert c > 0.9
    m = ranking_metrics(y, s)
    assert "cor" in m
    assert abs(m["cor"] - c) < 1e-12


def test_cor_constant_scores_nan():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    s = np.ones(4)
    assert np.isnan(cor_pa(y, s))
