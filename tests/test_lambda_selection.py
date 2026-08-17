"""λs selector unit tests."""

from __future__ import annotations

import numpy as np

from kanmaxent.evaluation.lambda_selector import select_lambda_s_po_cv
from kanmaxent.losses.bce import bce_with_logits_nll
import torch


def test_frozen_when_few_po():
    def dummy(tr, va, lam):
        return 0.5

    out = select_lambda_s_po_cv(20, 1000, dummy, min_po_for_cv=30)
    assert out["path"] == "frozen_n_po_lt_30"
    assert out["lambda_s"] == 1e-2


def test_selects_best_on_grid():
    def fit_eval(tr, va, lam):
        # prefer 1e-2 on the closure grid
        table = {1e-4: 0.4, 1e-3: 0.5, 1e-2: 0.9, 1e-1: 0.6, 1.0: 0.55}
        return table[float(lam)]

    out = select_lambda_s_po_cv(40, 1000, fit_eval, min_po_for_cv=30, seed=0)
    assert out["path"] == "po_random_5fold"
    assert out["lambda_s"] == 1e-2
    assert out["k_folds"] == 5


def test_bce_pos_weight_runs():
    logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    loss = bce_with_logits_nll(logits, y, pos_weight=10.0)
    assert torch.isfinite(loss)
