"""Anti-leakage gates for per-fold preprocessing (Phase 1 v2 §4.2)."""

from __future__ import annotations

import numpy as np

from kanmaxent.data.preprocess import FoldPreprocessor


def test_preprocessor_fit_only_sees_train_stats():
    """Changing test block values must not change train-fitted scaler."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(200, 3))
    X_test = rng.normal(5, 2, size=(50, 3))

    pre = FoldPreprocessor()
    pre.fit(X_train)
    mean1 = pre.mean_.copy()
    scale1 = pre.scale_.copy()

    # Corrupt test massively and re-transform; mean/scale must be unchanged
    X_test2 = X_test + 1000.0
    _ = pre.transform(X_test2)
    assert np.allclose(pre.mean_, mean1)
    assert np.allclose(pre.scale_, scale1)

    # Refit including test would change — prove difference if wrongly fit on all
    pre_bad = FoldPreprocessor().fit(np.vstack([X_train, X_test2]))
    assert not np.allclose(pre_bad.mean_, mean1)


def test_per_fold_scalers_differ():
    """Different train blocks yield different scalers (not global fit)."""
    rng = np.random.default_rng(1)
    # Two blocks with different means
    A = rng.normal(0, 1, size=(100, 2))
    B = rng.normal(10, 1, size=(100, 2))
    pre_a = FoldPreprocessor().fit(A)
    pre_b = FoldPreprocessor().fit(B)
    assert not np.allclose(pre_a.mean_, pre_b.mean_)


def test_transform_requires_fit():
    pre = FoldPreprocessor()
    try:
        pre.transform(np.zeros((5, 2)))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
