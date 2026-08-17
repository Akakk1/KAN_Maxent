"""I/O and integrity tests for Ginkgo Phase 1 data copy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kanmaxent.data.ginkgo_io import (
    EXPECTED_SHA256,
    env_columns,
    load_ginkgo,
    sha256_file,
    split_outer_fold,
)

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "ginkgo" / "ginkgo_training_with_coords.csv"


@pytest.mark.skipif(not CSV.is_file(), reason="data/ginkgo copy missing")
def test_sha256_matches_phase0():
    assert sha256_file(CSV) == EXPECTED_SHA256


@pytest.mark.skipif(not CSV.is_file(), reason="data/ginkgo copy missing")
def test_load_shape_and_folds():
    df = load_ginkgo(CSV)
    assert len(df) == 8948
    assert int((df["label"] == 1).sum()) == 238
    assert int((df["label"] == 0).sum()) == 8710
    assert set(df["fold"].unique()) == {0, 1, 2, 3, 4}
    cols = env_columns(df)
    assert len(cols) == 10
    assert "bio13" in cols and "bio11" in cols


@pytest.mark.skipif(not CSV.is_file(), reason="data/ginkgo copy missing")
def test_split_outer_fold_partition():
    df = load_ginkgo(CSV)
    for f in range(5):
        sp = split_outer_fold(df, f)
        assert len(sp.train_idx) + len(sp.test_idx) == len(df)
        assert set(sp.train_idx).isdisjoint(sp.test_idx)
        assert sp.n_test_presence > 0
        assert sp.n_train_presence > 0
