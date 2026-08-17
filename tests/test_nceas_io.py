"""NCEAS CAN I/O tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kanmaxent.data.categorical import CategoricalEncoder, prepare_matrices
from kanmaxent.data.nceas_io import get_variable_list, load_region, species_presence_counts

ROOT = Path(__file__).resolve().parents[1]
CAN_DIR = ROOT / "data" / "nceas" / "can"


@pytest.mark.skipif(not (CAN_DIR / "CAN_po.csv").is_file(), reason="CAN data missing")
def test_load_can_shapes():
    data = load_region("CAN", CAN_DIR)
    assert data.region == "CAN"
    assert len(data.species_ids) == 20
    assert len(data.po) == 5063
    assert len(data.bg) == 50000
    assert len(data.pa) == 14571
    assert "ontveg" in data.categorical
    assert "alt" in data.continuous
    counts = species_presence_counts(data.po)
    assert counts["can02"] == 740
    assert min(counts.values()) >= 5  # all CAN >= 5 after Valavi filter context


def test_variable_list_can():
    v = get_variable_list("CAN")
    assert v["categorical"] == ["ontveg"]
    assert len(v["continuous"]) == 6


def test_categorical_encoder_expand():
    import pandas as pd

    tr = pd.DataFrame({"ontveg": [1, 2, 3, 1]})
    te = pd.DataFrame({"ontveg": [1, 4]})  # 4 unseen
    enc = CategoricalEncoder(cat_cols=["ontveg"]).fit(tr)
    Xte = enc.transform(te)
    assert Xte.shape[1] == 3  # levels 1,2,3
    # unseen 4 → zeros
    assert np.allclose(Xte[1], 0.0)


@pytest.mark.skipif(not (CAN_DIR / "CAN_po.csv").is_file(), reason="CAN data missing")
def test_prepare_matrices_shapes():
    data = load_region("CAN", CAN_DIR)
    from kanmaxent.data.nceas_io import build_po_bg_frame, split_species

    po = split_species(data.po, "can01")
    covars = data.continuous + data.categorical
    tr = build_po_bg_frame(po, data.bg.sample(500, random_state=0), covars)
    te = data.env[covars].head(100)
    prep = prepare_matrices(tr, te, data.continuous, data.categorical)
    assert prep["X_train"].shape[0] == len(tr)
    assert prep["X_test"].shape[0] == 100
    assert prep["X_cont_train"].shape[1] == 6
    assert prep["n_categorical_oh"] >= 1
