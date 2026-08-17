"""Phase 3 multi-region I/O tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kanmaxent.data.nceas_io import (
    load_region,
    pa_labels_and_env_for_species,
    species_group,
    build_tgb_frame,
)

ROOT = Path(__file__).resolve().parents[1]
NCEAS = ROOT / "data" / "nceas"


@pytest.mark.skipif(not (NCEAS / "nsw" / "NSW_po.csv").is_file(), reason="NSW missing")
def test_nsw_group_pa():
    d = load_region("NSW")
    assert d.multi_group
    assert len(d.species_ids) >= 50
    sp = d.species_ids[0]
    g = species_group(d.po, sp)
    y, X, g2 = pa_labels_and_env_for_species(d, sp)
    assert g == g2
    assert len(y) == len(X)
    assert y.sum() >= 0


@pytest.mark.skipif(not (NCEAS / "awt" / "AWT_po.csv").is_file(), reason="AWT missing")
def test_awt_groups():
    d = load_region("AWT")
    groups = set(d.po["group"].astype(str))
    assert "bird" in groups and "plant" in groups
    assert d.multi_group


@pytest.mark.skipif(not (NCEAS / "nz" / "NZ_po.csv").is_file(), reason="NZ missing")
def test_nz_dual_categorical():
    d = load_region("NZ")
    assert "age" in d.categorical and "toxicats" in d.categorical
    sp = d.species_ids[0]
    y, X, g = pa_labels_and_env_for_species(d, sp)
    assert X.shape[1] == len(d.continuous) + len(d.categorical)


@pytest.mark.skipif(not (NCEAS / "awt" / "AWT_po.csv").is_file(), reason="AWT missing")
def test_tgb_excludes_focal():
    d = load_region("AWT")
    sp = d.species_ids[0]
    cov = d.continuous + d.categorical
    tr, meta = build_tgb_frame(d.po, sp, cov, max_bg=5000, seed=0)
    assert meta["tgb_excludes_focal_presence"] is True
    assert (tr["occ"] == 1).sum() > 0
