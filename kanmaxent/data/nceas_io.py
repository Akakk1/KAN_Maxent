"""NCEAS / disdat region loaders (Phase 2–3)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Valavi et al. 2021 / vars.R
REGION_COVARS: Dict[str, Dict[str, List[str]]] = {
    "CAN": {
        "continuous": ["alt", "asp2", "ontprec", "ontslp", "onttemp", "watdist"],
        "categorical": ["ontveg"],
    },
    "AWT": {
        "continuous": ["bc04", "bc05", "bc06", "bc12", "bc15", "slope", "topo", "tri"],
        "categorical": [],
    },
    "NSW": {
        "continuous": [
            "cti", "disturb", "mi", "rainann", "raindq", "rugged",
            "soildepth", "soilfert", "solrad", "tempann", "topo",
        ],
        "categorical": ["vegsys"],
    },
    "NZ": {
        "continuous": [
            "deficit", "hillshade", "mas", "mat", "r2pet", "slope",
            "sseas", "tseas", "vpd",
        ],
        "categorical": ["age", "toxicats"],
    },
    "SA": {
        "continuous": [
            "sabio12", "sabio15", "sabio17", "sabio18",
            "sabio2", "sabio4", "sabio5", "sabio6",
        ],
        "categorical": [],
    },
    "SWI": {
        "continuous": [
            "bcc", "ccc", "ddeg", "nutri", "pday", "precyy",
            "sfroyy", "slope", "sradyy", "swb", "topo",
        ],
        "categorical": ["calc"],
    },
}

MULTI_GROUP_REGIONS = {"AWT", "NSW"}


def default_region_dir(region: str = "CAN") -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "nceas" / region.lower()


def get_variable_list(region: str) -> Dict[str, List[str]]:
    r = region.upper()
    if r not in REGION_COVARS:
        raise KeyError(f"Unknown region {region}; known={list(REGION_COVARS)}")
    return {
        "continuous": list(REGION_COVARS[r]["continuous"]),
        "categorical": list(REGION_COVARS[r]["categorical"]),
        "all": list(REGION_COVARS[r]["continuous"]) + list(REGION_COVARS[r]["categorical"]),
    }


@dataclass
class RegionData:
    region: str
    po: pd.DataFrame
    bg: pd.DataFrame
    continuous: List[str]
    categorical: List[str]
    species_ids: List[str]
    manifest: dict
    # single-file regions
    pa: Optional[pd.DataFrame] = None
    env: Optional[pd.DataFrame] = None
    # multi-group: group -> (pa, env)
    pa_by_group: Dict[str, pd.DataFrame] = field(default_factory=dict)
    env_by_group: Dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def multi_group(self) -> bool:
        return self.region in MULTI_GROUP_REGIONS or bool(self.pa_by_group)


def _load_bg(data_dir: Path, region: str) -> pd.DataFrame:
    bg_path = data_dir / f"{region}_bg_50k.csv"
    if not bg_path.is_file():
        alt = (
            Path(__file__).resolve().parents[2]
            / "data"
            / "NCEAS_Valavi2021"
            / "DataS1"
            / "background_50k"
            / f"{region}.csv"
        )
        bg_path = alt
    return pd.read_csv(bg_path)


def load_region(
    region: str = "CAN",
    data_dir: Optional[Path] = None,
) -> RegionData:
    region = region.upper()
    data_dir = Path(data_dir) if data_dir is not None else default_region_dir(region)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)

    po = pd.read_csv(data_dir / f"{region}_po.csv")
    bg = _load_bg(data_dir, region)
    vars_ = get_variable_list(region)
    species_ids = sorted(po["spid"].astype(str).unique().tolist())

    pa_by_group: Dict[str, pd.DataFrame] = {}
    env_by_group: Dict[str, pd.DataFrame] = {}
    pa_single = None
    env_single = None

    # Multi-group files: REGION_pa_{group}.csv
    group_files = sorted(data_dir.glob(f"{region}_pa_*.csv"))
    if group_files:
        for pf in group_files:
            g = pf.name.replace(f"{region}_pa_", "").replace(".csv", "")
            pa_g = pd.read_csv(pf)
            env_f = data_dir / f"{region}_env_{g}.csv"
            if not env_f.is_file():
                raise FileNotFoundError(env_f)
            env_g = pd.read_csv(env_f)
            pa_by_group[g] = pa_g
            env_by_group[g] = env_g
        # verify each species appears in its group's PA
        for s in species_ids:
            g = str(po.loc[po["spid"].astype(str) == s, "group"].iloc[0])
            if g not in pa_by_group:
                raise ValueError(f"{region}/{s}: group={g} has no PA file")
            if s not in pa_by_group[g].columns:
                raise ValueError(f"{region}/{s}: missing column in PA group={g}")
    else:
        pa_single = pd.read_csv(data_dir / f"{region}_pa.csv")
        env_single = pd.read_csv(data_dir / f"{region}_env.csv")
        for s in species_ids:
            if s not in pa_single.columns:
                raise ValueError(f"PA missing species column {s}")

    man_path = data_dir / "MANIFEST.json"
    manifest = json.loads(man_path.read_text()) if man_path.is_file() else {}

    return RegionData(
        region=region,
        po=po,
        bg=bg,
        continuous=vars_["continuous"],
        categorical=vars_["categorical"],
        species_ids=species_ids,
        manifest=manifest,
        pa=pa_single,
        env=env_single,
        pa_by_group=pa_by_group,
        env_by_group=env_by_group,
    )


def split_species(po: pd.DataFrame, species_id: str) -> pd.DataFrame:
    return po.loc[po["spid"].astype(str) == str(species_id)].copy()


def species_group(po: pd.DataFrame, species_id: str) -> str:
    sub = po.loc[po["spid"].astype(str) == str(species_id), "group"]
    if len(sub) == 0:
        raise KeyError(species_id)
    return str(sub.iloc[0])


def species_presence_counts(po: pd.DataFrame) -> Dict[str, int]:
    return po.groupby(po["spid"].astype(str)).size().astype(int).to_dict()


def build_po_bg_frame(
    po_sp: pd.DataFrame,
    bg: pd.DataFrame,
    covars: Sequence[str],
) -> pd.DataFrame:
    """Stack species PO (occ=1) and background (occ=0)."""
    cols = list(covars)
    a = po_sp[cols].copy()
    a["occ"] = 1
    b = bg[cols].copy()
    b["occ"] = 0
    return pd.concat([a, b], axis=0, ignore_index=True)


def build_tgb_frame(
    po_all: pd.DataFrame,
    species_id: str,
    covars: Sequence[str],
    *,
    max_bg: Optional[int] = 50000,
    seed: int = 0,
) -> Tuple[pd.DataFrame, dict]:
    """Target-group background: other PO in same taxon group as occ=0.

    Does not include focal presence sites. If too few others, returns empty bg
    info for caller to fall back.
    """
    po_sp = split_species(po_all, species_id)
    g = species_group(po_all, species_id)
    others = po_all.loc[
        (po_all["group"].astype(str) == g) & (po_all["spid"].astype(str) != str(species_id))
    ].copy()
    # unique sites
    if "siteid" in others.columns:
        others = others.drop_duplicates(subset=["siteid"])
    n_raw = len(others)
    if max_bg is not None and n_raw > max_bg:
        others = others.sample(n=max_bg, random_state=seed)
    cols = list(covars)
    a = po_sp[cols].copy()
    a["occ"] = 1
    b = others[cols].copy()
    b["occ"] = 0
    meta = {
        "tgb_group": g,
        "tgb_n_other_sites": n_raw,
        "tgb_n_used": len(b),
        "tgb_excludes_focal_presence": True,
    }
    train = pd.concat([a, b], axis=0, ignore_index=True)
    return train, meta


def pa_labels_and_env_for_species(
    data: RegionData,
    species_id: str,
) -> Tuple[np.ndarray, pd.DataFrame, str]:
    """Return y_pa, env_df, taxon_group for a species."""
    covars = data.continuous + data.categorical
    g = species_group(data.po, species_id)
    if data.pa_by_group:
        pa = data.pa_by_group[g]
        env = data.env_by_group[g]
    else:
        assert data.pa is not None and data.env is not None
        pa = data.pa
        env = data.env
    y, Xdf = pa_labels_and_env(pa, env, species_id, covars)
    return y, Xdf, g


def pa_labels_and_env(
    pa: pd.DataFrame,
    env: pd.DataFrame,
    species_id: str,
    covars: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Align PA labels with env rows."""
    if len(pa) != len(env):
        m = pa.merge(env, on="siteid", how="inner", suffixes=("", "_env"))
        y = m[species_id].to_numpy(dtype=np.float64)
        Xdf = m[list(covars)].copy()
        return y, Xdf
    y = pa[species_id].to_numpy(dtype=np.float64)
    Xdf = env[list(covars)].copy()
    return y, Xdf


def write_region_manifest(region: str, data_dir: Path) -> dict:
    """Build and write MANIFEST.json after export."""
    data = load_region(region, data_dir)
    counts = species_presence_counts(data.po)
    groups = sorted(data.po["group"].astype(str).unique().tolist())
    man = {
        "region": region.upper(),
        "disdat_version": "1.1.0",
        "n_species": len(data.species_ids),
        "species_ids": data.species_ids,
        "n_po_total": int(len(data.po)),
        "n_bg_total": int(len(data.bg)),
        "groups": groups,
        "multi_group_pa": bool(data.pa_by_group),
        "presence_range": [int(min(counts.values())), int(max(counts.values()))],
        "covariates": {
            "continuous": data.continuous,
            "categorical": data.categorical,
        },
        "estimand": "global_conditional_density",
        "integration_support": "full_background_50k",
    }
    (data_dir / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man
