"""Ginkgo data loaders and outer-fold splits (Phase 1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "b7dd1e870066d935eb7817aadf88fb6d7ed514a506d6394f27e4d56d85223c1d"
NON_ENV = ("label", "fold", "decimalLatitude", "decimalLongitude")


def sha256_file(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in NON_ENV]


def default_ginkgo_csv() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ginkgo" / "ginkgo_training_with_coords.csv"


def load_ginkgo(
    path: Optional[Union[str, Path]] = None,
    *,
    check_sha256: bool = True,
    expected_sha256: str = EXPECTED_SHA256,
) -> pd.DataFrame:
    """Load Ginkgo training table; optionally verify SHA-256."""
    path = Path(path) if path is not None else default_ginkgo_csv()
    if not path.is_file():
        raise FileNotFoundError(path)
    if check_sha256:
        digest = sha256_file(path)
        if digest != expected_sha256:
            raise ValueError(
                f"SHA-256 mismatch for {path}: got {digest}, expected {expected_sha256}"
            )
    df = pd.read_csv(path)
    required = {"label", "fold", "decimalLongitude", "decimalLatitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if env_columns(df) == []:
        raise ValueError("No environmental columns found")
    return df


@dataclass
class OuterFoldSplit:
    """Index masks for one outer spatial fold."""

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_presence_idx: np.ndarray
    train_background_idx: np.ndarray
    test_presence_idx: np.ndarray
    test_background_idx: np.ndarray

    @property
    def n_train_presence(self) -> int:
        return int(len(self.train_presence_idx))

    @property
    def n_train_background(self) -> int:
        return int(len(self.train_background_idx))

    @property
    def n_test_presence(self) -> int:
        return int(len(self.test_presence_idx))

    @property
    def n_test_background(self) -> int:
        return int(len(self.test_background_idx))


def split_outer_fold(df: pd.DataFrame, fold_id: int) -> OuterFoldSplit:
    """Split by precomputed spatial fold column (Phase 1 default protocol)."""
    fold = df["fold"].to_numpy()
    label = df["label"].to_numpy()
    n = len(df)
    idx = np.arange(n)
    test_mask = fold == fold_id
    train_mask = ~test_mask
    train_idx = idx[train_mask]
    test_idx = idx[test_mask]
    return OuterFoldSplit(
        fold_id=int(fold_id),
        train_idx=train_idx,
        test_idx=test_idx,
        train_presence_idx=idx[train_mask & (label == 1)],
        train_background_idx=idx[train_mask & (label == 0)],
        test_presence_idx=idx[test_mask & (label == 1)],
        test_background_idx=idx[test_mask & (label == 0)],
    )


def extract_xy(
    df: pd.DataFrame,
    idx: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return X (n, P), y (n,), env column names."""
    cols = env_columns(df)
    if idx is None:
        X = df[cols].to_numpy(dtype=np.float64)
        y = df["label"].to_numpy(dtype=np.float64)
    else:
        X = df.iloc[idx][cols].to_numpy(dtype=np.float64)
        y = df.iloc[idx]["label"].to_numpy(dtype=np.float64)
    return X, y, cols


def write_data_manifest(dest_dir: Union[str, Path], source_rel: str, sha: str) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": source_rel,
        "sha256": sha,
        "copied_at": pd.Timestamp.utcnow().isoformat(),
    }
    out = dest_dir / "MANIFEST.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
