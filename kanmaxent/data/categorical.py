"""Categorical encoding with train-level expansion (Valavi fct_expand style)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class CategoricalEncoder:
    """One-hot encode categoricals; levels fixed from training data.

    Unknown test levels → all-zero one-hot row (no crash).
    """

    cat_cols: List[str] = field(default_factory=list)
    levels_: Dict[str, List[str]] = field(default_factory=dict)
    feature_names_: List[str] = field(default_factory=list)
    drop_first: bool = False  # full one-hot + ridge (scheme A)

    def fit(self, df: pd.DataFrame) -> "CategoricalEncoder":
        self.levels_ = {}
        self.feature_names_ = []
        self._fitted = True
        for c in self.cat_cols:
            if c not in df.columns:
                raise KeyError(c)
            vals = sorted({str(v) for v in df[c].dropna().unique().tolist()})
            self.levels_[c] = vals
            levels = vals[1:] if self.drop_first and len(vals) > 1 else vals
            for v in levels:
                self.feature_names_.append(f"{c}__{v}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("CategoricalEncoder must be fit first")
        n = len(df)
        if not self.feature_names_:
            return np.zeros((n, 0), dtype=np.float64)
        cols = []
        for c in self.cat_cols:
            s = df[c].astype(str)
            levels = self.levels_[c]
            use = levels[1:] if self.drop_first and len(levels) > 1 else levels
            for v in use:
                cols.append((s == v).to_numpy(dtype=np.float64))
        return np.column_stack(cols) if cols else np.zeros((n, 0), dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


@dataclass
class ContinuousScaler:
    """Global z-score on continuous columns (Valavi: fit on PO+BG)."""

    cols: List[str] = field(default_factory=list)
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "ContinuousScaler":
        X = df[self.cols].to_numpy(dtype=np.float64)
        self.mean_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd[sd < 1e-12] = 1.0
        self.scale_ = sd
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("not fitted")
        X = df[self.cols].to_numpy(dtype=np.float64)
        return (X - self.mean_) / self.scale_


def prepare_matrices(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str],
) -> Dict[str, object]:
    """Fit scaler+encoder on train; transform train and test.

    Returns dict with X_train, X_test (concat continuous scaled + one-hot),
    n_continuous, n_categorical_oh, encoder, scaler, feature meta.
    """
    cont = list(continuous)
    cats = list(categorical)
    scaler = ContinuousScaler(cols=cont).fit(train_df)
    enc = CategoricalEncoder(cat_cols=cats, drop_first=False).fit(train_df)

    Xc_tr = scaler.transform(train_df)
    Xc_te = scaler.transform(test_df)
    Xk_tr = enc.transform(train_df)
    Xk_te = enc.transform(test_df)

    X_tr = np.hstack([Xc_tr, Xk_tr]) if Xk_tr.size else Xc_tr
    X_te = np.hstack([Xc_te, Xk_te]) if Xk_te.size else Xc_te

    return {
        "X_train": X_tr,
        "X_test": X_te,
        "X_cont_train": Xc_tr,
        "X_cont_test": Xc_te,
        "X_cat_train": Xk_tr,
        "X_cat_test": Xk_te,
        "n_continuous": len(cont),
        "n_categorical_oh": Xk_tr.shape[1] if Xk_tr.ndim == 2 else 0,
        "scaler": scaler,
        "encoder": enc,
        "cat_feature_names": list(enc.feature_names_),
        "cont_cols": cont,
    }
