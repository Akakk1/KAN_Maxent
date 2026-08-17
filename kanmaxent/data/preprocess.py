"""Per-fold preprocessing: StandardScaler + bookkeeping (anti-leakage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class FoldPreprocessor:
    """Fit scaler on train rows only; transform train/test with that fit.

    Knots/bounds are handled by AdditiveSplineKAN.fit_bounds on scaled train X;
    this class only owns the covariate scaler to keep leakage tests simple.
    """

    scaler: Optional[StandardScaler] = None
    n_features_: Optional[int] = None
    fitted_on_n_: Optional[int] = None
    _fitted: bool = False
    # For audit: which fold this was fit on (optional metadata)
    meta: Dict[str, Any] = field(default_factory=dict)

    def fit(self, X_train: np.ndarray) -> "FoldPreprocessor":
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2D")
        if X_train.shape[0] < 1:
            raise ValueError("X_train is empty")
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        self.n_features_ = int(X_train.shape[1])
        self.fitted_on_n_ = int(X_train.shape[0])
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self.scaler is None:
            raise RuntimeError("FoldPreprocessor must be fit before transform")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X shape {X.shape} incompatible with fitted n_features={self.n_features_}"
            )
        return self.scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def mean_(self) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("not fitted")
        return np.asarray(self.scaler.mean_, dtype=np.float64)

    @property
    def scale_(self) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("not fitted")
        return np.asarray(self.scaler.scale_, dtype=np.float64)

    def export_state(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "n_features": self.n_features_,
            "fitted_on_n": self.fitted_on_n_,
            "meta": dict(self.meta),
        }
