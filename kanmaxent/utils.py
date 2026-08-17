"""Small numeric helpers (robust under torch/numpy ABI quirks)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def as_numpy_f64(x: Any) -> np.ndarray:
    """Convert tensor/array-like to a clean float64 ndarray.

    Uses tolist() for torch tensors to avoid rare torch↔numpy dtype ABI
    issues seen with some numpy 2.x + torch builds.
    """
    if isinstance(x, Tensor):
        return np.array(x.detach().cpu().tolist(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64).copy()
