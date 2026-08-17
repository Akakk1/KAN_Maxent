"""Shape metrics for centered univariate response curves φ(x)."""

from __future__ import annotations

from typing import Dict

import numpy as np


def shape_metrics(x: np.ndarray, phi: np.ndarray) -> Dict[str, float]:
    """Compute simple shape descriptors on a 1D curve.

    Parameters
    ----------
    x, phi
        Same length; x should be sorted ascending (will sort if needed).
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if x.size != phi.size or x.size < 3:
        return {
            "n": float(x.size),
            "monotonicity": float("nan"),
            "n_local_extrema": float("nan"),
            "argmax_x": float("nan"),
            "argmin_x": float("nan"),
            "phi_range": float("nan"),
            "roughness_l2": float("nan"),
        }
    order = np.argsort(x)
    x = x[order]
    phi = phi[order]

    d1 = np.diff(phi)
    # fraction of steps with consistent sign (ignore zeros)
    nz = d1[np.abs(d1) > 1e-12]
    if nz.size == 0:
        mono = 1.0
    else:
        pos = float(np.mean(nz > 0))
        mono = float(max(pos, 1.0 - pos))

    # local extrema: sign changes of first difference
    s = np.sign(d1)
    s[s == 0] = np.nan
    # forward-fill zeros already set nan; count sign flips on finite
    s_f = s[np.isfinite(s)]
    n_ext = 0
    if s_f.size >= 2:
        n_ext = int(np.sum(s_f[1:] * s_f[:-1] < 0))

    d2 = np.diff(phi, n=2)
    roughness = float(np.sqrt(np.mean(d2**2))) if d2.size else 0.0

    return {
        "n": float(x.size),
        "monotonicity": mono,
        "n_local_extrema": float(n_ext),
        "argmax_x": float(x[int(np.argmax(phi))]),
        "argmin_x": float(x[int(np.argmin(phi))]),
        "phi_range": float(np.nanmax(phi) - np.nanmin(phi)),
        "roughness_l2": roughness,
        "phi_mean": float(np.mean(phi)),
        "phi_std": float(np.std(phi)),
    }


def curve_peak_agreement(x: np.ndarray, phi_a: np.ndarray, phi_b: np.ndarray) -> Dict[str, float]:
    """Compare two curves on the same x-grid."""
    x = np.asarray(x, dtype=np.float64).ravel()
    a = np.asarray(phi_a, dtype=np.float64).ravel()
    b = np.asarray(phi_b, dtype=np.float64).ravel()
    if x.size < 3 or a.size != b.size:
        return {"pearson_r": float("nan"), "argmax_abs_delta": float("nan"), "max_abs_delta": float("nan")}
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    ia, ib = int(np.argmax(a)), int(np.argmax(b))
    return {
        "pearson_r": r,
        "argmax_abs_delta": float(abs(x[ia] - x[ib])),
        "max_abs_delta": float(np.max(np.abs(a - b))),
        "argmax_x_a": float(x[ia]),
        "argmax_x_b": float(x[ib]),
    }
