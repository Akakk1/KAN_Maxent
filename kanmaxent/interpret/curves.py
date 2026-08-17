"""Export centered component functions φ_p for mechanism plots (Phase 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from kanmaxent.interpret.shape_metrics import shape_metrics
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.utils import as_numpy_f64


def inverse_standardize(x_scaled: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return np.asarray(x_scaled, dtype=np.float64) * float(scale) + float(mean)


def component_phi_on_grid(
    model: AdditiveSplineKAN,
    feature_index: int,
    *,
    n_grid: int = 100,
    X_ref: Optional[Tensor] = None,
    center_weights: Optional[Tensor] = None,
    mids: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_scaled_grid, phi) for one feature, centered on X_ref if given."""
    p = int(feature_index)
    lo = float(model.x_min[p].item())
    hi = float(model.x_max[p].item())
    xs = np.linspace(lo, hi, int(n_grid))
    if mids is None:
        if X_ref is not None:
            mids = [float(X_ref[:, j].mean().item()) for j in range(model.n_features)]
        else:
            mids = [
                0.5 * (float(model.x_min[j]) + float(model.x_max[j]))
                for j in range(model.n_features)
            ]
    X = np.tile(np.asarray(mids, dtype=np.float64), (n_grid, 1))
    X[:, p] = xs
    Xt = torch.as_tensor(X, dtype=model.coeffs.dtype, device=model.coeffs.device)
    with torch.no_grad():
        if X_ref is not None:
            phis_ref = model.component_functions(X_ref)
            w = (
                center_weights
                if center_weights is not None
                else torch.ones(X_ref.shape[0], dtype=model.coeffs.dtype, device=model.coeffs.device)
            )
            w = w / w.sum()
            means = (w[:, None] * phis_ref).sum(dim=0)
            phis = model.component_functions(Xt) - means
        else:
            phis = model.component_functions(Xt)
            phis = phis - phis.mean(dim=0, keepdim=True)
        phi = as_numpy_f64(phis[:, p])
    return xs, phi


def export_component_curves(
    model: AdditiveSplineKAN,
    env_names: Sequence[str],
    out_dir: Union[str, Path],
    *,
    n_grid: int = 100,
    center_weights: Optional[Tensor] = None,
    X_ref: Optional[Tensor] = None,
    scaler_mean: Optional[Sequence[float]] = None,
    scaler_scale: Optional[Sequence[float]] = None,
    train_q: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Path:
    """Write one CSV per feature: x_scaled, x_raw (optional), phi.

    If scaler_mean/scale provided, also write physical units x_raw.
    train_q: optional (q_lo, q_hi) per feature on scaled axis for support flags.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if X_ref is not None:
        X_ref = X_ref.to(dtype=model.coeffs.dtype, device=model.coeffs.device)
        mids = [float(X_ref[:, p].mean().item()) for p in range(model.n_features)]
    else:
        mids = [
            0.5 * (float(model.x_min[p]) + float(model.x_max[p]))
            for p in range(model.n_features)
        ]

    rows_metrics: List[Dict] = []
    for p, name in enumerate(env_names):
        xs, phi = component_phi_on_grid(
            model, p, n_grid=n_grid, X_ref=X_ref, center_weights=center_weights, mids=mids
        )
        if scaler_mean is not None and scaler_scale is not None:
            x_raw = inverse_standardize(xs, scaler_mean[p], scaler_scale[p])
            header = "x_scaled,x_raw,phi"
            arr = np.column_stack([xs, x_raw, phi])
        else:
            header = "x_scaled,phi"
            arr = np.column_stack([xs, phi])
        if train_q is not None:
            qlo, qhi = train_q
            in_support = ((xs >= qlo[p]) & (xs <= qhi[p])).astype(np.float64)
            arr = np.column_stack([arr, in_support])
            header = header + ",in_train_support"
        path = out_dir / f"phi_{name}.csv"
        np.savetxt(path, arr, delimiter=",", header=header, comments="")
        sm = shape_metrics(xs, phi)
        sm["feature"] = name
        rows_metrics.append(sm)

    # shape metrics table
    import csv

    sm_path = out_dir / "shape_metrics.csv"
    if rows_metrics:
        keys = list(rows_metrics[0].keys())
        with sm_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows_metrics:
                w.writerow(r)
    return out_dir


def bootstrap_component_curves(
    fit_fn,
    *,
    n_boot: int,
    seed: int,
    env_names: Sequence[str],
    n_grid: int = 100,
    feature_index: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Generic bootstrap: fit_fn(rng) -> (model, X_ref_tensor, meta).

    Returns dict name -> array (n_boot, n_grid) of phi curves on a fixed grid
    from the first successful fit's x range (scaled).
    """
    rng = np.random.default_rng(seed)
    curves: Dict[str, List[np.ndarray]] = {n: [] for n in env_names}
    x_grid_ref: Optional[np.ndarray] = None
    for b in range(int(n_boot)):
        model, X_ref, _meta = fit_fn(rng)
        idxs = range(model.n_features) if feature_index is None else [feature_index]
        for p in idxs:
            name = env_names[p]
            xs, phi = component_phi_on_grid(model, p, n_grid=n_grid, X_ref=X_ref)
            if x_grid_ref is None and p == (feature_index if feature_index is not None else 0):
                x_grid_ref = xs
            curves[name].append(phi)
    out = {n: np.stack(v, axis=0) for n, v in curves.items() if v}
    if x_grid_ref is not None:
        out["_x_scaled"] = x_grid_ref
    return out


# backward-compatible alias used in older docs
def export_component_curves_legacy(*args, **kwargs):
    return export_component_curves(*args, **kwargs)
