"""End-to-end standard KAN–IPP on continuous covariates.

Protocol: ``docs/standard_kan_e2e_can20_protocol.md``

Normative formula (continuous part on scaled inputs z = T(x)):

    eta = KAN_[P, h, 1](z) + X_cat @ beta

This is **not** DeepKanHybrid (additive edges + residual mixer on phi).
Input to the KAN is scaled raw continuous x, not AdditiveSplineKAN edge outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple  # List used in loss_history

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kanmaxent.utils import as_numpy_f64

try:
    from kan.KANLayer import KANLayer
except ImportError:  # pragma: no cover
    KANLayer = None  # type: ignore


def _require_pykan() -> None:
    if KANLayer is None:
        raise ImportError(
            "pykan is required for StandardKanIPP. Install pykan 0.2.8 in the project env."
        )


def fit_affine_scaler(
    X: np.ndarray,
    *,
    q_low: float = 1.0,
    q_high: float = 99.0,
    out_low: float = -1.0,
    out_high: float = 1.0,
    eps: float = 1e-8,
) -> Dict[str, np.ndarray]:
    """Per-feature affine map from training quantiles into [out_low, out_high].

    Fitted on training PO+BG only. Same map applied to independent PA at test time.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    lo = np.nanpercentile(X, q_low, axis=0)
    hi = np.nanpercentile(X, q_high, axis=0)
    span = np.maximum(hi - lo, eps)
    # z = out_low + (x - lo) / span * (out_high - out_low)
    scale = (out_high - out_low) / span
    shift = out_low - lo * scale
    return {
        "lo": lo,
        "hi": hi,
        "scale": scale,
        "shift": shift,
        "out_low": np.array([out_low], dtype=np.float64),
        "out_high": np.array([out_high], dtype=np.float64),
        "q_low": np.array([q_low], dtype=np.float64),
        "q_high": np.array([q_high], dtype=np.float64),
    }


def apply_affine_scaler(X: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    z = X * scaler["scale"] + scaler["shift"]
    out_low = float(scaler["out_low"][0])
    out_high = float(scaler["out_high"][0])
    return np.clip(z, out_low, out_high)


def _make_kan_layer(
    in_dim: int,
    out_dim: int,
    *,
    n_intervals: int,
    degree: int,
    grid_range: Tuple[float, float],
    freeze_grid: bool,
    disable_silu: bool,
    device: str,
) -> "KANLayer":
    if disable_silu:
        base_fun: nn.Module = nn.Identity()
        sb_trainable = False
        scale_base_sigma = 0.0
    else:
        base_fun = nn.SiLU()
        sb_trainable = True
        scale_base_sigma = 0.1

    layer = KANLayer(
        in_dim=in_dim,
        out_dim=out_dim,
        num=n_intervals,
        k=degree,
        noise_scale=0.1,
        scale_base_mu=0.0,
        scale_base_sigma=scale_base_sigma,
        scale_sp=1.0,
        base_fun=base_fun,
        grid_range=list(grid_range),
        sp_trainable=True,
        sb_trainable=sb_trainable,
        device=device,
    )
    if freeze_grid and hasattr(layer, "grid"):
        layer.grid.requires_grad_(False)
    if disable_silu:
        with torch.no_grad():
            layer.scale_base.zero_()
        layer.scale_base.requires_grad_(False)
    return layer


class StandardKanIPP(nn.Module):
    """eta = KAN_[P,h,1](T(x)) + X_cat @ beta."""

    def __init__(
        self,
        n_continuous: int,
        n_cat_oh: int = 0,
        *,
        hidden_width: int = 4,
        n_intervals: int = 6,
        degree: int = 3,
        lambda_kan: float = 1e-4,
        lambda_r: float = 1e-6,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        freeze_grid: bool = True,
        disable_silu: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        _require_pykan()
        self.n_continuous = int(n_continuous)
        self.n_cat_oh = int(n_cat_oh)
        self.hidden_width = max(1, int(hidden_width))
        self.n_intervals = int(n_intervals)
        self.degree = int(degree)
        self.lambda_kan = float(lambda_kan)
        self.lambda_r = float(lambda_r)
        self.grid_range = (float(grid_range[0]), float(grid_range[1]))
        self.freeze_grid = bool(freeze_grid)
        self.disable_silu = bool(disable_silu)
        self._device = device
        self.scaler: Optional[Dict[str, np.ndarray]] = None

        h = self.hidden_width
        self.kan_hidden = _make_kan_layer(
            self.n_continuous,
            h,
            n_intervals=n_intervals,
            degree=degree,
            grid_range=self.grid_range,
            freeze_grid=freeze_grid,
            disable_silu=disable_silu,
            device=device,
        )
        self.kan_out = _make_kan_layer(
            h,
            1,
            n_intervals=n_intervals,
            degree=degree,
            grid_range=self.grid_range,
            freeze_grid=freeze_grid,
            disable_silu=disable_silu,
            device=device,
        )
        self.kan_layers = nn.ModuleList([self.kan_hidden, self.kan_out])

        if n_cat_oh > 0:
            self.beta_cat = nn.Parameter(torch.zeros(n_cat_oh, dtype=torch.float64))
        else:
            self.register_parameter("beta_cat", None)

    def fit_scaler(self, X_cont_train: np.ndarray) -> "StandardKanIPP":
        self.scaler = fit_affine_scaler(
            X_cont_train,
            out_low=self.grid_range[0],
            out_high=self.grid_range[1],
        )
        return self

    def transform_cont(self, X_cont: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("call fit_scaler on training continuous matrix first")
        return apply_affine_scaler(X_cont, self.scaler)

    def forward_cont(self, z: Tensor) -> Tensor:
        """z: (N, P) float32 preferred by pykan; returns (N,) float64 eta_cont."""
        # pykan layers typically run in float32
        z32 = z.to(dtype=torch.float32)
        h = self.kan_hidden(z32)[0]
        out = self.kan_out(h)[0].squeeze(-1)
        return out.to(dtype=torch.float64)

    def eta(
        self,
        z: Tensor,
        X_cat: Optional[Tensor] = None,
    ) -> Tensor:
        eta = self.forward_cont(z)
        if self.beta_cat is not None and X_cat is not None and X_cat.shape[1] > 0:
            eta = eta + X_cat.to(dtype=torch.float64) @ self.beta_cat
        return eta

    def penalty(self) -> Tensor:
        pen = torch.zeros((), dtype=torch.float64)
        for layer in self.kan_layers:
            for _, p in layer.named_parameters():
                if not p.requires_grad:
                    continue
                pen = pen + (p.to(dtype=torch.float64) ** 2).sum()
        pen = self.lambda_kan * pen
        if self.beta_cat is not None:
            pen = pen + self.lambda_r * (self.beta_cat ** 2).sum()
        return pen

    def export_manifest(self) -> Dict[str, object]:
        return {
            "protocol": "standard_kan_e2e_can20",
            "architecture": f"pykan_[{self.n_continuous},{self.hidden_width},1]",
            "formula": "eta = KAN(T(x)) + X_cat @ beta",
            "kan_input": "continuous_z",
            "residual_additive_skip": False,
            "warm_start_additive": False,
            "freeze_edges_from_additive": False,
            "n_continuous": self.n_continuous,
            "n_cat_oh": self.n_cat_oh,
            "hidden_width": self.hidden_width,
            "n_intervals": self.n_intervals,
            "degree": self.degree,
            "grid_range": list(self.grid_range),
            "freeze_grid": self.freeze_grid,
            "pykan_base_activation": "Identity" if self.disable_silu else "SiLU",
            "lambda_kan": self.lambda_kan,
            "lambda_r": self.lambda_r,
            "scaler": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in (self.scaler or {}).items()
            },
        }


def init_standard_kan(model: StandardKanIPP, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for layer in model.kan_layers:
        with torch.no_grad():
            if hasattr(layer, "coef"):
                noise = rng.normal(0, 0.05, size=tuple(layer.coef.shape))
                layer.coef.copy_(
                    torch.as_tensor(noise, dtype=layer.coef.dtype, device=layer.coef.device)
                )
            if hasattr(layer, "scale_sp"):
                layer.scale_sp.fill_(1.0)
            if hasattr(layer, "scale_base") and layer.scale_base.requires_grad:
                sb = rng.normal(0, 0.05, size=tuple(layer.scale_base.shape))
                layer.scale_base.copy_(
                    torch.as_tensor(sb, dtype=layer.scale_base.dtype, device=layer.scale_base.device)
                )
    if model.beta_cat is not None:
        with torch.no_grad():
            model.beta_cat.copy_(
                torch.as_tensor(
                    rng.normal(0, 0.01, size=(model.n_cat_oh,)),
                    dtype=torch.float64,
                )
            )


def fit_standard_kan_ipp(
    X_cont_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    X_cont_test: np.ndarray,
    X_cat_test: np.ndarray,
    *,
    hidden_width: int = 4,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_kan: float = 1e-4,
    lambda_r: float = 1e-6,
    seed: int = 0,
    adam_steps: int = 150,
    adam_lr: float = 0.03,
    lbfgs_steps: int = 10,
    lbfgs_lr: float = 0.25,
    lbfgs_max_iter: int = 10,
    freeze_grid: bool = True,
    disable_silu: bool = False,
    adaptive_budget: bool = True,
) -> Tuple[StandardKanIPP, np.ndarray, Dict]:
    """Train end-to-end standard KAN–IPP; return model, test scores, meta."""
    import time

    from kanmaxent.losses.ipp import ipp_nll

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_po = int(np.sum(np.asarray(y_train) > 0))
    if adaptive_budget:
        if n_po < 30:
            adam_steps = int(adam_steps * 1.5)
            lbfgs_steps = int(lbfgs_steps * 1.5)
        elif n_po >= 100:
            adam_steps = max(adam_steps // 2, 80)

    n_cat = int(X_cat_train.shape[1]) if getattr(X_cat_train, "ndim", 0) == 2 else 0
    model = StandardKanIPP(
        X_cont_train.shape[1],
        n_cat,
        hidden_width=hidden_width,
        n_intervals=n_intervals,
        degree=degree,
        lambda_kan=lambda_kan,
        lambda_r=lambda_r,
        freeze_grid=freeze_grid,
        disable_silu=disable_silu,
    )
    model.fit_scaler(X_cont_train)
    init_standard_kan(model, seed)

    z_tr = torch.as_tensor(model.transform_cont(X_cont_train), dtype=torch.float64)
    z_te = torch.as_tensor(model.transform_cont(X_cont_test), dtype=torch.float64)
    Xk_tr = torch.as_tensor(X_cat_train, dtype=torch.float64) if n_cat else None
    Xk_te = torch.as_tensor(X_cat_test, dtype=torch.float64) if n_cat else None
    counts = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(len(y_train), dtype=torch.float64)

    last_loss = float("nan")
    last_grad_norm = float("nan")
    loss_history: List[float] = []

    def _loss() -> Tensor:
        eta = model.eta(z_tr, Xk_tr)
        return ipp_nll(eta, counts, weights, model.penalty(), per_presence=False)

    params = [p for p in model.parameters() if p.requires_grad]

    # --- Adam ---
    opt = torch.optim.Adam(params, lr=adam_lr, weight_decay=0.0)
    for _ in range(int(adam_steps)):
        opt.zero_grad()
        loss = _loss()
        if not torch.isfinite(loss):
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        last_grad_norm = float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn)
        opt.step()
        last_loss = float(loss.detach().cpu())
        loss_history.append(last_loss)

    # --- L-BFGS refine ---
    params = [p for p in model.parameters() if p.requires_grad]
    opt_l = torch.optim.LBFGS(
        params,
        lr=lbfgs_lr,
        max_iter=int(lbfgs_max_iter),
        line_search_fn="strong_wolfe",
        history_size=20,
        tolerance_grad=1e-7,
    )

    def closure():
        opt_l.zero_grad()
        loss = _loss()
        if torch.isfinite(loss):
            loss.backward()
        return loss

    for _ in range(int(lbfgs_steps)):
        loss = opt_l.step(closure)
        if loss is None or not torch.isfinite(loss):
            break
        last_loss = float(loss.detach().cpu())
        loss_history.append(last_loss)

    with torch.no_grad():
        scores = as_numpy_f64(model.eta(z_te, Xk_te))
        param_norm = 0.0
        for p in model.parameters():
            param_norm += float((p.detach() ** 2).sum().cpu())
        param_norm = float(np.sqrt(param_norm))

    hist_out = loss_history
    if len(hist_out) > 200:
        idx = np.linspace(0, len(hist_out) - 1, 200).astype(int)
        hist_out = [loss_history[i] for i in idx]

    meta = {
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(np.isfinite(scores).all() and np.isfinite(last_loss)),
        "final_loss": last_loss,
        "last_grad_norm": last_grad_norm,
        "param_l2_norm": param_norm,
        "loss_history": hist_out,
        "steps_adam": int(adam_steps),
        "steps_lbfgs": int(lbfgs_steps),
        "optimizer": "adam_then_lbfgs",
        "adam_lr": adam_lr,
        "lbfgs_lr": lbfgs_lr,
        "n_po": n_po,
        "protocol": "standard_kan_e2e_six_region_closure_v1",
        "model_manifest": model.export_manifest(),
    }
    return model, scores, meta
