"""Deep KAN for interaction ablation (Phase 4 / Deep-3 appendix).

Architecture (plan_v6.3 §9.D / Plan_Phase4_DeepKAN_v1):

Default **residual** (fairer ablation of interaction marginal value):

    Deep-2:  eta(x) = sum_p φ_p(x_p) + Φ(u) + X_cat @ β
             Φ: R^P → R  via KANLayer(P, 1)

    Deep-3:  eta(x) = sum_p φ_p(x_p) + Ψ(Φ(u)) + X_cat @ β
             Φ: R^P → R^h, Ψ: R^h → R  via KAN layers [P, h, 1]

Mixer input ``u`` is either:
  - ``phi``: centered/scaled additive edge outputs (legacy fair residual)
  - ``raw_scaled``: T(x) continuous covariates, 1%–99% quantile affine → [-1,1]
    (Methodological Closure Gap II; same T family as standard KAN e2e)

When Φ≈0 (and Ψ≈0) this recovers the additive baseline. Optional pure form
drops the sum_p φ_p term.

Grids frozen; SiLU base optional (fair default: off); entropy/sparsity off.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN
from kanmaxent.models.standard_kan_ipp import apply_affine_scaler, fit_affine_scaler
from kanmaxent.utils import as_numpy_f64

try:
    from kan.KANLayer import KANLayer
except ImportError:  # pragma: no cover
    KANLayer = None  # type: ignore


def _require_pykan() -> None:
    if KANLayer is None:
        raise ImportError(
            "pykan is required for DeepKanHybrid. Install pykan 0.2.8 in kan_spe."
        )


def _make_kan_layer(
    in_dim: int,
    out_dim: int,
    *,
    n_intervals: int,
    degree: int,
    residual: bool,
    disable_silu: bool,
    kan_grid_range: Tuple[float, float],
    freeze_grid: bool,
    device: str,
) -> "KANLayer":
    """Build one pykan KANLayer with fair-protocol defaults."""
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
        noise_scale=0.05 if residual else 0.1,
        scale_base_mu=0.0,
        scale_base_sigma=scale_base_sigma,
        scale_sp=0.5 if residual else 1.0,
        base_fun=base_fun,
        grid_range=list(kan_grid_range),
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


class DeepKanHybrid(nn.Module):
    """Deep-2/3 hybrid: additive edges + residual KAN mixer stack + cat linear."""

    def __init__(
        self,
        n_continuous: int,
        n_cat_oh: int = 0,
        *,
        n_intervals: int = 6,
        degree: int = 3,
        lambda_s: float = 1e-2,
        lambda_r: float = 1e-6,
        kan_grid_range: Optional[Tuple[float, float]] = None,
        freeze_grid: bool = True,
        residual: bool = True,
        disable_silu: bool = True,
        lambda_phi: Optional[float] = None,
        depth: int = 2,
        hidden_width: int = 4,
        mixer_input: str = "phi",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        _require_pykan()
        depth = int(depth)
        if depth not in (2, 3):
            raise ValueError(f"depth must be 2 or 3, got {depth}")
        mi = str(mixer_input).strip().lower()
        if mi not in ("phi", "raw_scaled"):
            raise ValueError(
                f"mixer_input must be 'phi' or 'raw_scaled', got {mixer_input!r}"
            )
        self.mixer_input = mi
        # phi path: inputs ~O(1) after std scale → wide grid; raw_scaled: T(x)∈[-1,1]
        if kan_grid_range is None:
            kan_grid_range = (-1.0, 1.0) if mi == "raw_scaled" else (-5.0, 5.0)
        self.kan_grid_range = (float(kan_grid_range[0]), float(kan_grid_range[1]))
        self.n_continuous = int(n_continuous)
        self.n_cat_oh = int(n_cat_oh)
        self.n_intervals = int(n_intervals)
        self.degree = int(degree)
        self.lambda_s = float(lambda_s)
        self.lambda_r = float(lambda_r)
        # stronger ridge on interaction residual by default
        self.lambda_phi = float(lambda_r if lambda_phi is None else lambda_phi)
        self.freeze_grid = bool(freeze_grid)
        self.residual = bool(residual)
        self.disable_silu = bool(disable_silu)
        self.depth = depth
        self.hidden_width = int(hidden_width) if depth == 3 else 0
        self._device = device
        self._raw_scaler: Optional[Dict[str, np.ndarray]] = None

        self.spline = AdditiveSplineKAN(
            n_features=n_continuous,
            n_intervals=n_intervals,
            degree=degree,
            lambda_s=lambda_s,
            lambda_r=lambda_r,
            dtype=torch.float64,
        )

        # Mixer stack: depth-2 uses single KANLayer P→1 (attribute kan2 for
        # backward compatibility). depth-3 uses KAN [P, h, 1].
        self.kan_layers = nn.ModuleList()
        if depth == 2:
            self.kan2 = _make_kan_layer(
                n_continuous,
                1,
                n_intervals=n_intervals,
                degree=degree,
                residual=residual,
                disable_silu=disable_silu,
                kan_grid_range=self.kan_grid_range,
                freeze_grid=freeze_grid,
                device=device,
            )
            self.kan_layers.append(self.kan2)
            self.kan_hidden = None  # type: ignore
            self.kan_out = None  # type: ignore
        else:
            h = max(1, int(hidden_width))
            self.hidden_width = h
            self.kan_hidden = _make_kan_layer(
                n_continuous,
                h,
                n_intervals=n_intervals,
                degree=degree,
                residual=residual,
                disable_silu=disable_silu,
                kan_grid_range=self.kan_grid_range,
                freeze_grid=freeze_grid,
                device=device,
            )
            self.kan_out = _make_kan_layer(
                h,
                1,
                n_intervals=n_intervals,
                degree=degree,
                residual=residual,
                disable_silu=disable_silu,
                kan_grid_range=self.kan_grid_range,
                freeze_grid=freeze_grid,
                device=device,
            )
            self.kan_layers.extend([self.kan_hidden, self.kan_out])
            # alias for any code that still looks at .kan2
            self.kan2 = self.kan_out

        if n_cat_oh > 0:
            self.beta_cat = nn.Parameter(torch.zeros(n_cat_oh, dtype=torch.float64))
        else:
            self.register_parameter("beta_cat", None)

        self.register_buffer(
            "phi_scale",
            torch.ones(n_continuous, dtype=torch.float64),
        )
        self._phi_scale_fitted = False

    def fit_bounds(self, X_cont: np.ndarray) -> "DeepKanHybrid":
        self.spline.fit_bounds(X_cont)
        if self.mixer_input == "raw_scaled":
            self._raw_scaler = fit_affine_scaler(
                np.asarray(X_cont, dtype=np.float64),
                q_low=1.0,
                q_high=99.0,
                out_low=-1.0,
                out_high=1.0,
            )
        return self

    def transform_raw(self, X_cont: np.ndarray) -> np.ndarray:
        if self._raw_scaler is None:
            raise RuntimeError("raw scaler not fitted; call fit_bounds first")
        return apply_affine_scaler(X_cont, self._raw_scaler)

    def spline_design(self, X_cont: Tensor) -> Tensor:
        return self.spline.full_design_matrix(X_cont)

    def centered_phis(
        self,
        B_spline: Tensor,
        *,
        center_weights: Optional[Tensor] = None,
        train_means: Optional[Tensor] = None,
    ) -> Tensor:
        """Return centered φ_p, shape (N, P).

        If train_means is provided (length P), subtract those (for PA/test
        rows under train reference). Else center using center_weights on the
        same batch (training).
        """
        n_b = self.spline.n_basis_per_feature
        P = self.n_continuous
        phis = []
        for p in range(P):
            sl = slice(p * n_b, (p + 1) * n_b)
            phis.append(B_spline[:, sl] @ self.spline.coeffs[p])
        phis_t = torch.stack(phis, dim=1)
        if train_means is not None:
            return phis_t - train_means
        if center_weights is None:
            w = torch.ones(B_spline.shape[0], dtype=B_spline.dtype, device=B_spline.device)
        else:
            w = center_weights.to(dtype=B_spline.dtype, device=B_spline.device)
        w = w / w.sum()
        means = (w[:, None] * phis_t).sum(dim=0)
        return phis_t - means

    def set_phi_scale_from_phis(self, phis: Tensor) -> None:
        with torch.no_grad():
            s = phis.detach().std(dim=0)
            s = torch.clamp(s, min=1e-2)
            self.phi_scale.copy_(s)
            self._phi_scale_fitted = True

    def phi_for_kan(self, phis: Tensor) -> Tensor:
        scale = self.phi_scale.to(dtype=phis.dtype, device=phis.device)
        return phis / scale

    def _mixer_forward(self, z: Tensor) -> Tensor:
        """Apply Deep-2 Φ(z) or Deep-3 Ψ(Φ(z)); z shape (N, P) float32."""
        z = z.to(dtype=torch.float32)
        if self.depth == 2:
            return self.kan2(z)[0].squeeze(-1).to(dtype=torch.float64)
        h = self.kan_hidden(z)[0]
        return self.kan_out(h)[0].squeeze(-1).to(dtype=torch.float64)

    def phi_interaction(self, phis: Tensor) -> Tensor:
        """Apply residual mixer on scaled φ (mixer_input='phi')."""
        return self._mixer_forward(self.phi_for_kan(phis))

    def raw_interaction(self, z_raw: Tensor) -> Tensor:
        """Apply residual mixer on T(x) (mixer_input='raw_scaled')."""
        return self._mixer_forward(z_raw)

    def eta_from_parts(
        self,
        B_spline: Tensor,
        X_cat: Optional[Tensor],
        *,
        center_weights: Optional[Tensor] = None,
        train_means: Optional[Tensor] = None,
        z_raw: Optional[Tensor] = None,
    ) -> Tensor:
        phis = self.centered_phis(
            B_spline, center_weights=center_weights, train_means=train_means
        )
        if self.mixer_input == "phi":
            if not self._phi_scale_fitted:
                self.set_phi_scale_from_phis(phis)
            inter = self.phi_interaction(phis)
        else:
            if z_raw is None:
                raise ValueError("z_raw required when mixer_input='raw_scaled'")
            inter = self.raw_interaction(z_raw)
        if self.residual:
            eta = phis.sum(dim=1) + inter
        else:
            eta = inter
        if self.beta_cat is not None and X_cat is not None and X_cat.shape[1] > 0:
            eta = eta + X_cat @ self.beta_cat
        return eta

    def kan_mixer_ridge(self) -> Tensor:
        pen = torch.zeros((), dtype=torch.float64)
        for layer in self.kan_layers:
            for _, p in layer.named_parameters():
                if not p.requires_grad:
                    continue
                pen = pen + (p.to(dtype=torch.float64) ** 2).sum()
        return pen

    def kan2_ridge(self) -> Tensor:
        """Backward-compatible alias."""
        return self.kan_mixer_ridge()

    def penalty(self) -> Tensor:
        pen = self.lambda_s * self.spline.smoothness_penalty()
        pen = pen + self.lambda_r * self.spline.ridge_penalty()
        pen = pen + self.lambda_phi * self.kan_mixer_ridge()
        if self.beta_cat is not None:
            pen = pen + self.lambda_r * (self.beta_cat ** 2).sum()
        return pen

    def export_manifest(self) -> Dict[str, object]:
        u = "phi" if self.mixer_input == "phi" else "T(x)"
        if self.depth == 2:
            arch = (
                f"deep2_residual_{self.mixer_input}_kanlayer"
                if self.residual
                else f"deep2_{self.mixer_input}_vector_to_kanlayer"
            )
            formula = (
                f"eta = sum(phi) + Phi({u}) + X_cat @ beta"
                if self.residual
                else f"eta = Phi({u}) + X_cat @ beta"
            )
            layer_desc = "pykan.KANLayer(P, 1)"
        else:
            arch = (
                f"deep3_residual_{self.mixer_input}_kan_stack"
                if self.residual
                else f"deep3_{self.mixer_input}_vector_to_kan_stack"
            )
            formula = (
                f"eta = sum(phi) + Psi(Phi({u})) + X_cat @ beta"
                if self.residual
                else f"eta = Psi(Phi({u})) + X_cat @ beta"
            )
            layer_desc = f"pykan [P, h={self.hidden_width}, 1]"
        return {
            "architecture": arch,
            "formula": formula,
            "depth": self.depth,
            "hidden_width": self.hidden_width if self.depth == 3 else None,
            "mixer_input": self.mixer_input,
            "kan_grid_range": list(self.kan_grid_range),
            "layer1": "AdditiveSplineKAN edges (auditable B-spline)",
            "mixer": layer_desc,
            "layer2": layer_desc,  # keep key for older readers
            "residual": self.residual,
            "disable_silu": self.disable_silu,
            "pykan_base_activation": "Identity" if self.disable_silu else "SiLU",
            "n_continuous": self.n_continuous,
            "n_cat_oh": self.n_cat_oh,
            "n_intervals": self.n_intervals,
            "degree": self.degree,
            "lambda_s": self.lambda_s,
            "lambda_r": self.lambda_r,
            "lambda_phi": self.lambda_phi,
            "freeze_grid": self.freeze_grid,
            "pykan_entropy_penalty": False,
            "pykan_sparsity_penalty": False,
            "grid_update": False if self.freeze_grid else "enabled",
            "phi_scale": as_numpy_f64(self.phi_scale),
        }


def _shrink_layer(layer, rng: np.random.Generator, residual_small: bool) -> None:
    with torch.no_grad():
        if residual_small:
            if hasattr(layer, "coef"):
                layer.coef.mul_(0.05)
                layer.coef.add_(
                    torch.as_tensor(
                        rng.normal(0, 0.001, size=tuple(layer.coef.shape)),
                        dtype=layer.coef.dtype,
                        device=layer.coef.device,
                    )
                )
            if hasattr(layer, "scale_sp"):
                layer.scale_sp.mul_(0.1)
        else:
            if hasattr(layer, "coef"):
                layer.coef.add_(
                    torch.as_tensor(
                        rng.normal(0, 0.01, size=tuple(layer.coef.shape)),
                        dtype=layer.coef.dtype,
                        device=layer.coef.device,
                    )
                )


def init_deep_kan(model: DeepKanHybrid, seed: int, *, residual_small: bool = True) -> None:
    rng = np.random.default_rng(seed)
    model.spline.set_coeffs_from_numpy(
        rng.normal(0, 0.01, size=tuple(model.spline.coeffs.shape))
    )
    if model.beta_cat is not None:
        with torch.no_grad():
            model.beta_cat.copy_(
                torch.as_tensor(
                    rng.normal(0, 0.01, size=(model.n_cat_oh,)), dtype=torch.float64
                )
            )
    # Residual philosophy: mixer starts near 0 so η ≈ Σφ.
    # Deep-2: shrink the single P→1 layer.
    # Deep-3: keep hidden layer moderately expressive so Φ(φ) is not dead;
    # only shrink the final Ψ so the residual starts near zero. Double-shrinking
    # both layers collapses gradients (empirically ΔAUC≡0).
    if residual_small and model.residual and model.depth == 3:
        # moderate hidden (not residual_small)
        _shrink_layer(model.kan_hidden, rng, residual_small=False)
        with torch.no_grad():
            if hasattr(model.kan_hidden, "coef"):
                model.kan_hidden.coef.mul_(0.5)
            # leave scale_sp near construction residual default (~0.5)
        _shrink_layer(model.kan_out, rng, residual_small=True)
    else:
        for layer in model.kan_layers:
            _shrink_layer(layer, rng, residual_small and model.residual)


def _predict_scores(
    model: DeepKanHybrid,
    B_tr: Tensor,
    B_te: Tensor,
    Xk_te: Optional[Tensor],
    weights: Tensor,
    z_raw_te: Optional[Tensor] = None,
) -> np.ndarray:
    with torch.no_grad():
        n_b = model.spline.n_basis_per_feature
        P = model.n_continuous
        ptr = torch.stack(
            [B_tr[:, p * n_b : (p + 1) * n_b] @ model.spline.coeffs[p] for p in range(P)],
            dim=1,
        )
        ww = weights / weights.sum()
        means = (ww[:, None] * ptr).sum(dim=0)
        eta = model.eta_from_parts(
            B_te, Xk_te, train_means=means, z_raw=z_raw_te
        )
    return as_numpy_f64(eta)


def _adaptive_budget(n_po: int, base_lbfgs: int, base_adam: int) -> Tuple[int, int]:
    """More optimization for rare species; slightly less for very large n_po."""
    if n_po < 30:
        return int(base_lbfgs * 1.5), int(base_adam * 1.5)
    if n_po < 100:
        return base_lbfgs, base_adam
    return base_lbfgs, max(base_adam // 2, 40)


def fit_deep_kan_ipp(
    X_cont_train: np.ndarray,
    X_cat_train: np.ndarray,
    y_train: np.ndarray,
    X_cont_test: np.ndarray,
    X_cat_test: np.ndarray,
    *,
    n_intervals: int = 6,
    degree: int = 3,
    lambda_s: float = 1e-2,
    lambda_r: float = 1e-6,
    lambda_phi: Optional[float] = None,
    seed: int = 0,
    steps: int = 80,
    lr: float = 0.05,
    freeze_grid: bool = True,
    residual: bool = True,
    disable_silu: bool = True,
    warm_start_additive: bool = True,
    optimizer: str = "lbfgs",
    lbfgs_steps: int = 20,
    lbfgs_lr: float = 0.25,
    freeze_edges_after_warmstart: bool = False,
    adaptive_budget: bool = True,
    warm_start_state: Optional[Dict] = None,
    lbfgs_max_iter: int = 10,
    depth: int = 2,
    hidden_width: int = 4,
    mixer_input: str = "phi",
    record_loss_history: bool = True,
) -> Tuple[DeepKanHybrid, np.ndarray, Dict]:
    """Train Deep-2 or Deep-3 with IPP NLL.

    Recommended fair defaults (Phase 4 v2 / Closure):
      residual=True, disable_silu=True, warm_start_additive=True, optimizer='lbfgs'

    Pass ``warm_start_state`` from a prior additive fit to avoid a second
    additive LBFGS (keys: spline_coeffs, beta_cat optional).

    ``mixer_input``: 'phi' (legacy) or 'raw_scaled' (T(x); Closure Gap II).
    """
    import time

    from kanmaxent.losses.ipp import ipp_nll
    from kanmaxent.nceas_fit import fit_hybrid_ipp

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_po = int(np.sum(np.asarray(y_train) > 0))
    if adaptive_budget:
        lbfgs_steps, steps = _adaptive_budget(n_po, lbfgs_steps, steps)

    n_cat = int(X_cat_train.shape[1]) if getattr(X_cat_train, "ndim", 0) == 2 else 0
    # slightly stronger ridge on Φ for residual models
    if lambda_phi is None:
        lambda_phi = 1e-4 if residual else lambda_r

    model = DeepKanHybrid(
        X_cont_train.shape[1],
        n_cat,
        n_intervals=n_intervals,
        degree=degree,
        lambda_s=lambda_s,
        lambda_r=lambda_r,
        lambda_phi=lambda_phi,
        freeze_grid=freeze_grid,
        residual=residual,
        disable_silu=disable_silu,
        depth=depth,
        hidden_width=hidden_width,
        mixer_input=mixer_input,
    )
    model.fit_bounds(X_cont_train)
    init_deep_kan(model, seed, residual_small=residual)

    Xc_tr = torch.as_tensor(X_cont_train, dtype=torch.float64)
    Xc_te = torch.as_tensor(X_cont_test, dtype=torch.float64)
    Xk_tr = torch.as_tensor(X_cat_train, dtype=torch.float64) if n_cat else None
    Xk_te = torch.as_tensor(X_cat_test, dtype=torch.float64) if n_cat else None
    B_tr = model.spline_design(Xc_tr).detach()
    B_te = model.spline_design(Xc_te).detach()
    counts = torch.as_tensor(y_train, dtype=torch.float64)
    weights = torch.ones(len(y_train), dtype=torch.float64)

    z_raw_tr: Optional[Tensor] = None
    z_raw_te: Optional[Tensor] = None
    if model.mixer_input == "raw_scaled":
        z_raw_tr = torch.as_tensor(
            model.transform_raw(X_cont_train), dtype=torch.float64
        )
        z_raw_te = torch.as_tensor(
            model.transform_raw(X_cont_test), dtype=torch.float64
        )

    warm_meta: Dict = {}
    if warm_start_state is not None:
        with torch.no_grad():
            model.spline.coeffs.copy_(
                torch.as_tensor(warm_start_state["spline_coeffs"], dtype=torch.float64)
            )
            if model.beta_cat is not None and warm_start_state.get("beta_cat") is not None:
                model.beta_cat.copy_(
                    torch.as_tensor(warm_start_state["beta_cat"], dtype=torch.float64)
                )
        warm_meta = {"from": "provided_state"}
        if freeze_edges_after_warmstart:
            model.spline.coeffs.requires_grad_(False)
            if model.beta_cat is not None:
                model.beta_cat.requires_grad_(False)
    elif warm_start_additive:
        add_model, _, warm_meta = fit_hybrid_ipp(
            X_cont_train,
            X_cat_train,
            y_train,
            X_cont_test,
            X_cat_test,
            n_intervals=n_intervals,
            degree=degree,
            lambda_s=lambda_s,
            lambda_r=lambda_r,
            seed=seed,
            steps=15,
        )
        with torch.no_grad():
            model.spline.coeffs.copy_(add_model.spline.coeffs.detach())
            if model.beta_cat is not None and add_model.beta_cat is not None:
                model.beta_cat.copy_(add_model.beta_cat.detach())
        if freeze_edges_after_warmstart:
            model.spline.coeffs.requires_grad_(False)
            if model.beta_cat is not None:
                model.beta_cat.requires_grad_(False)

    with torch.no_grad():
        ph0 = model.centered_phis(B_tr, center_weights=weights)
        if model.mixer_input == "phi":
            model.set_phi_scale_from_phis(ph0)

    last_loss = float("nan")
    last_grad_norm = float("nan")
    loss_history: List[float] = []
    opt_name = optimizer.lower().strip()

    def _loss() -> Tensor:
        eta = model.eta_from_parts(
            B_tr, Xk_tr, center_weights=weights, z_raw=z_raw_tr
        )
        return ipp_nll(eta, counts, weights, model.penalty(), per_presence=False)

    params = [p for p in model.parameters() if p.requires_grad]

    if opt_name in ("adam", "adam_then_lbfgs"):
        opt = torch.optim.Adam(params, lr=lr, weight_decay=0.0)
        for _ in range(int(steps)):
            opt.zero_grad()
            loss = _loss()
            if not torch.isfinite(loss):
                break
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            last_grad_norm = float(gn.detach().cpu()) if torch.is_tensor(gn) else float(gn)
            opt.step()
            last_loss = float(loss.detach().cpu())
            if record_loss_history:
                loss_history.append(last_loss)

    if opt_name in ("lbfgs", "adam_then_lbfgs"):
        # refresh param list after possible freezes
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
            if record_loss_history:
                loss_history.append(last_loss)
            # approximate grad norm after last closure
            with torch.no_grad():
                sq = 0.0
                for p in params:
                    if p.grad is not None:
                        sq += float((p.grad.detach() ** 2).sum().cpu())
                last_grad_norm = float(np.sqrt(sq)) if sq > 0 else last_grad_norm

    scores = _predict_scores(model, B_tr, B_te, Xk_te, weights, z_raw_te=z_raw_te)

    param_norm = 0.0
    with torch.no_grad():
        for p in model.parameters():
            param_norm += float((p.detach() ** 2).sum().cpu())
    param_norm = float(np.sqrt(param_norm))

    # subsample long histories for status JSON
    hist_out = loss_history
    if len(hist_out) > 200:
        idx = np.linspace(0, len(hist_out) - 1, 200).astype(int)
        hist_out = [loss_history[i] for i in idx]

    meta = {
        "runtime_s": time.perf_counter() - t0,
        "converged": bool(np.isfinite(scores).all() and np.isfinite(last_loss)),
        "lambda_s": lambda_s,
        "lambda_phi": lambda_phi,
        "final_loss": last_loss,
        "last_grad_norm": last_grad_norm,
        "param_l2_norm": param_norm,
        "loss_history": hist_out,
        "steps_adam": steps if opt_name in ("adam", "adam_then_lbfgs") else 0,
        "steps_lbfgs": lbfgs_steps if opt_name in ("lbfgs", "adam_then_lbfgs") else 0,
        "optimizer": opt_name,
        "lr": lr,
        "lbfgs_lr": lbfgs_lr,
        "residual": residual,
        "disable_silu": disable_silu,
        "mixer_input": model.mixer_input,
        "warm_start_additive": warm_start_additive,
        "freeze_edges_after_warmstart": freeze_edges_after_warmstart,
        "depth": depth,
        "hidden_width": hidden_width if depth == 3 else None,
        "n_po": n_po,
        "warm_start_runtime_s": warm_meta.get("runtime_s"),
        "model_manifest": model.export_manifest(),
    }
    return model, scores, meta
