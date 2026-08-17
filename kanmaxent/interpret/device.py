"""Device resolution for Phase 5 curve fits (CPU-first LBFGS policy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class DevicePolicy:
    """Prefer CPU for LBFGS; optional CUDA with per-species fallback hooks."""

    request: str = "cpu"  # cpu | cuda | auto
    prefer_cpu_lbfgs: bool = True
    bootstrap_device: str = "inherit"  # inherit | cpu | cuda
    fallback_events: List[str] = field(default_factory=list)

    def resolve_fit_device(self) -> torch.device:
        req = self.request.lower().strip()
        if req == "auto":
            req = "cuda" if torch.cuda.is_available() else "cpu"
        if req == "cuda" and not torch.cuda.is_available():
            self.fallback_events.append("cuda_requested_but_unavailable")
            req = "cpu"
        if self.prefer_cpu_lbfgs:
            if req == "cuda":
                self.fallback_events.append("prefer_cpu_lbfgs_forced_cpu")
            return torch.device("cpu")
        return torch.device(req)

    def resolve_bootstrap_device(self, fit_device: torch.device) -> torch.device:
        b = self.bootstrap_device.lower().strip()
        if b == "inherit":
            return fit_device
        if b == "auto":
            b = "cuda" if torch.cuda.is_available() else "cpu"
        if b == "cuda" and not torch.cuda.is_available():
            self.fallback_events.append("bootstrap_cuda_unavailable")
            return torch.device("cpu")
        if b == "cuda" and self.prefer_cpu_lbfgs:
            self.fallback_events.append("bootstrap_prefer_cpu_lbfgs")
            return torch.device("cpu")
        return torch.device(b)

    def force_cpu(self, reason: str) -> None:
        self.fallback_events.append(reason)

    def to_manifest(self) -> Dict[str, Any]:
        return {
            "request": self.request,
            "prefer_cpu_lbfgs": self.prefer_cpu_lbfgs,
            "bootstrap_device": self.bootstrap_device,
            "fallback_events": list(self.fallback_events),
            "cuda_available": bool(torch.cuda.is_available()),
        }


def parse_device_policy(
    device: str = "cpu",
    *,
    prefer_cpu_lbfgs: bool = True,
    bootstrap_device: str = "inherit",
) -> DevicePolicy:
    return DevicePolicy(
        request=device,
        prefer_cpu_lbfgs=prefer_cpu_lbfgs,
        bootstrap_device=bootstrap_device,
    )
