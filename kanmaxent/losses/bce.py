"""BCE ablation for presence vs background (pseudo-absence).

Phase 1: unweighted BCEWithLogits.
Phase 2: optional pos_weight = n_background / n_presence (Valavi-style class balance).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor


def bce_with_logits_nll(
    logits: Tensor,
    labels: Tensor,
    penalty: Optional[Tensor] = None,
    *,
    pos_weight: Optional[Union[float, Tensor]] = None,
    reduction: str = "mean",
) -> Tensor:
    """Binary cross-entropy with logits + optional explicit penalty Omega.

    Parameters
    ----------
    pos_weight
        If set, weight for the positive class (broadcastable scalar or tensor).
        Phase 2 default: n_background / n_presence.
    """
    labels = labels.to(dtype=logits.dtype, device=logits.device)
    kw = {"reduction": reduction}
    if pos_weight is not None:
        pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        kw["pos_weight"] = pw
    loss = F.binary_cross_entropy_with_logits(logits, labels, **kw)
    if penalty is not None:
        loss = loss + penalty
    return loss
