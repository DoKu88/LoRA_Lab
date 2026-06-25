"""Reconstruction objective (T2L Eq. 6) — L1 on ΔW vs a target library LoRA.

Cheap warmup / plumbing: no base-model forward pass. We regress the generated
ΔW = scaling·(B·A) onto a target adapter's ΔW for the same target keys. Used in
Sprint 1 (overfit-a-toy-target plumbing) and Sprint 3 (warmup init over the
train-split library). The real generalization comes from SFT (Eq. 5), not this.
"""

from __future__ import annotations

import torch

from .model import delta_w


def reconstruction_loss(
    generated: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    scaling: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Smooth-L1 between generated and target ΔW over shared keys.

    ``generated``/``target`` map key -> (A, B). Only keys present in both are
    scored (so a target adapter covering a subset of targets is fine).
    """
    keys = [k for k in generated if k in target]
    if not keys:
        raise ValueError("no shared target keys between generated and target adapters")
    total = generated[keys[0]][0].new_zeros(())
    for k in keys:
        ga, gb = generated[k]
        ta, tb = target[k]
        dw_g = delta_w(ga, gb, scaling)
        dw_t = delta_w(ta, tb, scaling)
        total = total + torch.nn.functional.smooth_l1_loss(dw_g, dw_t, reduction=reduction)
    return total / len(keys)
