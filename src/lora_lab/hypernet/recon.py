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
    keys = [key for key in generated if key in target]
    if not keys:
        raise ValueError("no shared target keys between generated and target adapters")
    total = generated[keys[0]][0].new_zeros(())
    for key in keys:
        gen_a, gen_b = generated[key]
        target_a, target_b = target[key]
        delta_gen = delta_w(gen_a, gen_b, scaling)
        delta_target = delta_w(target_a, target_b, scaling)
        total = total + torch.nn.functional.smooth_l1_loss(delta_gen, delta_target, reduction=reduction)
    return total / len(keys)
