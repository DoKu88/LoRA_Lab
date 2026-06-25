"""Reconstruction objective (T2L Eq. 6) — regress generated ΔW onto a library LoRA.

Cheap warmup / plumbing: no base-model forward pass. We regress the generated
ΔW = scaling·(B·A) onto a target adapter's ΔW for the same target keys. Used in
Sprint 1 (overfit-a-toy-target plumbing) and Sprint 3 (warmup init over the
train-split library). The real generalization comes from SFT (Eq. 5), not this.

We use the **relative Frobenius error** ``‖ΔW_gen − ΔW_tgt‖_F / ‖ΔW_tgt‖_F`` rather
than mean-reduction L1: a LoRA ΔW has a healthy Frobenius norm (~1) but tiny
per-element values spread over millions of entries, so a *mean* over elements
collapses both the loss and its gradient to ~0 (vanishing-signal). The relative
Frobenius error is 1.0 at init (generated ΔW = 0), scale-robust, and has a usable
gradient throughout. (Note: target ΔW is formed with the *generator's* scaling;
exact for the alpha/r=16 library adapters, a mild magnitude approximation for the
r=43 ones — direction is unaffected.)
"""

from __future__ import annotations

import torch

from .model import delta_w


def reconstruction_loss(
    generated: dict[str, tuple[torch.Tensor, torch.Tensor]],
    target: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    scaling: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean relative Frobenius error between generated and target ΔW over shared keys.

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
        total = total + (delta_gen - delta_target).norm() / (delta_target.norm() + eps)
    return total / len(keys)
