"""LOMO / AdaLOMO strategies (Sprint 4) — VRAM-direct, fused backward+update.

LOMO fuses the gradient computation with the parameter update in a single
backward pass, so full gradients and optimizer state are never materialized
(SGD-like footprint). AdaLOMO adds Adafactor-style adaptive state back.

These don't use ``optimizer.step()`` — the update happens inside
``fused_backward(loss, lr)`` — so they drive the shared loop via its
``fused_backward`` hook. Gradient clipping is left off (LOMO's clipped path
needs a second backward pass, doubling compute); noted in the findings.
"""

from __future__ import annotations

from ...config import RunConfig
from ._common import load_bf16_model, run_manual_loop


def run_lomo(config: RunConfig) -> dict:
    from lomo_optim import AdaLomo, Lomo

    model, tok = load_bf16_model(config)
    lr = config.hparams.lr
    clip = config.technique.lomo_clip_grad_norm
    clip_arg = clip if clip and clip > 0 else None
    Opt = AdaLomo if config.technique.name == "adalomo" else Lomo
    optimizer = Opt(model, lr=lr, clip_grad_norm=clip_arg,
                    weight_decay=config.hparams.weight_decay)

    if clip_arg:
        # LOMO's clipped path is two passes over the SAME graph: grad_norm()
        # does backward(retain_graph=True) to compute the clip coefficient (and
        # clears grads), then fused_backward() reuses the retained graph to apply
        # the clipped update. ~2x backward cost, but no second forward.
        def fused(loss):
            optimizer.grad_norm(loss)
            optimizer.fused_backward(loss, lr)
    else:
        def fused(loss):
            optimizer.fused_backward(loss, lr)

    label = config.technique.name + ("_clip" if clip_arg else "")
    return run_manual_loop(config, model, tok, optimizer,
                           label=label, fused_backward=fused)
