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
    if config.technique.name == "adalomo":
        optimizer = AdaLomo(model, lr=lr, clip_grad_norm=None,
                            weight_decay=config.hparams.weight_decay)
    else:
        optimizer = Lomo(model, lr=lr, clip_grad_norm=None,
                         weight_decay=config.hparams.weight_decay)

    def fused(loss):
        optimizer.fused_backward(loss, lr)

    return run_manual_loop(config, model, tok, optimizer,
                           label=config.technique.name, fused_backward=fused)
