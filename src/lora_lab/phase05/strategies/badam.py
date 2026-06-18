"""BAdam strategy (Sprint 5) — block-coordinate, VRAM-direct.

BAdam keeps grads + optimizer state for only ONE transformer block at a time,
cycling through every block over the run (full-parameter coverage, not
simultaneous). It wraps a base AdamW and behaves like a normal optimizer — the
block switch happens inside ``.step()`` — so it rides the shared loop.

Because all-but-the-active block are frozen at any instant, the loop's
full-param assertion is disabled here (coverage is over the run).
"""

from __future__ import annotations

import torch
from transformers import get_scheduler

from ...config import RunConfig
from ._common import _num_opt_steps, load_bf16_model, run_manual_loop


def run_badam(config: RunConfig) -> dict:
    from badam import BlockOptimizer

    model, tok = load_bf16_model(config)
    # Combinable with 8-bit: BAdam already shrinks optimizer state to one block;
    # an 8-bit base optimizer shrinks that block's state further still.
    if config.levers.use_8bit_adam:
        import bitsandbytes as bnb

        base = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.hparams.lr,
            weight_decay=config.hparams.weight_decay,
        )
    else:
        base = torch.optim.AdamW(
            model.parameters(), lr=config.hparams.lr,
            weight_decay=config.hparams.weight_decay,
        )
    optimizer = BlockOptimizer(
        base_optimizer=base,
        named_parameters_list=list(model.named_parameters()),
        switch_block_every=config.technique.badam_switch_every,
        switch_mode=config.technique.badam_switch_mode,
        include_embedding=False,
        include_lm_head=False,
        verbose=0,
    )
    total = _num_opt_steps(
        config.hparams.max_train_samples if config.hparams.max_train_samples > 0 else 1024,
        config,
    )
    scheduler = get_scheduler(
        config.hparams.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=int(config.hparams.warmup_ratio * total),
        num_training_steps=total,
    )
    return run_manual_loop(config, model, tok, optimizer, label="badam",
                           scheduler=scheduler, full_param_check=False)
