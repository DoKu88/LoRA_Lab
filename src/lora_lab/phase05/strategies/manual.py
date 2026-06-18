"""On-GPU full-FT strategies that ride the shared manual loop.

`baseline` is the working feasibility recipe from the Sprint 2 smoke test:
bf16 Mistral-7B on the GPU + (paged 8-bit) AdamW + gradient checkpointing,
~27 GB VRAM at ~1.7 s/step. With ``levers.use_8bit_adam=False`` it falls back to
fp32 AdamW (expected to OOM the 32 GB GPU on 7B — a real fits=no data point and
the 8-bit ablation's other arm).
"""

from __future__ import annotations

import torch
from transformers import get_scheduler

from ...config import RunConfig
from ._common import _num_opt_steps, load_bf16_model, run_manual_loop


def _build_adamw(config: RunConfig, params):
    lr = config.hparams.lr
    wd = config.hparams.weight_decay
    if config.levers.use_8bit_adam:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def run_baseline(config: RunConfig) -> dict:
    model, tok = load_bf16_model(config)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _build_adamw(config, params)
    total_steps = _num_opt_steps(config.hparams.max_train_samples
                                 if config.hparams.max_train_samples > 0 else 1024,
                                 config)
    scheduler = get_scheduler(
        config.hparams.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=int(config.hparams.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    label = "paged8bit" if config.levers.use_8bit_adam else "fp32-adamw"
    return run_manual_loop(config, model, tok, optimizer,
                           label=label, scheduler=scheduler)
