"""GaLore / Q-GaLore strategies (Sprint 4) — VRAM-direct, on-GPU.

GaLore low-rank-projects the optimizer state for the large 2D weight matrices
(attention + MLP linears), collapsing the optimizer-memory pool while keeping
*full-parameter* updates. It's a drop-in optimizer, so it rides the shared
manual loop. Q-GaLore adds 8-bit optimizer state on top (GaLoreAdamW8bit) — the
optimizer-side of Q-GaLore; full Q-GaLore also quantizes weights (separate
package), noted as a caveat in the findings.
"""

from __future__ import annotations

import torch
from transformers import get_scheduler

from ...config import RunConfig
from ._common import _num_opt_steps, load_bf16_model, run_manual_loop

# GaLore projects the big 2D linears; embeddings/norms/lm_head train normally.
_GALORE_TARGET_KEYS = ("self_attn", "mlp")


def _galore_param_groups(model, config: RunConfig):
    galore_params, regular = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(
            k in name for k in _GALORE_TARGET_KEYS
        ):
            galore_params.append(module.weight)
            if module.bias is not None:
                regular.append(module.bias)
    galore_ids = {id(p) for p in galore_params}
    regular += [p for p in model.parameters()
                if p.requires_grad and id(p) not in galore_ids]
    t = config.technique
    return [
        {"params": regular},
        {"params": galore_params, "rank": t.galore_rank,
         "update_proj_gap": t.galore_update_proj_gap,
         "scale": t.galore_scale, "proj_type": t.galore_proj_type},
    ]


def run_galore(config: RunConfig) -> dict:
    from galore_torch import GaLoreAdamW, GaLoreAdamW8bit

    model, tok = load_bf16_model(config)
    groups = _galore_param_groups(model, config)
    Opt = GaLoreAdamW8bit if config.technique.name == "qgalore" else GaLoreAdamW
    optimizer = Opt(groups, lr=config.hparams.lr,
                    weight_decay=config.hparams.weight_decay)
    total = _num_opt_steps(
        config.hparams.max_train_samples if config.hparams.max_train_samples > 0 else 1024,
        config,
    )
    scheduler = get_scheduler(
        config.hparams.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=int(config.hparams.warmup_ratio * total),
        num_training_steps=total,
    )
    return run_manual_loop(config, model, tok, optimizer,
                           label=config.technique.name, scheduler=scheduler)
