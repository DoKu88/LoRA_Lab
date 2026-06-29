"""The hypernetwork — (task-description embedding) -> per-target LoRA A/B.

A shared trunk over the task embedding plus learned layer and module embeddings
feed a per-target output head that emits LoRA factors (A:(rank,in), B:(out,rank))
for every targeted Linear, in a single forward pass. The B path is zero-init so
ΔW = 0 at start (the no-op invariant); the A path varies with the task embedding
from the start. ``delta_w`` forms the effective weight delta ΔW = scaling·(B·A).
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from .heads import HEADS


def delta_w(lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """ΔW = scaling · (B · A), shape (out, in) — the effective weight delta."""
    return scaling * (lora_b @ lora_a)


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _parse_key(key: str) -> tuple[int, str]:
    """('...layers.5.self_attn.q_proj') -> (layer_idx=5, module='q_proj')."""
    match = _LAYER_RE.search(key)
    layer = int(match.group(1)) if match else 0
    module = key.rsplit(".", 1)[-1]
    return layer, module


class HyperLoRAGenerator(nn.Module):
    """Generator: (task-description embedding) -> per-target LoRA A/B.

    A **shared trunk** over the task embedding plus learned **layer** and
    **module** embeddings feeds a per-target output head (default low-rank A/B —
    generates real A/B through a bottleneck, ~151 M on Mistral-7B; the smaller
    VeRA rung can't reconstruct a target ΔW, so it is not the default).
    Cross-layer/module structure is shared through the trunk and the embeddings,
    not one independent generator per target. Forward contract:
    ``task_emb -> {key: (A:(rank,in), B:(out,rank))}``, with the B path zero-init
    so ΔW = 0 at start (the no-op invariant). Conditioning is live at init (the A
    path varies with the task embedding); the B path opens up during training.
    """

    def __init__(
        self,
        target_specs: dict[str, tuple[int, int]],
        task_dim: int,
        *,
        rank: int = 16,
        alpha: int = 32,
        parameterization: str = "lowrank",
        layer_dim: int = 16,
        module_dim: int = 8,
        trunk_hidden: int = 128,
        head_kwargs: dict | None = None,
    ):
        super().__init__()
        self.keys = list(target_specs)
        self.rank = rank
        self.scaling = alpha / rank
        self.parameterization = parameterization

        parsed = {key: _parse_key(key) for key in self.keys}
        n_layers = max((layer for layer, _ in parsed.values()), default=0) + 1
        modules = sorted({module for _, module in parsed.values()})
        self.module_ids = {module: index for index, module in enumerate(modules)}

        self.trunk = nn.Sequential(
            nn.Linear(task_dim, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
        )
        self.layer_emb = nn.Embedding(n_layers, layer_dim)
        self.module_emb = nn.Embedding(len(modules), module_dim)
        cond_dim = trunk_hidden + layer_dim + module_dim

        head_cls = HEADS[parameterization]
        self.heads = nn.ModuleDict()
        self._meta = {}
        for index, key in enumerate(self.keys):
            in_features, out_features = target_specs[key]
            self.heads[self._param_key(index)] = head_cls(
                cond_dim, in_features, out_features, rank, **(head_kwargs or {})
            )
            self._meta[self._param_key(index)] = parsed[key]

    @staticmethod
    def _param_key(index: int) -> str:
        return f"t{index}"

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, task_emb: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        task_hidden = self.trunk(task_emb)
        device = task_hidden.device  # index tensors must match the (possibly cuda) embeddings
        adapter: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, key in enumerate(self.keys):
            layer, module = self._meta[self._param_key(index)]
            conditioning = torch.cat([
                task_hidden,
                self.layer_emb(torch.tensor(layer, device=device)),
                self.module_emb(torch.tensor(self.module_ids[module], device=device)),
            ], dim=-1)
            adapter[key] = self.heads[self._param_key(index)](conditioning)
        return adapter
