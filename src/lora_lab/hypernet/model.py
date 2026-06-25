"""The hypernetwork (Sprint-1 stub) — (task embedding) -> per-target LoRA A/B.

This is the *plumbing* stub: it produces correctly-shaped, task-conditioned A/B
for every target Linear and is fully differentiable, so Sprint 1 can validate the
generate -> apply -> backprop loop. It is **not** the final architecture — Sprint 2
replaces the conditioning/heads (shared trunk + layer/module embeddings, and the
output-parameterization decision: VeRA-style vs low-rank vs full A/B).

Design (cheap, so it runs on a tiny base on CPU): per target key we hold base
factors ``base_a:(rank,in)`` (small random) and ``base_b:(out,rank)`` (**zero-init**, so
ΔW = 0 at start — the no-op-adapter invariant). A small gate maps the task
embedding to a per-(key, rank) modulation, so different descriptions yield
different adapters and base_b can train away from zero. Real conditioning power comes
in Sprint 2.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from .heads import HEADS


class HyperLoRA(nn.Module):
    def __init__(
        self,
        target_specs: dict[str, tuple[int, int]],
        task_dim: int,
        *,
        rank: int = 16,
        alpha: int = 32,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.keys = list(target_specs)
        self.rank = rank
        self.scaling = alpha / rank
        self.specs = dict(target_specs)

        # Per-target base factors. base_a small-random, base_b zero (=> ΔW=0 at init).
        self.base_a = nn.ParameterDict()
        self.base_b = nn.ParameterDict()
        for index, key in enumerate(self.keys):
            in_features, out_features = target_specs[key]
            self.base_a[self._param_key(index)] = nn.Parameter(torch.randn(rank, in_features) * init_scale)
            self.base_b[self._param_key(index)] = nn.Parameter(torch.zeros(out_features, rank))

        # Task -> per-(key, rank) modulation gate. Cheap: one Linear.
        self.gate = nn.Linear(task_dim, len(self.keys) * rank)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _param_key(index: int) -> str:
        # ParameterDict keys can't contain '.', so index by position.
        return f"t{index}"

    def num_params(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def forward(self, task_emb: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """task_emb: (task_dim,) -> {key: (A:(rank,in), B:(out,rank))}."""
        gate_modulation = self.gate(task_emb).view(len(self.keys), self.rank)  # (num_keys, rank)
        adapter: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, key in enumerate(self.keys):
            modulation = 1.0 + gate_modulation[index]                          # (rank,) per-rank modulation
            lora_a = self.base_a[self._param_key(index)] * modulation.unsqueeze(1)  # (rank, in)
            lora_b = self.base_b[self._param_key(index)] * modulation.unsqueeze(0)  # (out, rank)
            adapter[key] = (lora_a, lora_b)
        return adapter


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
    """Real (Sprint-2) generator: (task description embedding) -> per-target LoRA.

    Replaces the S1 stub's per-target free parameters with a **shared trunk** over
    the task embedding plus learned **layer** and **module** embeddings, fed to a
    per-target output head (default VeRA — the smallest parameterization, ~13 M on
    Mistral-7B). Cross-layer/module structure is shared through the trunk and the
    embeddings, not 96 independent generators. Same forward contract as the stub:
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
        parameterization: str = "vera",
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
        adapter: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for index, key in enumerate(self.keys):
            layer, module = self._meta[self._param_key(index)]
            conditioning = torch.cat([
                task_hidden,
                self.layer_emb(torch.tensor(layer)),
                self.module_emb(torch.tensor(self.module_ids[module])),
            ], dim=-1)
            adapter[key] = self.heads[self._param_key(index)](conditioning)
        return adapter
