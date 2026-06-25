"""The hypernetwork (Sprint-1 stub) — (task embedding) -> per-target LoRA A/B.

This is the *plumbing* stub: it produces correctly-shaped, task-conditioned A/B
for every target Linear and is fully differentiable, so Sprint 1 can validate the
generate -> apply -> backprop loop. It is **not** the final architecture — Sprint 2
replaces the conditioning/heads (shared trunk + layer/module embeddings, and the
output-parameterization decision: VeRA-style vs low-rank vs full A/B).

Design (cheap, so it runs on a tiny base on CPU): per target key we hold base
factors ``A0:(r,in)`` (small random) and ``B0:(out,r)`` (**zero-init**, so ΔW = 0
at start — the no-op-adapter invariant). A small gate maps the task embedding to a
per-(key, rank) modulation, so different descriptions yield different adapters and
B0 can train away from zero. Real conditioning power comes in Sprint 2.
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
        d_task: int,
        *,
        r: int = 16,
        alpha: int = 32,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.keys = list(target_specs)
        self.r = r
        self.scaling = alpha / r
        self.specs = dict(target_specs)

        # Per-target base factors. A0 small-random, B0 zero (=> ΔW=0 at init).
        self.A0 = nn.ParameterDict()
        self.B0 = nn.ParameterDict()
        for i, key in enumerate(self.keys):
            in_f, out_f = target_specs[key]
            self.A0[self._pk(i)] = nn.Parameter(torch.randn(r, in_f) * init_scale)
            self.B0[self._pk(i)] = nn.Parameter(torch.zeros(out_f, r))

        # Task -> per-(key, rank) modulation gate. Cheap: one Linear.
        self.gate = nn.Linear(d_task, len(self.keys) * r)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    @staticmethod
    def _pk(i: int) -> str:
        # ParameterDict keys can't contain '.', so index by position.
        return f"t{i}"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, task_emb: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """task_emb: (d_task,) -> {key: (A:(r,in), B:(out,r))}."""
        g = self.gate(task_emb).view(len(self.keys), self.r)  # (num_keys, r)
        adapter: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, key in enumerate(self.keys):
            mod = 1.0 + g[i]                       # (r,) per-rank modulation
            a = self.A0[self._pk(i)] * mod.unsqueeze(1)   # (r, in)
            b = self.B0[self._pk(i)] * mod.unsqueeze(0)   # (out, r)
            adapter[key] = (a, b)
        return adapter


def delta_w(a: torch.Tensor, b: torch.Tensor, scaling: float) -> torch.Tensor:
    """ΔW = scaling · (B · A), shape (out, in) — the effective weight delta."""
    return scaling * (b @ a)


_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _parse_key(key: str) -> tuple[int, str]:
    """('...layers.5.self_attn.q_proj') -> (layer_idx=5, module='q_proj')."""
    m = _LAYER_RE.search(key)
    layer = int(m.group(1)) if m else 0
    module = key.rsplit(".", 1)[-1]
    return layer, module


class HyperLoRAGenerator(nn.Module):
    """Real (Sprint-2) generator: (task description embedding) -> per-target LoRA.

    Replaces the S1 stub's per-target free parameters with a **shared trunk** over
    the task embedding plus learned **layer** and **module** embeddings, fed to a
    per-target output head (default VeRA — the smallest parameterization, ~13 M on
    Mistral-7B). Cross-layer/module structure is shared through the trunk and the
    embeddings, not 96 independent generators. Same forward contract as the stub:
    ``task_emb -> {key: (A:(r,in), B:(out,r))}``, with the B path zero-init so
    ΔW = 0 at start (the no-op invariant). Conditioning is live at init (the A
    path varies with the task embedding); the B path opens up during training.
    """

    def __init__(
        self,
        target_specs: dict[str, tuple[int, int]],
        d_task: int,
        *,
        r: int = 16,
        alpha: int = 32,
        parameterization: str = "vera",
        d_layer: int = 16,
        d_module: int = 8,
        trunk_hidden: int = 128,
        head_kwargs: dict | None = None,
    ):
        super().__init__()
        self.keys = list(target_specs)
        self.r = r
        self.scaling = alpha / r
        self.parameterization = parameterization

        parsed = {k: _parse_key(k) for k in self.keys}
        n_layers = max((lyr for lyr, _ in parsed.values()), default=0) + 1
        modules = sorted({mod for _, mod in parsed.values()})
        self.module_ids = {m: i for i, m in enumerate(modules)}

        self.trunk = nn.Sequential(
            nn.Linear(d_task, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
        )
        self.layer_emb = nn.Embedding(n_layers, d_layer)
        self.module_emb = nn.Embedding(len(modules), d_module)
        d_cond = trunk_hidden + d_layer + d_module

        head_cls = HEADS[parameterization]
        self.heads = nn.ModuleDict()
        self._meta = {}
        for i, key in enumerate(self.keys):
            in_f, out_f = target_specs[key]
            self.heads[self._pk(i)] = head_cls(d_cond, in_f, out_f, r, **(head_kwargs or {}))
            self._meta[self._pk(i)] = parsed[key]

    @staticmethod
    def _pk(i: int) -> str:
        return f"t{i}"

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, task_emb: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        h_task = self.trunk(task_emb)
        adapter: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i, key in enumerate(self.keys):
            layer, module = self._meta[self._pk(i)]
            cond = torch.cat([
                h_task,
                self.layer_emb(torch.tensor(layer)),
                self.module_emb(torch.tensor(self.module_ids[module])),
            ], dim=-1)
            adapter[key] = self.heads[self._pk(i)](cond)
        return adapter
