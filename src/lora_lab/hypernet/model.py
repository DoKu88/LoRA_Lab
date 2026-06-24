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

import torch
import torch.nn as nn


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
