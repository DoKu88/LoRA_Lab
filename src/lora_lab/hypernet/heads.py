"""Output-parameterization heads (Sprint 2) — conditioning vector -> LoRA A/B.

The output dimensionality is the single biggest lever on hypernetwork size and
trainability (notes.md §A.2). Each head maps a per-target conditioning vector
``conditioning`` (task embedding ⊕ layer/module embedding) to factors
``A:(rank,in)``, ``B:(out,rank)`` for one target Linear. Three options,
smallest-output first. The S2 decision is **LowRankABHead** (the committed
default): it is the smallest rung that can actually *reconstruct* a target ΔW —
the diagnostic showed VeRA cannot (its frozen random basis only gets reweighted,
not reshaped), and FullABHead OOMs on 32 GB. VeRA/Full are kept as the documented
rungs of the ladder:

  LowRankABHead  generate A and B through a low-rank bottleneck (bottleneck_dim ≪
                 in,out), so the head weight is O(bottleneck_dim·(in+out)). **Default.**
  VeRAHead       frozen random A/B shared per target; generate only small scaling
                 vectors (VeRA-style, §2.4). Smallest output, but can only reweight
                 fixed directions → cannot reconstruct a specific adapter.
  FullABHead     dense heads emit A and B directly (the T2L default). Largest; OOMs.

All heads zero-init the B path so ΔW = 0 at start (the no-op invariant the S1
plumbing relies on). ``estimate_params`` compares total hypernetwork size across
parameterizations for a real target set — the number that drives the S2 choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FullABHead(nn.Module):
    """Dense: conditioning -> A (rank·in) and B (out·rank). Largest output; T2L default."""

    def __init__(self, cond_dim: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.a_head = nn.Linear(cond_dim, rank * in_features)
        self.b_head = nn.Linear(cond_dim, out_features * rank)
        nn.init.zeros_(self.b_head.weight)
        nn.init.zeros_(self.b_head.bias)

    def forward(self, conditioning: torch.Tensor):
        lora_a = self.a_head(conditioning).view(self.rank, self.in_features)
        lora_b = self.b_head(conditioning).view(self.out_features, self.rank)
        return lora_a, lora_b


class LowRankABHead(nn.Module):
    """Low-rank: conditioning -> a bottleneck code -> A, B via fixed factor banks.

    The generated A = a_from_code(code), B = b_from_code(code) keep the *learned*
    head small: O(bottleneck_dim·(in+out)) instead of O(cond_dim·rank·(in+out)).
    """

    def __init__(self, cond_dim: int, in_features: int, out_features: int, rank: int,
                 bottleneck_dim: int = 16):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.bottleneck_dim = bottleneck_dim
        self.code = nn.Linear(cond_dim, bottleneck_dim)
        self.a_from_code = nn.Linear(bottleneck_dim, rank * in_features, bias=False)
        self.b_from_code = nn.Linear(bottleneck_dim, out_features * rank, bias=False)
        nn.init.zeros_(self.b_from_code.weight)

    def forward(self, conditioning: torch.Tensor):
        code = torch.tanh(self.code(conditioning))
        lora_a = self.a_from_code(code).view(self.rank, self.in_features)
        lora_b = self.b_from_code(code).view(self.out_features, self.rank)
        return lora_a, lora_b


class VeRAHead(nn.Module):
    """VeRA-style (§2.4): BOTH A and B are frozen random buffers; the head
    generates only the two scaling vectors.

    ΔW = scaling · (Λ_b B)(Λ_d A) with Λ_d = diag(d):(rank), Λ_b = diag(b):(out).
    We fold the scalings into the returned factors — A_eff = A ⊙ d (per-row),
    B_eff = B ⊙ b (per-row) — so the (A, B) interface is unchanged. The only
    learned output is (rank + out) per target, so the head is O(cond_dim·(rank+out))
    — far smaller than generating a dense B. b is zero-init => ΔW = 0 at start.
    """

    def __init__(self, cond_dim: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        self.register_buffer("frozen_a", torch.randn(rank, in_features) / (in_features ** 0.5))
        self.register_buffer("frozen_b", torch.randn(out_features, rank) / (rank ** 0.5))
        self.d_scale = nn.Linear(cond_dim, rank)             # per-rank scale on A
        self.b_scale = nn.Linear(cond_dim, out_features)     # per-output scale on B
        nn.init.ones_(self.d_scale.bias)
        nn.init.zeros_(self.b_scale.weight)
        nn.init.zeros_(self.b_scale.bias)                  # b=0 => ΔW=0 at init

    def forward(self, conditioning: torch.Tensor):
        lora_a = self.frozen_a * self.d_scale(conditioning).view(self.rank, 1)
        lora_b = self.frozen_b * self.b_scale(conditioning).view(self.out_features, 1)
        return lora_a, lora_b


HEADS = {"full": FullABHead, "lowrank": LowRankABHead, "vera": VeRAHead}


def estimate_params(parameterization: str, target_specs: dict[str, tuple[int, int]],
                    cond_dim: int, rank: int, **head_kwargs) -> int:
    """Total *learned* hypernetwork params if each target gets its own head.

    Drives the S2 choice: e.g. for Mistral-7B (q/k/v over 32 layers) the Full
    parameterization is far larger than VeRA — this quantifies the gap before we
    commit a training run.
    """
    head_cls = HEADS[parameterization]
    total = 0
    for in_features, out_features in target_specs.values():
        head = head_cls(cond_dim, in_features, out_features, rank, **head_kwargs)
        total += sum(param.numel() for param in head.parameters())
    return total
