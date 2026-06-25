"""Output-parameterization heads (Sprint 2) — conditioning vector -> LoRA A/B.

The output dimensionality is the single biggest lever on hypernetwork size and
trainability (notes.md §A.2). Each head maps a per-target conditioning vector
``h`` (task embedding ⊕ layer/module embedding) to factors ``A:(r,in)``,
``B:(out,r)`` for one target Linear. Three options, smallest-output first — the
S2 decision picks the smallest that clears the gate (with a documented ladder):

  VeRAHead       frozen random A shared per target; generate only small per-rank
                 scaling vectors + B (VeRA-style, §2.4). Smallest output.
  LowRankABHead  generate A and B through a low-rank bottleneck (k ≪ in,out), so
                 the head weight is O(k·(in+out)) not O(r·(in+out)·d_cond).
  FullABHead     dense heads emit A and B directly (the T2L default). Largest.

All heads zero-init the B path so ΔW = 0 at start (the no-op invariant the S1
plumbing relies on). ``estimate_params`` compares total hypernetwork size across
parameterizations for a real target set — the number that drives the S2 choice.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FullABHead(nn.Module):
    """Dense: h -> A (r·in) and B (out·r). Largest output; T2L default."""

    def __init__(self, d_cond: int, in_f: int, out_f: int, r: int):
        super().__init__()
        self.in_f, self.out_f, self.r = in_f, out_f, r
        self.A = nn.Linear(d_cond, r * in_f)
        self.B = nn.Linear(d_cond, out_f * r)
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

    def forward(self, h: torch.Tensor):
        a = self.A(h).view(self.r, self.in_f)
        b = self.B(h).view(self.out_f, self.r)
        return a, b


class LowRankABHead(nn.Module):
    """Low-rank: h -> a k-dim code -> A, B via fixed factor banks.

    The generated A = Wa_out @ diag(code_a) @ Wa_in-style bottleneck keeps the
    *learned* head small: O(k·(in+out)) instead of O(d_cond·r·(in+out)).
    """

    def __init__(self, d_cond: int, in_f: int, out_f: int, r: int, k: int = 16):
        super().__init__()
        self.in_f, self.out_f, self.r, self.k = in_f, out_f, r, k
        self.code = nn.Linear(d_cond, k)
        self.A_from_code = nn.Linear(k, r * in_f, bias=False)
        self.B_from_code = nn.Linear(k, out_f * r, bias=False)
        nn.init.zeros_(self.B_from_code.weight)

    def forward(self, h: torch.Tensor):
        c = torch.tanh(self.code(h))
        a = self.A_from_code(c).view(self.r, self.in_f)
        b = self.B_from_code(c).view(self.out_f, self.r)
        return a, b


class VeRAHead(nn.Module):
    """VeRA-style (§2.4): BOTH A and B are frozen random buffers; the head
    generates only the two scaling vectors.

    ΔW = scaling · (Λ_b B)(Λ_d A) with Λ_d = diag(d):(r), Λ_b = diag(b):(out).
    We fold the scalings into the returned factors — A_eff = A ⊙ d (per-row),
    B_eff = B ⊙ b (per-row) — so the (A, B) interface is unchanged. The only
    learned output is (r + out) per target, so the head is O(d_cond·(r+out)) —
    far smaller than generating a dense B. b is zero-init => ΔW = 0 at start.
    """

    def __init__(self, d_cond: int, in_f: int, out_f: int, r: int):
        super().__init__()
        self.in_f, self.out_f, self.r = in_f, out_f, r
        self.register_buffer("A_frozen", torch.randn(r, in_f) / (in_f ** 0.5))
        self.register_buffer("B_frozen", torch.randn(out_f, r) / (r ** 0.5))
        self.d_scale = nn.Linear(d_cond, r)        # per-rank scale on A
        self.b_scale = nn.Linear(d_cond, out_f)    # per-output scale on B
        nn.init.ones_(self.d_scale.bias)
        nn.init.zeros_(self.b_scale.weight)
        nn.init.zeros_(self.b_scale.bias)          # b=0 => ΔW=0 at init

    def forward(self, h: torch.Tensor):
        a = self.A_frozen * self.d_scale(h).view(self.r, 1)
        b = self.B_frozen * self.b_scale(h).view(self.out_f, 1)
        return a, b


HEADS = {"full": FullABHead, "lowrank": LowRankABHead, "vera": VeRAHead}


def estimate_params(parameterization: str, target_specs: dict[str, tuple[int, int]],
                    d_cond: int, r: int, **kw) -> int:
    """Total *learned* hypernetwork params if each target gets its own head.

    Drives the S2 choice: e.g. for Mistral-7B (q/k/v over 32 layers) the Full
    parameterization is far larger than VeRA — this quantifies the gap before we
    commit a training run.
    """
    cls = HEADS[parameterization]
    total = 0
    for in_f, out_f in target_specs.values():
        head = cls(d_cond, in_f, out_f, r, **kw)
        total += sum(p.numel() for p in head.parameters())
    return total
