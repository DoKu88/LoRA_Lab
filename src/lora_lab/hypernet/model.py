"""The Text-to-LoRA model: text description -> LoRA A/B for every target Linear.

Three pieces:
  MeanPoolEncoder     frozen text encoder: description -> fixed task embedding
  *Head               output heads that map a conditioning vector -> (A, B)
  HyperLoRAGenerator  shared trunk + layer/module embeddings + per-target head

The generator emits, in one forward pass, ``{key: (A:(rank,in), B:(out,rank))}``
for every targeted Linear. The B path is zero-init so ΔW = 0 at start.
"""

from __future__ import annotations

import re
from typing import Protocol

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Encoder: task description -> fixed embedding (frozen; conditioning only)
# ---------------------------------------------------------------------------
class TaskEncoder(Protocol):
    """encode a list of descriptions -> (N, dim) embeddings; expose ``dim``."""

    dim: int

    def encode(self, descriptions: list[str]) -> torch.Tensor: ...


class MeanPoolEncoder:
    """Mean-pool a frozen HF encoder's last hidden state over real tokens."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu", max_len: int = 128):
        self.device = device
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:  # decoder-only encoders (e.g. SmolLM2) lack one
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def encode(self, descriptions: list[str]) -> torch.Tensor:
        batch = self.tokenizer(descriptions, return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.device)
        out = self.model(**batch).last_hidden_state          # (N, T, H)
        mask = batch["attention_mask"].unsqueeze(-1).float()  # (N, T, 1)
        summed = (out * mask).sum(1)
        counts = mask.sum(1).clamp(min=1.0)
        return summed / counts                                # (N, H)


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows — used by the retrieval baseline + as a stable
    conditioning input."""
    return torch.nn.functional.normalize(embeddings, dim=-1)


# ---------------------------------------------------------------------------
# Output heads: conditioning vector -> LoRA (A, B). Smallest-output first.
# Committed default is LowRankABHead (the smallest that can reconstruct a target
# ΔW). All heads zero-init the B path so ΔW = 0 at start (the no-op invariant).
# ---------------------------------------------------------------------------
class FullABHead(nn.Module):
    """Dense: conditioning -> A (rank·in) and B (out·rank). Largest output."""

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
    """Low-rank: conditioning -> a bottleneck code -> A, B (the default).

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
    """VeRA-style: BOTH A and B are frozen random buffers; the head generates
    only the two scaling vectors.

    ΔW = scaling · (Λ_b B)(Λ_d A) with Λ_d = diag(d):(rank), Λ_b = diag(b):(out).
    We fold the scalings into the returned factors — A_eff = A ⊙ d (per-row),
    B_eff = B ⊙ b (per-row) — so the (A, B) interface is unchanged. The only
    learned output is (rank + out) per target. b is zero-init => ΔW = 0 at start.
    Smallest output, but can only reweight fixed directions, so it cannot
    reconstruct a specific adapter (hence not the default).
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


# ---------------------------------------------------------------------------
# Generator: (task embedding) -> per-target (A, B)
# ---------------------------------------------------------------------------
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
    **module** embeddings feeds a per-target output head (default low-rank A/B).
    Cross-layer/module structure is shared through the trunk and the embeddings,
    not one independent generator per target. Forward contract:
    ``task_emb -> {key: (A:(rank,in), B:(out,rank))}``, with the B path zero-init
    so ΔW = 0 at start. The A path varies with the task embedding from init.
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
