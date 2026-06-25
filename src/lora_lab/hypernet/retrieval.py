"""Nearest-neighbor retrieval baseline (Sprint 5) — the bar the gate must clear.

The Phase-2 gate (notes.md §C2, §A.10) is: generated LoRAs must beat *retrieving
the closest existing library LoRA by task-description similarity*. This module
builds a cosine index over the **train-split** task descriptions and, for a
held-out description, returns the nearest train task — whose library adapter is
then applied as the "retrieved" baseline.

Train-split only by construction: a held-out task can never retrieve itself (the
index simply doesn't contain held-out tasks), which keeps the baseline honest.
Encoder-agnostic: anything implementing ``encode(list[str]) -> (N, d)`` works, so
tests inject a deterministic fake (no network).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .encoder import normalize_embeddings


@dataclass
class Retrieved:
    task: str
    score: float
    payload: dict[str, Any]


class RetrievalIndex:
    """Cosine-similarity index over train-split task-description embeddings."""

    def __init__(self, tasks: list[str], embeddings: torch.Tensor,
                 payloads: dict[str, dict]):
        assert embeddings.dim() == 2 and embeddings.shape[0] == len(tasks)
        self.tasks = tasks
        self.emb = normalize_embeddings(embeddings)  # (N, d), unit rows
        self.payloads = payloads

    @classmethod
    def build(cls, items: dict[str, dict], encoder) -> "RetrievalIndex":
        """items: {task_name: {"description": str, ...payload}}."""
        tasks = list(items)
        descriptions = [items[t]["description"] for t in tasks]
        emb = encoder.encode(descriptions)
        return cls(tasks, emb, {t: items[t] for t in tasks})

    def query_embedding(self, emb: torch.Tensor, k: int = 1) -> list[Retrieved]:
        q = normalize_embeddings(emb.view(1, -1))          # (1, d)
        sims = (self.emb @ q.T).squeeze(1)                 # (N,)
        k = min(k, len(self.tasks))
        top = torch.topk(sims, k)
        return [Retrieved(self.tasks[i], float(sims[i]), self.payloads[self.tasks[i]])
                for i in top.indices.tolist()]

    def query(self, description: str, encoder, k: int = 1) -> list[Retrieved]:
        return self.query_embedding(encoder.encode([description]), k=k)

    def __len__(self) -> int:
        return len(self.tasks)
