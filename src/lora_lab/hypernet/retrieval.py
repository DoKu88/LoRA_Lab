"""Nearest-neighbor retrieval baseline — the bar a generated LoRA must clear.

The baseline: instead of generating a LoRA, retrieve *the closest existing
library LoRA by task-description similarity*. This module builds a cosine index
over the **train-split** task descriptions and, for a held-out description,
returns the nearest train task — whose library adapter is then applied as the
"retrieved" baseline.

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
        self.embeddings = normalize_embeddings(embeddings)  # (N, d), unit rows
        self.payloads = payloads

    @classmethod
    def build(cls, items: dict[str, dict], encoder) -> "RetrievalIndex":
        """items: {task_name: {"description": str, ...payload}}."""
        tasks = list(items)
        descriptions = [items[task]["description"] for task in tasks]
        embeddings = encoder.encode(descriptions)
        return cls(tasks, embeddings, {task: items[task] for task in tasks})

    def query_embedding(self, embedding: torch.Tensor, top_k: int = 1) -> list[Retrieved]:
        query_vec = normalize_embeddings(embedding.view(1, -1))   # (1, d)
        similarities = (self.embeddings @ query_vec.T).squeeze(1)  # (N,)
        top_k = min(top_k, len(self.tasks))
        top = torch.topk(similarities, top_k)
        return [Retrieved(self.tasks[index], float(similarities[index]), self.payloads[self.tasks[index]])
                for index in top.indices.tolist()]

    def query(self, description: str, encoder, top_k: int = 1) -> list[Retrieved]:
        return self.query_embedding(encoder.encode([description]), top_k=top_k)

    def __len__(self) -> int:
        return len(self.tasks)
