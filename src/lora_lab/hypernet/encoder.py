"""Task-description encoder — text -> fixed task embedding.

The hypernetwork is *conditioned* on a natural-language task description. This
module turns that text into a fixed-width embedding the generator consumes. The
encoder is **frozen** (it is conditioning, not something we train) — only the
hypernetwork heads train.

Default path (``MeanPoolEncoder``): mean-pool a frozen HF model's hidden states
over the description tokens. Cheap, CPU-loadable, no extra dependency beyond
``transformers``. A sentence-transformers model can be swapped in behind the same
``encode()`` interface.
"""

from __future__ import annotations

from typing import Protocol

import torch
from transformers import AutoModel, AutoTokenizer


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
