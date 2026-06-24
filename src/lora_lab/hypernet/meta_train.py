"""Meta-training loop (Sprint 3/4 skeleton) — sample -> generate -> objective -> step.

The single loop that drives both objectives:
  reconstruction (S3 warmup): L1 between generated ΔW and a target library LoRA;
                              no base forward.
  sft (S4 gate run):          apply the generated adapter to the frozen base, run
                              the task batch, backprop the task loss into the
                              hypernetwork.

This module is **device/model-agnostic** so it runs on the tiny-plumbing config
(SmolLM2-135M, CPU) for validation. The GPU/4-bit Mistral path is the SAME code;
the entrypoint (scripts/phase2_meta_train.py, to be added) gates it behind an
explicit ``--allow-gpu`` so it is never launched autonomously.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .apply import LoRARegistry, inject, remove
from .recon import reconstruction_loss


class Sampler(Protocol):
    """Yields one training item: (description_text, target).

    target is a ``{key: (A, B)}`` adapter for reconstruction, or a tokenized
    ``{input_ids, attention_mask, labels}`` batch for SFT.
    """

    def sample(self) -> tuple[str, object]: ...


def meta_train(
    generator,
    base,
    target_modules,
    sampler: Sampler,
    encoder,
    *,
    steps: int,
    lr: float,
    objective: str = "reconstruction",
    device: str = "cpu",
    log_every: int = 10,
    log=print,
) -> list[float]:
    """Run ``steps`` of meta-training; return the per-step loss trace.

    ``generator`` trains; ``base`` stays frozen. For SFT we inject the generator's
    output as a live adapter so the task loss flows back into the generator only.
    """
    if objective not in ("reconstruction", "sft"):
        raise ValueError(f"objective must be reconstruction|sft, got {objective!r}")
    generator.to(device).train()
    base.to(device).eval()
    opt = torch.optim.Adam(generator.parameters(), lr=lr)
    scaling = generator.scaling

    reg = LoRARegistry()
    handles = inject(base, target_modules, reg, scaling=scaling) if objective == "sft" else []
    losses: list[float] = []
    try:
        for step in range(steps):
            desc, target = sampler.sample()
            task_emb = encoder.encode([desc]).to(device).squeeze(0)
            adapter = generator(task_emb)

            if objective == "reconstruction":
                loss = reconstruction_loss(adapter, target, scaling=scaling)
            else:  # sft — task loss through the frozen base
                reg.set_adapter(adapter)
                batch = {k: v.to(device) for k, v in target.items()}
                loss = base(**batch).loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
            if (step + 1) % log_every == 0:
                log(f"[meta-train] step {step+1}/{steps} {objective} loss={losses[-1]:.5f}")
    finally:
        remove(base, handles)
    return losses


class SyntheticReconSampler:
    """CPU plumbing sampler: random target adapters + dummy descriptions.

    Lets the loop run end-to-end with no library downloads (used by the tiny-
    plumbing smoke + tests). The real S3 sampler reads the Phase-1 train split +
    library adapters; the real S4 sampler tokenizes each task's SNI batch.
    """

    def __init__(self, target_specs: dict[str, tuple[int, int]], r: int, seed: int = 0):
        self.specs = target_specs
        self.r = r
        self.g = torch.Generator().manual_seed(seed)
        self._descs = ["classify sentiment", "translate text", "answer the question",
                       "summarize the passage", "detect entailment"]
        self._i = 0

    def sample(self) -> tuple[str, dict]:
        target = {k: (torch.randn(self.r, in_f, generator=self.g),
                      torch.randn(out_f, self.r, generator=self.g))
                  for k, (in_f, out_f) in self.specs.items()}
        desc = self._descs[self._i % len(self._descs)]
        self._i += 1
        return desc, target
