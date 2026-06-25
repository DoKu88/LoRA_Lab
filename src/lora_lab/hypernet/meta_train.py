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

import time
from typing import Protocol

import torch

from ..utils.vram import cuda_mem_snapshot, reset_peak_memory
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
    logger=None,
) -> list[float]:
    """Run ``steps`` of meta-training; return the per-step loss trace.

    ``generator`` trains; ``base`` stays frozen. For SFT we inject the generator's
    output as a live adapter so the task loss flows back into the generator only.
    If ``logger`` (a ``RunLogger``) is given, every step logs ``<objective>/loss``,
    lr, grad norm, step time, and the VRAM snapshot (the W&B tracking contract).
    """
    if objective not in ("reconstruction", "sft"):
        raise ValueError(f"objective must be reconstruction|sft, got {objective!r}")
    generator.to(device).train()
    base.to(device).eval()
    optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
    scaling = generator.scaling
    if device == "cuda":
        reset_peak_memory()

    registry = LoRARegistry()
    handles = inject(base, target_modules, registry, scaling=scaling) if objective == "sft" else []
    losses: list[float] = []
    try:
        for step in range(steps):
            t_step = time.time()
            description, target = sampler.sample()
            task_emb = encoder.encode([description]).to(device).squeeze(0)
            adapter = generator(task_emb)

            if objective == "reconstruction":
                loss = reconstruction_loss(adapter, target, scaling=scaling)
            else:  # sft — task loss through the frozen base
                registry.set_adapter(adapter)
                batch = {key: value.to(device) for key, value in target.items()}
                loss = base(**batch).loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (p for p in generator.parameters() if p.requires_grad), 1.0
            )
            optimizer.step()
            losses.append(float(loss.item()))

            if logger is not None:
                snap = cuda_mem_snapshot() if device == "cuda" else {"allocated_gb": 0.0, "reserved_gb": 0.0}
                logger.log_metrics(step, {
                    f"{objective}/loss": round(losses[-1], 6),
                    "lr": lr,
                    "grad_norm": round(float(grad_norm), 4),
                    "step_time_s": round(time.time() - t_step, 4),
                    "gpu_mem_gb": round(snap["allocated_gb"], 4),
                    "gpu_mem_reserved_gb": round(snap["reserved_gb"], 4),
                })
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

    def __init__(self, target_specs: dict[str, tuple[int, int]], rank: int, seed: int = 0):
        self.specs = target_specs
        self.rank = rank
        self.torch_generator = torch.Generator().manual_seed(seed)
        self._descriptions = ["classify sentiment", "translate text", "answer the question",
                              "summarize the passage", "detect entailment"]
        self._index = 0

    def sample(self) -> tuple[str, dict]:
        target = {key: (torch.randn(self.rank, in_features, generator=self.torch_generator),
                        torch.randn(out_features, self.rank, generator=self.torch_generator))
                  for key, (in_features, out_features) in self.specs.items()}
        description = self._descriptions[self._index % len(self._descriptions)]
        self._index += 1
        return description, target


def assert_run_allowed(cfg, allow_gpu: bool) -> None:
    """Refuse the GPU / 4-bit (Mistral SFT) path unless the user opts in.

    The autonomous overnight loop must never launch the expensive, design-
    sensitive Mistral run; the entrypoint requires an explicit ``--allow-gpu``.
    """
    if (getattr(cfg, "device", "cpu") == "cuda" or getattr(cfg, "load_in_4bit", False)) \
            and not allow_gpu:
        raise RuntimeError(
            "refusing GPU/4-bit run without --allow-gpu "
            "(the Mistral-7B SFT run is launched manually after design review)"
        )
