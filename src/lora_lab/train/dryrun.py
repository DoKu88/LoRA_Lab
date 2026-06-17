"""Synthetic end-to-end run for --dry-run (no GPU, no model).

Exercises the whole harness path — config snapshot, per-step metrics
(including a gpu_mem_gb time series), memory-trace CSV, and summary.json —
so the logging/plumbing can be validated on a CPU-only box (sprint plan S3
required testing: "dry-run completes without a GPU").
"""

from __future__ import annotations

import math
from pathlib import Path

from ..config import RunConfig
from ..utils.vram import MemoryTracer
from .run_logger import RunLogger


def run_dry(config: RunConfig, n_steps: int = 20) -> dict:
    """Produce a fake but well-formed run and return its summary."""
    logger = RunLogger(config)
    tracer = MemoryTracer()

    # Method-dependent fake peak so dry-run artifacts still show the expected
    # ordering QLoRA < LoRA < full FT (sanity-checkable downstream).
    base_mem = {"qlora": 1.5, "lora": 2.5, "full_ft": 5.0}.get(config.method, 2.0)

    for step in range(1, n_steps + 1):
        # decaying synthetic loss
        loss = 2.5 * math.exp(-step / 8.0) + 0.2
        # fake memory curve: ramps up then plateaus
        fake_alloc = base_mem * (1.0 - math.exp(-step / 3.0))
        tracer.steps.append(step)
        tracer.allocated_gb.append(round(fake_alloc, 4))
        tracer.reserved_gb.append(round(fake_alloc * 1.1, 4))
        logger.log_metrics(
            step,
            {
                "train_loss": round(loss, 4),
                "gpu_mem_gb": round(fake_alloc, 4),
                "gpu_mem_reserved_gb": round(fake_alloc * 1.1, 4),
                "tokens_per_sec": 1234.0,
                "step_time_s": 0.05,
                "lr": config.hparams.lr,
            },
        )

    # persist the memory trace where Sprint 5 expects it
    trace_path = Path("results/mem_trace") / f"{config.name}.csv"
    tracer.save_csv(trace_path)
    logger.log_artifact_path(trace_path, "mem_trace")

    logger.set_summary(
        dry_run=True,
        method=config.method,
        base_model=config.base_model,
        task=config.task,
        trainable_params=460800,
        total_params=134000000,
        pct_params=0.34,
        peak_vram_gb=tracer.peak_gb,
        final_train_loss=round(2.5 * math.exp(-n_steps / 8.0) + 0.2, 4),
        wallclock_per_epoch_s=n_steps * 0.05,
    )
    logger.finish()
    return logger.summary
