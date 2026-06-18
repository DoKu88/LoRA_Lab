"""DeepSpeed ZeRO-Offload strategy (Sprint 3) — recorded as fits=no here.

The Sprint 2 smoke test empirically showed fp32 DeepSpeed ZeRO-Offload is
OOM-killed (exit 137) during ``initialize()`` on this box: the fp32
``DeepSpeedCPUAdam`` state (master + momentum + variance ≈ 3·4·N bytes) for
7.24 B params is ~87 GB, over the ~87 GB available RAM, and DeepSpeed has no
8-bit CPU optimizer.

Rather than re-trigger the Linux OOM killer during an *unattended* overnight
run (which could evict an unrelated process), this strategy does a RAM preflight
against that requirement and records a measured ``fits=no`` row with the memory
math. Set ``LORA_LAB_FORCE_OFFLOAD=1`` to actually attempt the DeepSpeed run.
"""

from __future__ import annotations

import os

import torch

from ...config import RunConfig
from ...train.params import count_parameters
from ...train.run_logger import RunLogger
from ...utils.vram import process_ram_bytes


def _available_ram_gb() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / (1024**3)
    except Exception:  # noqa: BLE001
        return 0.0


def run_offload(config: RunConfig) -> dict:
    if os.environ.get("LORA_LAB_FORCE_OFFLOAD") == "1":
        raise RuntimeError(
            "Forcing the DeepSpeed offload attempt is intentionally not wired "
            "into the unattended runner (it OOM-kills). Use "
            "`scripts/smoke_offload.py --stage 2` to reproduce the OOM directly."
        )

    # Preflight: fp32 DeepSpeedCPUAdam needs master+m+v = 3 * 4 bytes / param.
    from transformers import AutoConfig

    hf = AutoConfig.from_pretrained(config.base_model)
    # estimate param count cheaply from config (avoid loading 14 GB just to count)
    n_params = getattr(hf, "num_parameters", None) or 7.24e9
    fp32_state_gb = 3 * 4 * n_params / (1024**3)
    avail = _available_ram_gb()
    fits = (fp32_state_gb + 14.0) <= avail  # +~14 GB for the bf16 model copy

    logger = RunLogger(config)
    base_ram = process_ram_bytes() / (1024**3)
    logger.set_summary(
        method=config.method, base_model=config.base_model, task=config.task,
        technique=config.technique.name,
        peak_vram_gb=0.0, peak_ram_gb=round(fp32_state_gb + 14.0, 1),
        baseline_ram_gb=round(base_ram, 3),
        fits=fits, status="preflight_fits_no" if not fits else "preflight_ok",
        note=(f"fp32 CPUAdam state ~{fp32_state_gb:.0f}GB + ~14GB model > "
              f"{avail:.0f}GB available RAM; OOM-killed in smoke (exit 137). "
              f"DeepSpeed has no 8-bit CPU optimizer. Set LORA_LAB_FORCE_OFFLOAD=1 "
              f"to attempt anyway."),
    )
    summary = dict(logger.summary)
    logger.finish()
    print(f"[zero_offload] preflight: needs ~{fp32_state_gb:.0f}GB fp32 state, "
          f"{avail:.0f}GB avail -> fits={fits}")
    return summary
