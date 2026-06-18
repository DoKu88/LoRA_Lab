#!/usr/bin/env python
"""Feasibility smoke test: DeepSpeed ZeRO offload full-FT of Mistral-7B.

Answers the core Phase 0.5 question fast — does the offload path actually fit
in 32 GB VRAM / 96 GB RAM and take real optimizer steps? Uses random token
batches (no SNI data needed) and a handful of steps.

    CUDA_HOME=/home/shadow1/miniconda3/envs/lora_lab \
    conda run -n lora_lab python scripts/smoke_offload.py --stage 2 --steps 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_lab.utils.vram import (  # noqa: E402
    HostRamTracer,
    bytes_to_gb,
)

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def ds_config(stage: int, micro_bs: int) -> dict:
    zero = {
        "stage": stage,
        "offload_optimizer": {"device": "cpu", "pin_memory": True},
        "contiguous_gradients": True,
        "overlap_comm": True,
    }
    if stage == 3:
        # also offload the params themselves to CPU (ZeRO-3) for max headroom
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    return {
        "train_micro_batch_size_per_gpu": micro_bs,
        "gradient_accumulation_steps": 1,
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2, choices=[2, 3])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-bs", type=int, default=1)
    ap.add_argument("--grad-checkpointing", action="store_true", default=True)
    args = ap.parse_args()

    # Single-process distributed env so DeepSpeed uses the torch backend and
    # does NOT try MPI discovery (mpi4py isn't installed and we don't need it
    # for one GPU). Must be set before deepspeed touches the dist layer.
    import os
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from transformers import AutoModelForCausalLM, AutoConfig

    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)

    print(f"== smoke offload: ZeRO-{args.stage}, {args.steps} steps, "
          f"seq={args.seq_len}, micro_bs={args.micro_bs}")
    ram = HostRamTracer(interval_s=0.25).start()

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    print(f"   model loaded in {time.time() - t_load:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    optimizer = DeepSpeedCPUAdam(model.parameters(), lr=1e-5)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config=ds_config(args.stage, args.micro_bs)
    )
    device = engine.device

    cfg = AutoConfig.from_pretrained(MODEL)
    vocab = cfg.vocab_size

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        ids = torch.randint(0, vocab, (args.micro_bs, args.seq_len), device=device)
        mask = torch.ones_like(ids)
        out = engine(input_ids=ids, attention_mask=mask, labels=ids)
        engine.backward(out.loss)
        engine.step()
        ram.record(step)
        losses.append(float(out.loss))
        peak_vram = bytes_to_gb(torch.cuda.max_memory_allocated())
        print(f"   step {step}: loss={losses[-1]:.3f} "
              f"peak_vram={peak_vram:.2f}GB ram={ram.ram_gb[-1]:.1f}GB")
    dt = time.time() - t0
    ram.stop()

    peak_vram = bytes_to_gb(torch.cuda.max_memory_allocated())
    print("\n== RESULT")
    print(f"   peak VRAM       : {peak_vram:.2f} GB  (limit 32)")
    print(f"   peak RAM        : {ram.peak_ram_gb:.1f} GB  (limit 96)")
    print(f"   peak RAM delta  : {ram.peak_ram_delta_gb:.1f} GB")
    print(f"   wall-clock/step : {dt/args.steps:.2f} s")
    print(f"   FITS            : {peak_vram <= 32.0 and ram.peak_ram_gb <= 96.0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
