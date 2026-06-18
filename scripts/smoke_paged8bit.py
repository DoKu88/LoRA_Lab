#!/usr/bin/env python
"""Feasibility smoke: full-FT Mistral-7B with bitsandbytes paged 8-bit AdamW.

DeepSpeed ZeRO-Offload needs fp32 CPUAdam (~84 GB CPU state) and OOMs the 96 GB
RAM here. The 8-bit alternative keeps optimizer state in 8 bits and *pages* it to
CPU on demand, with the full model resident on the 32 GB GPU. This smoke checks
whether that fits in VRAM and takes real steps.

    conda run -n lora_lab python scripts/smoke_paged8bit.py --steps 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_lab.utils.vram import HostRamTracer, bytes_to_gb  # noqa: E402

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-bs", type=int, default=1)
    args = ap.parse_args()

    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoConfig

    print(f"== smoke paged-8bit: {args.steps} steps seq={args.seq_len} bs={args.micro_bs}")
    ram = HostRamTracer(interval_s=0.25).start()

    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model = model.to("cuda")
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"   model on GPU in {time.time()-t_load:.1f}s; "
          f"resident VRAM={bytes_to_gb(torch.cuda.memory_allocated()):.2f}GB")

    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=1e-5)
    cfg = AutoConfig.from_pretrained(MODEL)
    vocab = cfg.vocab_size

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        ids = torch.randint(0, vocab, (args.micro_bs, args.seq_len), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ram.record(step)
        losses.append(float(out.loss))
        print(f"   step {step}: loss={losses[-1]:.3f} "
              f"peak_vram={bytes_to_gb(torch.cuda.max_memory_allocated()):.2f}GB "
              f"ram={ram.ram_gb[-1]:.1f}GB")
    dt = time.time() - t0
    ram.stop()

    peak_vram = bytes_to_gb(torch.cuda.max_memory_allocated())
    print("\n== RESULT")
    print(f"   peak VRAM       : {peak_vram:.2f} GB  (limit 32)")
    print(f"   peak RAM        : {ram.peak_ram_gb:.1f} GB  (limit 96)")
    print(f"   wall-clock/step : {dt/args.steps:.2f} s")
    print(f"   FITS            : {peak_vram <= 32.0 and ram.peak_ram_gb <= 96.0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
