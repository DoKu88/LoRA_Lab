"""FSDP CPU-offload strategy (Sprint 3/8) — the second offload data point.

PyTorch-native FullyShardedDataParallel with `CPUOffload(offload_params=True)`:
on a single GPU this behaves like ZeRO-3-offload — params/grads/optimizer live on
CPU and stream to the GPU per transformer block. Cross-checks the DeepSpeed
finding. With a standard fp32 AdamW the optimizer state is the same ~84 GB that
OOMed DeepSpeed, so this is expected to also be RAM-bound unless paired with an
8-bit optimizer (`levers.use_8bit_adam`).
"""

from __future__ import annotations

import functools
import math
import os
import time
from pathlib import Path

import torch

from ...config import RunConfig
from ...data.sni import DataCollatorForSupervised, get_dataset
from ...methods.build import _load_tokenizer
from ...train.params import count_parameters
from ...train.run_logger import RunLogger
from ...utils.vram import HostRamTracer, MemoryTracer, bytes_to_gb, reset_peak_memory
from ._common import PHASE05_TRACE_DIR, _num_opt_steps


def _setup_single_process_dist() -> None:
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")


def run_fsdp(config: RunConfig) -> dict:
    import torch.distributed as dist
    from torch.distributed.fsdp import CPUOffload, MixedPrecision
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

    _setup_single_process_dist()
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)

    device = "cuda"
    logger = RunLogger(config)
    tracer = MemoryTracer()
    ram_tracer = HostRamTracer().start()
    reset_peak_memory()

    tok = _load_tokenizer(config)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16)
    model.config.use_cache = False
    if config.levers.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls={MistralDecoderLayer}
    )
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16,
                        buffer_dtype=torch.bfloat16)
    model = FSDP(
        model, auto_wrap_policy=wrap_policy,
        cpu_offload=CPUOffload(offload_params=True), mixed_precision=mp,
        device_id=torch.cuda.current_device(), use_orig_params=True,
    )
    if config.levers.gradient_checkpointing:
        model.enable_input_require_grads()

    if config.levers.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=config.hparams.lr,
                                        weight_decay=config.hparams.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.hparams.lr,
                                      weight_decay=config.hparams.weight_decay)

    bundle = get_dataset(
        config.task, tok, max_seq_len=config.hparams.max_seq_len,
        max_train_samples=config.hparams.max_train_samples,
        seed=config.hparams.seed, max_eval_samples=config.eval.max_eval_samples,
    )
    collate = DataCollatorForSupervised(tok)
    loader = torch.utils.data.DataLoader(
        bundle.train, batch_size=config.hparams.batch_size, shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.hparams.seed), drop_last=True,
    )
    total_steps = _num_opt_steps(len(bundle.train), config)
    grad_accum = config.hparams.grad_accum
    pinfo = count_parameters(model)
    print(f"[fsdp] {config.name}: {total_steps} steps, 8bit={config.levers.use_8bit_adam}")

    model.train()
    step = 0
    final_loss = float("nan")
    interval_loss = 0.0
    interval_micro = 0
    t0 = time.time()
    interval_t = time.time()
    log_every = config.logging.log_every
    done = False
    while not done:
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            interval_loss += out.loss.item()
            interval_micro += 1
            if (micro + 1) % grad_accum == 0:
                model.clip_grad_norm_(1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                tracer.record(step)
                ram_tracer.record(step)
                if step % log_every == 0 or step == total_steps:
                    dt = time.time() - interval_t
                    final_loss = interval_loss / max(1, interval_micro)
                    logger.log_metrics(step, {
                        "train_loss": round(final_loss, 5),
                        "gpu_mem_gb": round(bytes_to_gb(torch.cuda.memory_allocated()), 4),
                        "ram_gb": round(ram_tracer.ram_gb[-1], 4) if len(ram_tracer) else 0.0,
                        "step_time_s": round(dt / max(1, log_every), 4),
                    })
                    interval_loss = 0.0
                    interval_micro = 0
                    interval_t = time.time()
                if step >= total_steps:
                    done = True
                    break

    wallclock = time.time() - t0
    ram_tracer.stop()
    trace_path = PHASE05_TRACE_DIR / f"{config.name}.csv"
    ram_trace_path = PHASE05_TRACE_DIR / f"{config.name}.ram.csv"
    tracer.save_csv(trace_path)
    ram_tracer.save_csv(ram_trace_path)
    logger.log_artifact_path(trace_path, "mem_trace")
    logger.log_artifact_path(ram_trace_path, "ram_trace")

    steps_per_epoch = max(1, math.ceil(len(bundle.train) / (config.hparams.batch_size * grad_accum)))
    logger.set_summary(
        **pinfo, method=config.method, base_model=config.base_model, task=config.task,
        peak_vram_gb=round(tracer.peak_gb, 4),
        peak_reserved_gb=round(tracer.peak_reserved_gb, 4),
        peak_ram_gb=round(ram_tracer.peak_ram_gb, 4),
        peak_ram_delta_gb=round(ram_tracer.peak_ram_delta_gb, 4),
        baseline_ram_gb=round(ram_tracer.baseline_ram_gb, 4),
        final_train_loss=round(final_loss, 5),
        wallclock_s=round(wallclock, 2),
        wallclock_per_step_s=round(wallclock / max(1, step), 4),
        wallclock_per_epoch_s=round(wallclock * steps_per_epoch / max(1, step), 2),
        steps=step, mem_trace=str(trace_path),
    )
    summary = dict(logger.summary)
    logger.finish()
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()
    return summary
