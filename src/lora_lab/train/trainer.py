"""Shared trainer for full FT / LoRA / QLoRA (Sprint 4).

A manual loop (not HF Trainer) so we control exactly when GPU memory is
sampled: at every logged step we append (step, gpu_mem_gb) to a MemoryTracer,
giving the memory-vs-iteration trace the comparison plots. The per-run peak is
just ``max`` of that trace and becomes the ``peak_vram_gb`` column.

``train(config)`` returns a summary dict (the row Sprint 5 collects) and saves
a reloadable checkpoint: adapter-only for LoRA/QLoRA, full weights for FT.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from ..config import RunConfig
from ..data.sni import DataCollatorForSupervised, get_dataset
from ..methods.build import build_model_and_tokenizer, build_optimizer
from ..utils.vram import (
    HostRamTracer,
    MemoryTracer,
    cuda_mem_snapshot,
    reset_peak_memory,
)
from .params import count_parameters
from .run_logger import RunLogger


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _num_training_steps(n_examples: int, config: RunConfig) -> int:
    eff_batch = config.hparams.batch_size * config.hparams.grad_accum
    steps_per_epoch = max(1, math.ceil(n_examples / eff_batch))
    if config.hparams.max_steps and config.hparams.max_steps > 0:
        return config.hparams.max_steps
    return int(steps_per_epoch * config.hparams.num_epochs)


def _dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024**2), 3)


def train(config: RunConfig) -> dict:
    _set_seed(config.hparams.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = RunLogger(config)
    tracer = MemoryTracer()
    # Host-RAM probe: baseline captured now (before model load) so peak-delta is
    # the run's own footprint; the background thread catches CPU-side spikes
    # (offloaded optimizer step) that land between logged training steps.
    ram_tracer = HostRamTracer().start()
    reset_peak_memory()

    # ---- data ----------------------------------------------------------
    model, tok = build_model_and_tokenizer(config)
    bundle = get_dataset(
        config.task,
        tok,
        max_seq_len=config.hparams.max_seq_len,
        max_train_samples=config.hparams.max_train_samples,
        seed=config.hparams.seed,
        max_eval_samples=config.eval.max_eval_samples,
    )
    collate = DataCollatorForSupervised(tok)
    loader = torch.utils.data.DataLoader(
        bundle.train,
        batch_size=config.hparams.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.hparams.seed),
        drop_last=False,
    )

    # ---- optimizer + schedule -----------------------------------------
    from transformers import get_scheduler

    total_steps = _num_training_steps(len(bundle.train), config)
    optimizer = build_optimizer(config, model)
    scheduler = get_scheduler(
        config.hparams.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=int(config.hparams.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    pinfo = count_parameters(model)
    logger.set_summary(**pinfo, total_train_examples=len(bundle.train),
                       planned_steps=total_steps, **bundle.summary())
    print(f"[train] {config.name}: {total_steps} steps, "
          f"{pinfo['trainable_params']:,} trainable ({pinfo['pct_params']}%)")

    # ---- loop ----------------------------------------------------------
    model.train()
    grad_accum = config.hparams.grad_accum
    log_every = config.logging.log_every
    mem_every = max(1, config.logging.mem_trace_every)

    step = 0
    t0 = time.time()
    interval_loss = 0.0
    interval_tokens = 0
    interval_t = time.time()
    final_loss = float("nan")
    done = False

    while not done:
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # bf16 autocast: necessary for fp32-master full FT, harmless for the
            # already-bf16/4-bit LoRA/QLoRA bases. bf16 needs no GradScaler.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(**batch)
            loss = out.loss / grad_accum
            loss.backward()
            interval_loss += out.loss.item()
            interval_tokens += int(batch["attention_mask"].sum().item())

            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % mem_every == 0:
                    tracer.record(step)
                    ram_tracer.record(step)

                if step % log_every == 0 or step == total_steps:
                    dt = time.time() - interval_t
                    n_optsteps = log_every if step % log_every == 0 else (step % log_every)
                    avg_loss = interval_loss / max(1, n_optsteps * grad_accum)
                    snap = cuda_mem_snapshot()
                    final_loss = avg_loss
                    logger.log_metrics(
                        step,
                        {
                            "train_loss": round(avg_loss, 5),
                            "gpu_mem_gb": round(snap["allocated_gb"], 4),
                            "gpu_mem_reserved_gb": round(snap["reserved_gb"], 4),
                            "ram_gb": round(ram_tracer.ram_gb[-1], 4) if len(ram_tracer) else 0.0,
                            "tokens_per_sec": round(interval_tokens / dt, 1) if dt > 0 else 0.0,
                            "step_time_s": round(dt / max(1, n_optsteps), 4),
                            "lr": scheduler.get_last_lr()[0],
                        },
                    )
                    interval_loss = 0.0
                    interval_tokens = 0
                    interval_t = time.time()

                if step >= total_steps:
                    done = True
                    break
        else:
            continue
        # inner break -> ensure trace has at least one sample
        if len(tracer) == 0:
            tracer.record(step)

    wallclock = time.time() - t0
    ram_tracer.stop()

    # ---- persist memory traces + checkpoint ---------------------------
    if len(tracer) == 0:
        tracer.record(max(step, 1))
    if len(ram_tracer) == 0:
        ram_tracer.record(max(step, 1))
    trace_path = Path("results/mem_trace") / f"{config.name}.csv"
    tracer.save_csv(trace_path)
    logger.log_artifact_path(trace_path, "mem_trace")
    ram_trace_path = Path("results/mem_trace") / f"{config.name}.ram.csv"
    ram_tracer.save_csv(ram_trace_path)
    logger.log_artifact_path(ram_trace_path, "ram_trace")

    ckpt_dir = config.output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))  # adapter-only for PEFT, full for FT
    tok.save_pretrained(str(ckpt_dir))
    ckpt_mb = _dir_size_mb(ckpt_dir)

    eff_batch = config.hparams.batch_size * config.hparams.grad_accum
    steps_per_epoch = max(1, math.ceil(len(bundle.train) / eff_batch))
    logger.set_summary(
        method=config.method,
        base_model=config.base_model,
        task=config.task,
        peak_vram_gb=round(tracer.peak_gb, 4),
        peak_reserved_gb=round(tracer.peak_reserved_gb, 4),
        peak_ram_gb=round(ram_tracer.peak_ram_gb, 4),
        peak_ram_delta_gb=round(ram_tracer.peak_ram_delta_gb, 4),
        baseline_ram_gb=round(ram_tracer.baseline_ram_gb, 4),
        final_train_loss=round(final_loss, 5),
        wallclock_s=round(wallclock, 2),
        wallclock_per_epoch_s=round(wallclock * steps_per_epoch / max(1, step), 2),
        steps=step,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_size_mb=ckpt_mb,
        mem_trace=str(trace_path),
    )
    summary = dict(logger.summary)
    logger.finish()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
