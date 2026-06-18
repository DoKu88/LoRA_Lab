"""Shared manual training loop for the on-GPU Phase 0.5 strategies.

Most full-FT techniques (paged-8bit baseline, GaLore/Q-GaLore, BAdam, LOMO)
are "load bf16 Mistral-7B on the GPU, then train with a memory-frugal optimizer"
— they differ only in the optimizer (and, for LOMO, in fusing backward with the
update). This module factors out everything else: model loading, the SNI data
loop, GPU + host-RAM tracing, the full-parameter assertion, checkpointing, and
the summary row — so each strategy is just its optimizer plus a thin call here.

DeepSpeed/FSDP (offload engines) and MeZO (no backward) have their own modules.

Note on precision: a 7B fp32 master copy (~29 GB) does not fit beside bf16
weights on a 32 GB GPU, so the on-GPU full-FT runs train bf16 weights directly
(equivalent to the `drop_fp32_master` lever always being on). This is a known
quality caveat recorded in the findings, not an oversight.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, get_scheduler

from ...config import RunConfig
from ...data.sni import DataCollatorForSupervised, get_dataset
from ...methods.build import _load_tokenizer
from ...train.params import count_parameters
from ...train.run_logger import RunLogger
from ...utils.vram import HostRamTracer, MemoryTracer, bytes_to_gb, reset_peak_memory

PHASE05_TRACE_DIR = Path("results/phase05/mem_trace")


def load_bf16_model(config: RunConfig):
    """Load the base in bf16 on the GPU with every weight trainable.

    Gradient checkpointing is applied when the lever is set (the on-GPU 7B runs
    need it to keep activation memory down).
    """
    tok = _load_tokenizer(config)
    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16)
    model.config.use_cache = False
    if config.levers.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    for p in model.parameters():
        p.requires_grad_(True)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tok


def _num_opt_steps(n_examples: int, config: RunConfig) -> int:
    if config.hparams.max_steps and config.hparams.max_steps > 0:
        return config.hparams.max_steps
    eff = config.hparams.batch_size * config.hparams.grad_accum
    return int(math.ceil(n_examples / eff) * config.hparams.num_epochs)


def run_manual_loop(
    config: RunConfig,
    model,
    tok,
    optimizer,
    *,
    label: str,
    scheduler=None,
    fused_backward: Callable[[torch.Tensor], None] | None = None,
    on_opt_step: Callable[[int], None] | None = None,
    save_checkpoint: bool = False,  # benchmark runs don't persist 14 GB/run; Sprint 7 evals inline
    full_param_check: bool = True,
) -> dict:
    """Run the fixed protocol on ``model`` and return the trade-off row.

    ``fused_backward(loss)`` — if given (LOMO), replaces ``loss.backward()`` +
    ``optimizer.step()``: the callable computes grads and applies the update in
    one fused pass (the raw, un-divided loss is passed — LOMO updates every
    micro-step with no gradient accumulation).
    ``on_opt_step(step)`` — optional post-step hook (e.g. BAdam block switch).
    ``full_param_check`` — assert every param is trainable at the end; set False
    for BAdam, which freezes all but the active block (full coverage over the
    run, not simultaneously).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = RunLogger(config)
    tracer = MemoryTracer()
    ram_tracer = HostRamTracer().start()
    reset_peak_memory()

    bundle = get_dataset(
        config.task, tok,
        max_seq_len=config.hparams.max_seq_len,
        max_train_samples=config.hparams.max_train_samples,
        seed=config.hparams.seed,
        max_eval_samples=config.eval.max_eval_samples,
    )
    collate = DataCollatorForSupervised(tok)
    loader = torch.utils.data.DataLoader(
        bundle.train, batch_size=config.hparams.batch_size, shuffle=True,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(config.hparams.seed),
        drop_last=True,
    )
    total_steps = _num_opt_steps(len(bundle.train), config)

    pinfo = count_parameters(model)
    logger.set_summary(**pinfo, total_train_examples=len(bundle.train),
                       planned_steps=total_steps, **bundle.summary())
    print(f"[{label}] {config.name}: {total_steps} opt-steps, "
          f"{pinfo['trainable_params']:,} trainable ({pinfo['pct_params']}%)")

    grad_accum = config.hparams.grad_accum
    log_every = config.logging.log_every
    mem_every = max(1, config.logging.mem_trace_every)
    model.train()

    step = 0
    final_loss = float("nan")
    interval_loss = 0.0
    interval_micro = 0
    interval_tokens = 0
    t0 = time.time()
    interval_t = time.time()
    done = False
    while not done:
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(**batch)
            loss = out.loss / grad_accum
            interval_loss += out.loss.item()
            interval_micro += 1
            interval_tokens += int(batch["attention_mask"].sum().item())

            if fused_backward is not None:
                # LOMO: fused backward+update every micro-step (no grad accum).
                fused_backward(loss)
                boundary = True
            else:
                loss.backward()
                boundary = (micro + 1) % grad_accum == 0
                if boundary:
                    torch.nn.utils.clip_grad_norm_(
                        (p for p in model.parameters() if p.requires_grad), 1.0
                    )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            if boundary:
                step += 1
                if on_opt_step is not None:
                    on_opt_step(step)
                if step % mem_every == 0:
                    tracer.record(step)
                    ram_tracer.record(step)
                if step % log_every == 0 or step == total_steps:
                    dt = time.time() - interval_t
                    avg_loss = interval_loss / max(1, interval_micro)
                    final_loss = avg_loss
                    snap_alloc = bytes_to_gb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0
                    logger.log_metrics(step, {
                        "train_loss": round(avg_loss, 5),
                        "gpu_mem_gb": round(snap_alloc, 4),
                        "ram_gb": round(ram_tracer.ram_gb[-1], 4) if len(ram_tracer) else 0.0,
                        "tokens_per_sec": round(interval_tokens / dt, 1) if dt > 0 else 0.0,
                        "step_time_s": round(dt / max(1, log_every), 4),
                    })
                    interval_loss = 0.0
                    interval_micro = 0
                    interval_tokens = 0
                    interval_t = time.time()
                if step >= total_steps:
                    done = True
                    break
        if not done and step == 0:
            # avoid an infinite loop if the loader was too small to step once
            break

    wallclock = time.time() - t0
    ram_tracer.stop()

    # full-parameter check — skipped for BAdam, which freezes all but the active
    # block at any instant (full coverage is over the run, not simultaneously).
    if full_param_check:
        assert pinfo["trainable_params"] == pinfo["total_params"], (
            f"full-FT expects all params trainable, got "
            f"{pinfo['trainable_params']}/{pinfo['total_params']}"
        )

    if len(tracer) == 0:
        tracer.record(max(step, 1))
    if len(ram_tracer) == 0:
        ram_tracer.record(max(step, 1))
    trace_path = PHASE05_TRACE_DIR / f"{config.name}.csv"
    ram_trace_path = PHASE05_TRACE_DIR / f"{config.name}.ram.csv"
    tracer.save_csv(trace_path)
    ram_tracer.save_csv(ram_trace_path)
    logger.log_artifact_path(trace_path, "mem_trace")
    logger.log_artifact_path(ram_trace_path, "ram_trace")

    eff_batch = config.hparams.batch_size * config.hparams.grad_accum
    steps_per_epoch = max(1, math.ceil(len(bundle.train) / eff_batch))
    if save_checkpoint:
        ckpt_dir = config.output_dir / "checkpoint"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tok.save_pretrained(str(ckpt_dir))

    logger.set_summary(
        method=config.method, base_model=config.base_model, task=config.task,
        peak_vram_gb=round(tracer.peak_gb, 4),
        peak_reserved_gb=round(tracer.peak_reserved_gb, 4),
        peak_ram_gb=round(ram_tracer.peak_ram_gb, 4),
        peak_ram_delta_gb=round(ram_tracer.peak_ram_delta_gb, 4),
        baseline_ram_gb=round(ram_tracer.baseline_ram_gb, 4),
        final_train_loss=round(final_loss, 5),
        wallclock_s=round(wallclock, 2),
        wallclock_per_step_s=round(wallclock / max(1, step), 4),
        wallclock_per_epoch_s=round(wallclock * steps_per_epoch / max(1, step), 2),
        steps=step,
        mem_trace=str(trace_path),
    )
    summary = dict(logger.summary)
    logger.finish()

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
