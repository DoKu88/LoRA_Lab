"""MeZO strategy (Sprint 5/8) — zeroth-order, forward-only.

MeZO estimates the gradient from two forward passes with a shared random
perturbation (no backprop), so memory is inference-level: weights + activations,
**no gradients, no optimizer state**. The trade is convergence — zeroth-order is
noisy and needs many more steps than first-order methods, so in a fixed 50-step
budget its quality is expected to be low (the point is to measure the memory
floor and the per-step cost, and confirm it *runs*).

This loop is fundamentally different from the others (two forwards, no backward,
no optimizer), so it does not use the shared `run_manual_loop`; it reuses the
same measurement + eval scaffolding directly.

Algorithm (Malladi et al. 2023): for direction z ~ N(0, I) regenerated from a
per-step seed,
    L+ = f(θ + εz),  L− = f(θ − εz),  ĝ = (L+ − L−) / (2ε)
    θ ← θ − lr · ĝ · z          (z regenerated from the same seed)
"""

from __future__ import annotations

import math
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


def _perturb(params, scale: float, seed: int) -> None:
    """θ += scale · z, with z ~ N(0,1) regenerated deterministically from seed.

    Reseeds once, then draws one z per param in a fixed order so the *same* z is
    reproduced for the + pass, the − pass, and the update. z is never stored.
    """
    if scale == 0.0:
        return
    g = torch.Generator(device=params[0].device)
    g.manual_seed(seed)
    for p in params:
        z = torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)
        p.data.add_(z, alpha=scale)


def run_mezo(config: RunConfig) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = RunLogger(config)
    tracer = MemoryTracer()
    ram_tracer = HostRamTracer().start()
    reset_peak_memory()

    tok = _load_tokenizer(config)
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.eval()  # forward-only; no autograd graph needed
    for p in model.parameters():
        p.requires_grad_(False)  # MeZO stores no grads — this is the memory win
    if device == "cuda":
        model = model.to(device)
    params = [p for p in model.parameters()]

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
        generator=torch.Generator().manual_seed(config.hparams.seed), drop_last=True,
    )
    total_steps = _num_opt_steps(len(bundle.train), config)
    eps = config.technique.mezo_eps
    lr = config.technique.mezo_lr if config.technique.mezo_lr else config.hparams.lr
    pinfo = count_parameters(model)
    print(f"[mezo] {config.name}: {total_steps} steps, eps={eps}, lr={lr}")

    def _loss(batch) -> float:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            return float(model(**batch).loss)

    step = 0
    final_loss = float("nan")
    t0 = time.time()
    interval_t = time.time()
    log_every = config.logging.log_every
    rng = torch.Generator().manual_seed(config.hparams.seed)
    done = False
    while not done:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
            _perturb(params, +eps, seed)
            loss_plus = _loss(batch)
            _perturb(params, -2 * eps, seed)
            loss_minus = _loss(batch)
            _perturb(params, +eps, seed)  # restore θ
            projected_grad = (loss_plus - loss_minus) / (2 * eps)
            _perturb(params, -lr * projected_grad, seed)  # θ -= lr·ĝ·z
            step += 1
            final_loss = 0.5 * (loss_plus + loss_minus)
            tracer.record(step)
            ram_tracer.record(step)
            if step % log_every == 0 or step == total_steps:
                dt = time.time() - interval_t
                logger.log_metrics(step, {
                    "train_loss": round(final_loss, 5),
                    "gpu_mem_gb": round(bytes_to_gb(torch.cuda.memory_allocated()) if device == "cuda" else 0.0, 4),
                    "ram_gb": round(ram_tracer.ram_gb[-1], 4) if len(ram_tracer) else 0.0,
                    "projected_grad": round(projected_grad, 6),
                    "step_time_s": round(dt / max(1, log_every), 4),
                })
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

    eval_score = None
    if config.eval.max_eval_samples > 0 and len(bundle.test_eval) > 0:
        from ...eval.evaluate import evaluate_inline

        try:
            scored = evaluate_inline(model, tok, bundle, config.eval.metric,
                                     max_new_tokens=config.eval.gen_max_new_tokens,
                                     batch_size=config.eval.batch_size)
            eval_score = round(scored["score"], 5)
            print(f"[mezo] eval {config.eval.metric}={eval_score}")
        except Exception as e:  # noqa: BLE001
            print(f"[mezo] eval failed ({type(e).__name__}: {e})")

    logger.set_summary(
        **pinfo, method=config.method, base_model=config.base_model, task=config.task,
        peak_vram_gb=round(tracer.peak_gb, 4),
        peak_reserved_gb=round(tracer.peak_reserved_gb, 4),
        peak_ram_gb=round(ram_tracer.peak_ram_gb, 4),
        peak_ram_delta_gb=round(ram_tracer.peak_ram_delta_gb, 4),
        baseline_ram_gb=round(ram_tracer.baseline_ram_gb, 4),
        final_train_loss=round(final_loss, 5),
        eval_metric=config.eval.metric, eval_score=eval_score,
        wallclock_s=round(wallclock, 2),
        wallclock_per_step_s=round(wallclock / max(1, step), 4),
        steps=step, mem_trace=str(trace_path),
    )
    summary = dict(logger.summary)
    logger.finish()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary
