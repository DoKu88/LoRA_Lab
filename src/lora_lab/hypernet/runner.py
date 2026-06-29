"""Build the frozen base + hypernetwork and run the meta-training loop.

Shared orchestration for both training entrypoints (reconstruction and SFT):
load the (optionally 4-bit) frozen base, build the encoder + generator, optionally
warm-start, run ``meta_train``, and save the hypernetwork checkpoint + run
summary. Each entrypoint supplies the sampler for its objective.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .apply import target_specs
from .encoder import MeanPoolEncoder
from .logging import build_run_logger
from .meta_train import assert_run_allowed, meta_train
from .model import HyperLoRAGenerator


def build_components(cfg, *, allow_gpu):
    """Load (base, tokenizer, encoder, generator, specs). Guards GPU first."""
    assert_run_allowed(cfg, allow_gpu)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb,
                                                    device_map={"": 0})
    else:
        dtype = torch.float32 if cfg.device == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(cfg.device)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    if cfg.gradient_checkpointing and cfg.objective == "sft":
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    encoder = MeanPoolEncoder(cfg.encoder_model, device=cfg.device)
    specs = target_specs(base, cfg.target_modules)
    gen = HyperLoRAGenerator(specs, task_dim=encoder.dim, rank=cfg.rank, alpha=cfg.alpha,
                             parameterization=cfg.parameterization, layer_dim=cfg.layer_dim,
                             module_dim=cfg.module_dim, trunk_hidden=cfg.trunk_hidden)
    return base, tok, encoder, gen, specs


def run_training(cfg, sampler_factory, *, allow_gpu, steps=None, stage=None):
    """Build everything, run ``meta_train``, save the checkpoint + run summary.

    ``sampler_factory(cfg, base, tok, specs)`` returns the sampler for the
    objective; the caller (the entrypoint) decides which one. Returns the run
    summary dict.
    """
    base, tok, encoder, gen, specs = build_components(cfg, allow_gpu=allow_gpu)
    if cfg.warmup_from and Path(cfg.warmup_from).exists():
        gen.load_state_dict(torch.load(cfg.warmup_from, map_location=cfg.device))
        print(f"[train] warm-started from {cfg.warmup_from}")

    sampler = sampler_factory(cfg, base, tok, specs)
    n = steps or cfg.max_steps
    print(f"[train] {cfg.name}: {n} steps, objective={cfg.objective}, "
          f"params={gen.num_params():,}, targets={len(specs)}, sampler={type(sampler).__name__}")

    logger = build_run_logger(cfg, stage=stage or cfg.objective)
    losses = meta_train(gen, base, cfg.target_modules, sampler, encoder,
                        steps=n, lr=cfg.lr, objective=cfg.objective, device=cfg.device,
                        batch_size=cfg.batch_size, logger=logger, progress=True)

    out_dir = logger.output_dir
    torch.save(gen.state_dict(), out_dir / "hypernet.pt")
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if cfg.device == "cuda" else 0.0
    logger.set_summary(name=cfg.name, objective=cfg.objective, parameterization=cfg.parameterization,
                       steps=n, hypernet_params=gen.num_params(), n_targets=len(specs),
                       loss_first=round(losses[0], 6), loss_last=round(losses[-1], 6),
                       peak_vram_gb=round(peak_vram_gb, 4))
    logger.log_artifact_path(out_dir / "hypernet.pt", "hypernet_checkpoint")
    logger.finish()
    print(f"[train] done: loss {losses[0]:.4f} -> {losses[-1]:.4f}; saved {out_dir}/hypernet.pt")
    return dict(logger.summary)
