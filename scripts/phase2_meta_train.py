#!/usr/bin/env python
"""Phase 2 — meta-training entrypoint (turnkey, gated).

Ties HyperConfig + encoder + HyperLoRAGenerator + frozen base + sampler + the
meta_train loop into one command. The GPU / 4-bit (Mistral-7B SFT) path is gated
behind ``--allow-gpu`` so it is never launched autonomously.

    # CPU plumbing smoke (offline; SmolLM2 as both base and encoder, synthetic data)
    CUDA_VISIBLE_DEVICES="" python scripts/phase2_meta_train.py \
        --config configs/phase2/tiny-plumbing.yaml --synthetic --steps 5

    # the real reconstruction warmup (needs the library adapters + a GPU)
    python scripts/phase2_meta_train.py --config configs/phase2/recon-warmup.yaml --allow-gpu

    # the gate run — Mistral-7B SFT (launch only after design review)
    python scripts/phase2_meta_train.py --config configs/phase2/sft-mistral.yaml --allow-gpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.hypernet.apply import target_specs  # noqa: E402
from lora_lab.hypernet.config import HyperConfig  # noqa: E402
from lora_lab.hypernet.encoder import MeanPoolEncoder  # noqa: E402
from lora_lab.hypernet.meta_train import (  # noqa: E402
    SyntheticReconSampler, assert_run_allowed, meta_train,
)
from lora_lab.hypernet.model import HyperLoRAGenerator  # noqa: E402


def build_components(cfg: HyperConfig, *, allow_gpu: bool):
    """Load (base, tokenizer, encoder, generator, specs). Guards GPU first."""
    assert_run_allowed(cfg, allow_gpu)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cfg.load_in_4bit:
        from transformers import BitsAndBytesConfig
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
    gen = HyperLoRAGenerator(specs, d_task=encoder.dim, r=cfg.r, alpha=cfg.alpha,
                             parameterization=cfg.parameterization, d_layer=cfg.d_layer,
                             d_module=cfg.d_module, trunk_hidden=cfg.trunk_hidden)
    return base, tok, encoder, gen, specs


def build_sampler(cfg: HyperConfig, base, tok, specs, *, synthetic: bool):
    if synthetic:
        return SyntheticReconSampler(specs, r=cfg.r, seed=cfg.seed)
    if cfg.objective == "reconstruction":
        from lora_lab.hypernet.samplers import LibraryReconSampler
        return LibraryReconSampler(cfg.split_path, cfg.library_path, seed=cfg.seed)
    from lora_lab.hypernet.samplers import SNISFTSampler
    return SNISFTSampler(cfg.split_path, cfg.library_path, tok,
                         batch_size=cfg.batch_size, max_seq_len=cfg.max_seq_len, seed=cfg.seed)


def run(cfg: HyperConfig, *, allow_gpu: bool, synthetic: bool, steps: int | None):
    base, tok, encoder, gen, specs = build_components(cfg, allow_gpu=allow_gpu)
    if cfg.warmup_from and Path(cfg.warmup_from).exists():
        gen.load_state_dict(torch.load(cfg.warmup_from, map_location=cfg.device))
        print(f"[meta-train] warm-started from {cfg.warmup_from}")
    sampler = build_sampler(cfg, base, tok, specs, synthetic=synthetic)
    n = steps or cfg.max_steps
    print(f"[meta-train] {cfg.name}: {n} steps, objective={cfg.objective}, "
          f"params={gen.num_params():,}, targets={len(specs)}, sampler={type(sampler).__name__}")
    losses = meta_train(gen, base, cfg.target_modules, sampler, encoder,
                        steps=n, lr=cfg.lr, objective=cfg.objective, device=cfg.device)

    out_dir = Path(cfg.output_root) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), out_dir / "hypernet.pt")
    summary = {"name": cfg.name, "objective": cfg.objective, "steps": n,
               "params": gen.num_params(), "loss_first": losses[0], "loss_last": losses[-1]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[meta-train] done: loss {losses[0]:.4f} -> {losses[-1]:.4f}; saved {out_dir}/hypernet.pt")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true",
                    help="required to run the cuda/4-bit (Mistral SFT) path")
    ap.add_argument("--synthetic", action="store_true",
                    help="use the synthetic recon sampler (CPU plumbing smoke)")
    args = ap.parse_args()
    cfg = HyperConfig.load(args.config)
    run(cfg, allow_gpu=args.allow_gpu, synthetic=args.synthetic, steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
