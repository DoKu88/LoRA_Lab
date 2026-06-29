#!/usr/bin/env python
"""Train the hypernetwork by reconstruction.

Regress the generated LoRA (ΔW = scaling·B·A) onto each train-split library
LoRA's ΔW — no base-model forward pass. Produces the checkpoint the SFT run
warm-starts from.

    # CPU smoke (random synthetic targets, offline)
    CUDA_VISIBLE_DEVICES="" python scripts/train_reconstruction.py \
        --config configs/phase2/tiny-plumbing.yaml --synthetic --steps 5

    # the real reconstruction run (needs the library adapters + a GPU)
    python scripts/train_reconstruction.py --config configs/phase2/recon-warmup.yaml --allow-gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.hypernet.config import HyperConfig  # noqa: E402
from lora_lab.hypernet.meta_train import SyntheticReconSampler  # noqa: E402
from lora_lab.hypernet.runner import run_training  # noqa: E402
from lora_lab.hypernet.samplers import LibraryReconSampler  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true",
                    help="required to run the cuda/4-bit path")
    ap.add_argument("--synthetic", action="store_true",
                    help="use random synthetic targets instead of library adapters (CPU smoke)")
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default=None,
                    help="override the config's wandb_mode for this run")
    ap.add_argument("--stage", default=None, help="W&B group tag")
    args = ap.parse_args()

    cfg = HyperConfig.load(args.config)
    if cfg.objective != "reconstruction":
        raise SystemExit(
            f"train_reconstruction.py expects objective=reconstruction, got {cfg.objective!r}"
        )
    if args.wandb:
        cfg.wandb_mode = args.wandb

    def sampler_factory(cfg, base, tok, specs):
        if args.synthetic:
            return SyntheticReconSampler(specs, rank=cfg.rank, seed=cfg.seed)
        return LibraryReconSampler(cfg.split_path, cfg.library_path, seed=cfg.seed)

    run_training(cfg, sampler_factory, allow_gpu=args.allow_gpu, steps=args.steps, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
