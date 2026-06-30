#!/usr/bin/env python
"""Train the hypernetwork by reconstruction.

Regress the generated LoRA (ΔW = scaling·B·A) onto each train-split library
LoRA's ΔW — no base-model forward pass. Produces the checkpoint the
generalization run warm-starts from.

    # smoke (random synthetic targets, tiny model)
    python scripts/train_reconstruction.py \
        --config configs/phase2/tiny-plumbing.yaml --synthetic --steps 5

    # the real reconstruction run (needs the library adapters + a GPU)
    python scripts/train_reconstruction.py --config configs/phase2/recon-warmup.yaml
"""

from __future__ import annotations

import argparse

from lora_lab.hypernet.config import HyperConfig
from lora_lab.hypernet.train import train


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--synthetic", action="store_true",
                    help="use random synthetic targets instead of library adapters (smoke)")
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

    train(cfg, steps=args.steps, synthetic=args.synthetic, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
