#!/usr/bin/env python
"""Train the hypernetwork by SFT.

Generate a LoRA from a task description, apply it to the frozen 4-bit base, run
the task batch, and backprop the task cross-entropy THROUGH the base into the
hypernetwork. Warm-starts from a reconstruction checkpoint if ``warmup_from`` is
set in the config. GPU-heavy.

    python scripts/train_sft.py --config configs/phase2/sft-mistral.yaml
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
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default=None,
                    help="override the config's wandb_mode for this run")
    ap.add_argument("--stage", default=None, help="W&B group tag")
    args = ap.parse_args()

    cfg = HyperConfig.load(args.config)
    if cfg.objective != "sft":
        raise SystemExit(f"train_sft.py expects objective=sft, got {cfg.objective!r}")
    if args.wandb:
        cfg.wandb_mode = args.wandb

    train(cfg, steps=args.steps, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
