#!/usr/bin/env python
"""Train the hypernetwork by SFT.

Generate a LoRA from a task description, apply it to the frozen 4-bit base, run
the task batch, and backprop the task cross-entropy THROUGH the base into the
hypernetwork. Warm-starts from a reconstruction checkpoint if ``warmup_from`` is
set in the config. GPU-heavy; gated behind ``--allow-gpu``.

    python scripts/train_sft.py --config configs/phase2/sft-mistral.yaml --allow-gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.hypernet.config import HyperConfig  # noqa: E402
from lora_lab.hypernet.train import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--allow-gpu", action="store_true",
                    help="required to run the cuda/4-bit path")
    ap.add_argument("--wandb", choices=["online", "offline", "disabled"], default=None,
                    help="override the config's wandb_mode for this run")
    ap.add_argument("--stage", default=None, help="W&B group tag")
    args = ap.parse_args()

    cfg = HyperConfig.load(args.config)
    if cfg.objective != "sft":
        raise SystemExit(f"train_sft.py expects objective=sft, got {cfg.objective!r}")
    if args.wandb:
        cfg.wandb_mode = args.wandb

    train(cfg, allow_gpu=args.allow_gpu, steps=args.steps, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
