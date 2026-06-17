#!/usr/bin/env python
"""Unified training entrypoint (Sprints 3 + 4).

    # synthetic run, no GPU — validates the harness end to end
    conda run -n lora_lab python scripts/train.py --config configs/runs/example.yaml --dry-run

    # real run
    conda run -n lora_lab python scripts/train.py --config configs/runs/lora-smol-trivia.yaml

    # ad-hoc config with overrides (no file)
    conda run -n lora_lab python scripts/train.py \
        --method qlora --base-model HuggingFaceTB/SmolLM2-135M \
        --task task1564_triviaqa_answer_generation \
        --set hparams.max_steps=20 logging.wandb_mode=offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.config import RunConfig, apply_overrides  # noqa: E402


def build_config(args) -> RunConfig:
    if args.config:
        with open(args.config) as f:
            import yaml

            data = yaml.safe_load(f) or {}
    else:
        data = {}
    # top-level conveniences
    for k_cli, k_cfg in (("method", "method"), ("base_model", "base_model"), ("task", "task")):
        v = getattr(args, k_cli)
        if v is not None:
            data[k_cfg] = v
    if args.set:
        apply_overrides(data, args.set)
    if args.wandb_mode is not None:
        data.setdefault("logging", {})["wandb_mode"] = args.wandb_mode
    return RunConfig.from_dict(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="LoRA_Lab training entrypoint")
    ap.add_argument("--config", default=None, help="YAML run config")
    ap.add_argument("--method", default=None, choices=["full_ft", "lora", "qlora"])
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted.key=value overrides")
    ap.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    ap.add_argument("--dry-run", action="store_true", help="synthetic run, no GPU/model")
    args = ap.parse_args()

    config = build_config(args)
    print(f"== run: {config.name} (method={config.method}, model={config.base_model})")
    print(f"== output_dir: {config.output_dir}")

    if args.dry_run:
        from lora_lab.train.dryrun import run_dry

        summary = run_dry(config)
        print("== DRY RUN complete; summary:")
        for k, v in summary.items():
            print(f"   {k}: {v}")
        return 0

    from lora_lab.train.trainer import train

    summary = train(config)
    print("== RUN complete; summary:")
    for k, v in summary.items():
        print(f"   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
