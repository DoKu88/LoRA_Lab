#!/usr/bin/env python
"""Phase 0.5 benchmark entrypoint (Sprint 1).

Run one full-FT technique under the fixed measurement protocol and emit its
trade-off-table row.

    # baseline sanity (small model — validates instrumentation)
    conda run -n lora_lab python scripts/benchmark.py \
        --protocol configs/phase05/_fixed_protocol.yaml \
        --set base_model=HuggingFaceTB/SmolLM2-135M hparams.max_steps=5

    # a real technique on Mistral-7B (offload needs CUDA_HOME; auto-set for
    # zero_* techniques if nvcc is found in the env)
    conda run -n lora_lab python scripts/benchmark.py \
        --protocol configs/phase05/_fixed_protocol.yaml \
        --set technique.name=zero_offload levers.use_8bit_adam=true
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from lora_lab.config import RunConfig, apply_overrides  # noqa: E402


def build_config(args) -> RunConfig:
    with open(args.protocol) as f:
        data = yaml.safe_load(f) or {}
    if args.set:
        apply_overrides(data, args.set)
    if args.wandb_mode is not None:
        data.setdefault("logging", {})["wandb_mode"] = args.wandb_mode
    return RunConfig.from_dict(data)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 0.5 technique benchmark")
    ap.add_argument("--protocol", default="configs/phase05/_fixed_protocol.yaml",
                    help="fixed measurement protocol YAML")
    ap.add_argument("--set", nargs="*", default=[],
                    help="dotted.key=value overrides (e.g. technique.name=galore)")
    ap.add_argument("--wandb-mode", default=None,
                    choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    config = build_config(args)
    print(f"== benchmark: {config.technique.name} on {config.base_model}")

    from lora_lab.phase05.benchmark import benchmark

    summary = benchmark(config)
    print("== BENCHMARK complete; row:")
    for k in ("technique", "fits", "peak_vram_gb", "peak_ram_gb",
              "peak_ram_delta_gb", "wallclock_per_epoch_s", "final_train_loss",
              "steps"):
        if k in summary:
            print(f"   {k}: {summary[k]}")
    # full row as JSON for the runner to capture
    print("ROW_JSON " + json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
