#!/usr/bin/env python
"""Run the Phase 0 comparison matrix: {method × model × task}, smallest-first.

Trains each cell, scores it on the held-out split, and (at the end) writes the
comparison table/dataset + the overlaid memory plots. Single GPU => cells run
serially; the ladder is walked smallest-first so plumbing fails fast and the
big runs come last. A per-cell OOM is caught and recorded, not fatal.

    # quick smoke over the two smallest models, capped train size
    conda run -n lora_lab python scripts/run_matrix.py \
        --models HuggingFaceTB/SmolLM2-135M Qwen/Qwen2.5-0.5B-Instruct \
        --tasks task1564_triviaqa_answer_generation \
        --max-train-samples 200 --max-eval-samples 50

    # ungated ladder, all default tasks
    conda run -n lora_lab python scripts/run_matrix.py --tier ungated

    # gated follow-on (Sprint 7) — needs HF_TOKEN + accepted licenses
    conda run -n lora_lab python scripts/run_matrix.py --tier gated

    # the full ladder (ungated + gated) in one go — needs HF_TOKEN
    conda run -n lora_lab python scripts/run_matrix.py --tier all

    # drive the whole sweep from a YAML file (CLI flags still override)
    conda run -n lora_lab python scripts/run_matrix.py --config configs/matrix/run-matrix.yaml

The resolved config for every run is saved to <output_root>/_matrix/ so the
sweep round-trips: load -> run -> reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_lab.eval.evaluate import evaluate_checkpoint  # noqa: E402
from lora_lab.eval.plot import plot_memory_traces  # noqa: E402
from lora_lab.eval.table import collect_rows, write_table  # noqa: E402
from lora_lab.matrix import (  # noqa: E402
    MODELS,
    MatrixConfig,
    build_config,
)
from lora_lab.train.trainer import train  # noqa: E402


def run_cell(model_id, task, method, mcfg: MatrixConfig) -> dict:
    cfg = build_config(
        model_id, task, method,
        max_train_samples=mcfg.max_train_samples,
        max_steps=mcfg.max_steps,
        num_epochs=mcfg.epochs,
        max_eval_samples=mcfg.max_eval_samples,
        wandb_mode=mcfg.wandb_mode,
        wandb_project=mcfg.wandb_project,
        wandb_entity=mcfg.wandb_entity,
        lora_rank=mcfg.lora_rank,
    )
    print(f"\n{'=' * 78}\n[cell] {cfg.name}\n{'=' * 78}")
    try:
        summary = train(cfg)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print(f"[OOM] {cfg.name}: {e}")
        return {"name": cfg.name, "status": "oom", "method": method,
                "base_model": model_id, "task": task}
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {cfg.name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"name": cfg.name, "status": "error", "error": str(e),
                "method": method, "base_model": model_id, "task": task}

    # peak guard (the matrix invariant: never exceed 32 GB)
    peak = summary.get("peak_vram_gb", 0.0)
    if peak > 32.0:
        print(f"[WARN] {cfg.name} peak {peak} GB exceeded 32 GB ceiling!")

    if not mcfg.skip_eval:
        try:
            res = evaluate_checkpoint(
                cfg.output_dir / "checkpoint", cfg.base_model, task,
                summary.get("metric", "rougeL"),
                max_eval_samples=mcfg.max_eval_samples,
            )
            summary["eval_metric"] = round(res["score"], 5)
            summary["eval_metric_name"] = res["metric"]
            summary["eval_n"] = res["n"]
            summary["eval_samples"] = {
                "predictions": res["sample_predictions"],
                "references": res["sample_references"],
            }
            sp = cfg.output_dir / "summary.json"
            sp.write_text(json.dumps(summary, indent=2, default=str))
            print(f"[eval] {cfg.name}: {res['metric']}={res['score']:.4f} (n={res['n']})")
        except Exception as e:  # noqa: BLE001
            print(f"[eval-FAIL] {cfg.name}: {type(e).__name__}: {e}")
    summary["status"] = "ok"
    return summary


def build_matrix_config(argv=None) -> MatrixConfig:
    """Resolve a MatrixConfig from an optional --config YAML + CLI overrides.

    Precedence: dataclass defaults < --config YAML < explicit CLI flags. CLI
    args use SUPPRESS defaults so only flags the user actually passed land in
    the namespace and override the YAML; unset flags leave the YAML/default
    value untouched.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="YAML matrix config; CLI flags below override its fields")
    ap.add_argument("--models", nargs="*", default=argparse.SUPPRESS,
                    help=f"HF model ids from {MODELS} (default: ungated ladder)")
    ap.add_argument("--tier", choices=["ungated", "gated", "all"], default=argparse.SUPPRESS)
    ap.add_argument("--tasks", nargs="*", default=argparse.SUPPRESS)
    ap.add_argument("--methods", nargs="*", default=argparse.SUPPRESS)
    ap.add_argument("--max-train-samples", dest="max_train_samples", type=int,
                    default=argparse.SUPPRESS)
    ap.add_argument("--max-steps", dest="max_steps", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--epochs", type=float, default=argparse.SUPPRESS)
    ap.add_argument("--max-eval-samples", dest="max_eval_samples", type=int,
                    default=argparse.SUPPRESS)
    ap.add_argument("--lora-rank", dest="lora_rank", type=int, default=argparse.SUPPRESS)
    ap.add_argument("--wandb-mode", dest="wandb_mode",
                    choices=["online", "offline", "disabled"], default=argparse.SUPPRESS)
    ap.add_argument("--wandb-project", dest="wandb_project", default=argparse.SUPPRESS)
    ap.add_argument("--wandb-entity", dest="wandb_entity", default=argparse.SUPPRESS,
                    help="W&B entity (team/user); omit to use your default entity")
    ap.add_argument("--skip-eval", dest="skip_eval", action="store_true",
                    default=argparse.SUPPRESS)
    ap.add_argument("--no-aggregate", dest="no_aggregate", action="store_true",
                    default=argparse.SUPPRESS, help="skip table/plot build")
    args = ap.parse_args(argv)

    overrides = vars(args)
    config_path = overrides.pop("config", None)
    mcfg = MatrixConfig.load(config_path) if config_path else MatrixConfig()
    for key, value in overrides.items():  # only explicitly-passed flags remain
        setattr(mcfg, key, value)
    mcfg.validate()
    return mcfg


def main() -> int:
    mcfg = build_matrix_config()
    models = mcfg.resolved_models()

    # persist the resolved sweep config next to the run logs (round-trippable)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg_path = mcfg.save(Path(mcfg.output_root) / "_matrix" / f"matrix-{stamp}.yaml")
    print(f"[matrix] config -> {cfg_path}")
    print(f"[matrix] models={models} tasks={mcfg.tasks} methods={mcfg.methods}")

    results = []
    for model_id in models:  # outer: model (smallest first)
        for task in mcfg.tasks:
            for method in mcfg.methods:
                results.append(run_cell(model_id, task, method, mcfg))

    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") != "ok"]
    print(f"\n[matrix] {len(ok)} ok, {len(bad)} not-ok")
    for r in bad:
        print(f"   {r.get('status')}: {r.get('name')}")

    if not mcfg.no_aggregate:
        rows = collect_rows()
        if rows:
            paths = write_table(rows)
            print(f"[matrix] table -> {paths}")
        plot_memory_traces()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
