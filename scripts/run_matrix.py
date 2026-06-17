#!/usr/bin/env python
"""Run the Phase 0 comparison matrix: {method × model × task}, smallest-first.

Trains each cell, scores it on the held-out split, and (at the end) writes the
comparison table/dataset + the overlaid memory plots. Single GPU => cells run
serially; the ladder is walked smallest-first so plumbing fails fast and the
big runs come last. A per-cell OOM is caught and recorded, not fatal.

    # quick smoke over the two smallest rungs, capped train size
    conda run -n lora_lab python scripts/run_matrix.py \
        --models tiny small --tasks task1564_triviaqa_answer_generation \
        --max-train-samples 200 --max-eval-samples 50

    # ungated ladder, all default tasks
    conda run -n lora_lab python scripts/run_matrix.py --models tiny small mid

    # gated follow-on (Sprint 7) — needs HF_TOKEN + accepted licenses
    conda run -n lora_lab python scripts/run_matrix.py --tier gated
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_lab.eval.evaluate import evaluate_checkpoint  # noqa: E402
from lora_lab.eval.plot import plot_memory_traces  # noqa: E402
from lora_lab.eval.table import collect_rows, write_table  # noqa: E402
from lora_lab.matrix import (  # noqa: E402
    GATED_TIERS,
    MODELS,
    UNGATED_TIERS,
    build_config,
    order_key,
)
from lora_lab.train.trainer import train  # noqa: E402

DEFAULT_TASKS = [
    "task1564_triviaqa_answer_generation",
    "task843_financial_phrasebank_classification",
    "task512_twitter_emotion_classification",
    "task1344_glue_entailment_classification",
    "task639_multi_woz_user_utterance_generation",
]
DEFAULT_METHODS = ["qlora", "lora", "full_ft"]


def run_cell(model_key, task, method, args) -> dict:
    cfg = build_config(
        model_key, task, method,
        max_train_samples=args.max_train_samples,
        max_steps=args.max_steps,
        num_epochs=args.epochs,
        max_eval_samples=args.max_eval_samples,
        wandb_mode=args.wandb_mode,
        lora_rank=args.lora_rank,
    )
    print(f"\n{'=' * 78}\n[cell] {cfg.name}\n{'=' * 78}")
    try:
        summary = train(cfg)
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        print(f"[OOM] {cfg.name}: {e}")
        return {"name": cfg.name, "status": "oom", "method": method,
                "base_model": MODELS[model_key], "task": task}
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {cfg.name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"name": cfg.name, "status": "error", "error": str(e),
                "method": method, "base_model": MODELS[model_key], "task": task}

    # peak guard (the matrix invariant: never exceed 32 GB)
    peak = summary.get("peak_vram_gb", 0.0)
    if peak > 32.0:
        print(f"[WARN] {cfg.name} peak {peak} GB exceeded 32 GB ceiling!")

    if not args.skip_eval:
        try:
            res = evaluate_checkpoint(
                cfg.output_dir / "checkpoint", cfg.base_model, task,
                summary.get("metric", "rougeL"),
                max_eval_samples=args.max_eval_samples,
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help=f"model keys from {list(MODELS)} (default: ungated ladder)")
    ap.add_argument("--tier", choices=["ungated", "gated"], default="ungated")
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--max-train-samples", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-eval-samples", type=int, default=200)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true", help="skip table/plot build")
    args = ap.parse_args()

    if args.models:
        models = args.models
    else:
        models = GATED_TIERS if args.tier == "gated" else UNGATED_TIERS

    # smallest-first
    models = sorted(models, key=order_key)
    print(f"[matrix] models={models} tasks={args.tasks} methods={args.methods}")

    results = []
    for model_key in models:  # outer: model (smallest first)
        for task in args.tasks:
            for method in args.methods:
                results.append(run_cell(model_key, task, method, args))

    ok = [r for r in results if r.get("status") == "ok"]
    bad = [r for r in results if r.get("status") != "ok"]
    print(f"\n[matrix] {len(ok)} ok, {len(bad)} not-ok")
    for r in bad:
        print(f"   {r.get('status')}: {r.get('name')}")

    if not args.no_aggregate:
        rows = collect_rows()
        if rows:
            paths = write_table(rows)
            print(f"[matrix] table -> {paths}")
        plot_memory_traces()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
