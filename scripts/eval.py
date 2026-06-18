#!/usr/bin/env python
"""Evaluate a finished run's checkpoint and write eval_metric into its summary.

Reads results/runs/{name}/summary.json (method/base_model/task), generates on
the held-out test split, scores with the task's metric, and merges
``eval_metric`` (+ metric name and samples) back into summary.json so
build_table picks it up.

    conda run -n lora_lab python scripts/eval.py --run results/runs/lora-...
    conda run -n lora_lab python scripts/eval.py --all          # every run dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.data.sni import load_tasks_manifest  # noqa: E402
from lora_lab.eval.evaluate import evaluate_checkpoint  # noqa: E402

RUNS_DIR = Path("results/runs")


def eval_run(run_dir: Path, manifest: dict, max_eval_samples: int) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"[skip] no summary.json in {run_dir}")
        return None
    summary = json.loads(summary_path.read_text())
    task = summary["task"]
    metric = manifest[task].metric if task in manifest else summary.get("metric", "rougeL")
    ckpt = run_dir / "checkpoint"
    if not ckpt.exists():
        print(f"[skip] no checkpoint in {run_dir}")
        return None

    print(f"[eval] {run_dir.name} task={task} metric={metric}")
    result = evaluate_checkpoint(
        ckpt, summary["base_model"], task, metric, max_eval_samples=max_eval_samples
    )
    summary["eval_metric"] = round(result["score"], 5)
    summary["eval_metric_name"] = metric
    summary["eval_n"] = result["n"]
    summary["eval_samples"] = {
        "predictions": result["sample_predictions"],
        "references": result["sample_references"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"       {metric} = {result['score']:.4f} over {result['n']} examples")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="a single results/runs/<name> dir")
    ap.add_argument("--all", action="store_true", help="evaluate every run dir")
    ap.add_argument("--max-eval-samples", type=int, default=200)
    args = ap.parse_args()

    manifest = load_tasks_manifest()
    if args.all:
        dirs = sorted(d for d in RUNS_DIR.iterdir() if (d / "summary.json").exists())
    elif args.run:
        dirs = [Path(args.run)]
    else:
        ap.error("pass --run <dir> or --all")

    for d in dirs:
        eval_run(d, manifest, args.max_eval_samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
