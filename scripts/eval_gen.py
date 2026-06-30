#!/usr/bin/env python
"""Read a reconstruction run's metrics.jsonl and report how the generation is
doing vs the trivial baselines — no model loading, just the logs.

Compares the live training/probe metrics against:
  predict-zero   output all zeros        (mean-L1 == mean|ΔW_target|)
  predict-mean   output the average LoRA (mean-L1 of tgt vs its cross-task mean)
  target dW_mag / target diversity       (what real reconstruction should reach)

Defaults are the 5-task overfit set (configs/phase2/overfit_split.yaml); override
with the flags for a different task set.

Usage:
    python scripts/eval_gen.py --run recon-overfit
    python scripts/eval_gen.py --metrics results/phase2/runs/recon-overfit/metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, YELLOW, RED, CYAN = "\033[32m", "\033[33m", "\033[31m", "\033[36m"


def load(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    train = [(r["step"], r["train/loss"]) for r in rows if "train/loss" in r]
    val = [(r["step"], r["val/loss"]) for r in rows if "val/loss" in r]
    probe = [r for r in rows if "gen/diversity" in r]
    return rows, train, val, probe


def fmt(x, ref):
    """Color a value by how it compares to a 'good' reference (closer = greener)."""
    if ref <= 0:
        return f"{x:.3e}"
    ratio = x / ref
    color = GREEN if ratio >= 0.8 else YELLOW if ratio >= 0.3 else RED
    return f"{color}{x:.3e}{RESET}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="recon-overfit", help="run name under --root")
    ap.add_argument("--metrics", default=None, help="explicit metrics.jsonl path (overrides --run)")
    ap.add_argument("--root", default="results/phase2/runs")
    ap.add_argument("--predict-zero", type=float, default=2.31e-4)
    ap.add_argument("--predict-mean", type=float, default=2.14e-4)
    ap.add_argument("--target-mag", type=float, default=2.31e-4)
    ap.add_argument("--target-task-std", type=float, default=2.57e-4)
    ap.add_argument("--target-diversity", type=float, default=1.11)
    ap.add_argument("--smooth", type=int, default=20, help="window for the smoothed train loss")
    args = ap.parse_args()

    path = Path(args.metrics) if args.metrics else Path(args.root) / args.run / "metrics.jsonl"
    if not path.exists():
        raise SystemExit(f"no metrics.jsonl at {path}")
    rows, train, val, probe = load(path)
    if not train:
        raise SystemExit(f"no train/loss in {path}")

    last_step = rows[-1]["step"]
    tl = [v for _, v in train]
    sm = sum(tl[-args.smooth:]) / len(tl[-args.smooth:])

    print(f"\n{BOLD}{CYAN}── generation eval: {path.parent.name} ──{RESET}")
    print(f"  metrics: {path}")
    print(f"  step {last_step}  |  {len(probe)} probe point(s)\n")

    print(f"  {BOLD}baselines & target{RESET} {DIM}(5-task overfit set){RESET}")
    print(f"    predict-zero  L1 = {args.predict_zero:.3e}   {DIM}(output zeros){RESET}")
    print(f"    predict-mean  L1 = {args.predict_mean:.3e}   {DIM}(output the average LoRA){RESET}")
    print(f"    target dW_mag    = {args.target_mag:.3e}")
    print(f"    target diversity = {args.target_diversity:.2f}\n")

    # ---- losses vs baselines ----
    last_val = val[-1][1] if val else float("nan")
    print(f"  {BOLD}where we stand{RESET}")
    print(f"    train L1 (mean-{args.smooth})  {sm:.3e}")
    print(f"    val   L1 (latest)     {last_val:.3e}")
    if sm < args.predict_mean:
        verdict = f"{GREEN}below predict-mean — starting to reconstruct{RESET}"
    elif sm <= args.predict_zero:
        verdict = f"{YELLOW}between predict-mean and predict-zero — at the do-nothing floor{RESET}"
    else:
        verdict = f"{RED}above predict-zero — worse than outputting nothing{RESET}"
    print(f"    -> {verdict}\n")

    # ---- generation trio vs target ----
    if probe:
        p = probe[-1]
        mag, std, div = p["gen/dW_mag"], p["gen/task_std"], p["gen/diversity"]
        print(f"  {BOLD}generation (latest probe @ step {p['step']}){RESET}")
        print(f"    gen/dW_mag     {fmt(mag, args.target_mag)}   "
              f"{DIM}= {100*mag/args.target_mag:4.0f}% of target {args.target_mag:.2e}{RESET}")
        print(f"    gen/task_std   {fmt(std, args.target_task_std)}   "
              f"{DIM}= {100*std/args.target_task_std:4.0f}% of target {args.target_task_std:.2e}{RESET}")
        print(f"    gen/diversity  {fmt(div, args.target_diversity)}   "
              f"{DIM}= {100*div/args.target_diversity:4.0f}% of target {args.target_diversity:.2f}{RESET}")

        # ---- probe history (is it climbing?) ----
        print(f"\n  {BOLD}probe history{RESET}")
        print(f"    {'step':>6} {'train/L1':>10} {'val/L1':>10} {'dW_mag':>11} {'task_std':>11} {'diversity':>10}")
        for r in probe:
            v = f"{r['val/loss']:.3e}" if "val/loss" in r else "    -    "
            print(f"    {r['step']:>6} {r.get('train/loss', float('nan')):>10.3e} {v:>10} "
                  f"{r['gen/dW_mag']:>11.3e} {r['gen/task_std']:>11.3e} {r['gen/diversity']:>10.3f}")
    else:
        print(f"  {YELLOW}no gen/* probe points yet (probe_every not reached / not enabled){RESET}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
