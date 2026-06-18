#!/usr/bin/env python
"""Phase 0.5 overnight matrix runner (Sprints 2-6).

Runs each technique (and, with --ablation, each lever-ablation cell) as an
ISOLATED subprocess so a single OOM/crash is captured as fits=no and the batch
continues — the unattended-run hygiene from the sprint plan. Rows are parsed
from each subprocess's ``ROW_JSON`` line and written to
``results/phase05/feasibility_table.{csv,parquet}`` + a markdown table.

    # full overnight matrix (technique rows + ablation)
    conda run -n lora_lab python scripts/run_phase05_matrix.py --ablation

    # quick shake-out (3 steps/run) to prove the runner end-to-end
    conda run -n lora_lab python scripts/run_phase05_matrix.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_PREFIX = os.path.dirname(os.path.dirname(sys.executable))
PROTOCOL = "configs/phase05/_fixed_protocol.yaml"
OUT_DIR = ROOT / "results" / "phase05"

# Cheap, high-value runs first so a truncated night still yields the core
# results (feasibility proof + fastest-route data), then the long tail.
TECHNIQUE_MATRIX = [
    ("baseline_paged8bit", {"technique.name": "baseline", "levers.use_8bit_adam": "true",
                            "levers.gradient_checkpointing": "true"}),
    ("galore", {"technique.name": "galore", "levers.gradient_checkpointing": "true"}),
    ("qgalore", {"technique.name": "qgalore", "levers.gradient_checkpointing": "true"}),
    ("lomo", {"technique.name": "lomo", "levers.gradient_checkpointing": "true"}),
    ("adalomo", {"technique.name": "adalomo", "levers.gradient_checkpointing": "true"}),
    # switch_every=5 so the 50-step benchmark exercises block-cycling (~10 blocks);
    # full coverage of all blocks needs many more steps — noted in findings.
    ("badam", {"technique.name": "badam", "technique.badam_switch_every": "5",
               "levers.gradient_checkpointing": "true"}),
    # offload (DeepSpeed fp32) — expected fits=no (host-RAM OOM); measured, captured.
    ("zero_offload_fp32", {"technique.name": "zero_offload", "levers.use_8bit_adam": "false",
                           "levers.gradient_checkpointing": "true"}),
]

# Lever ablation (Sprint 6): toggle one lever at a time vs the working anchor
# (paged-8bit). Each cell differs from the anchor by exactly one flag.
ABLATION_ANCHOR = {"technique.name": "baseline", "levers.use_8bit_adam": "true",
                   "levers.gradient_checkpointing": "true"}
ABLATION_CELLS = [
    ("abl_anchor", {}),                                          # full stack (reference)
    ("abl_no_8bit", {"levers.use_8bit_adam": "false"}),         # 8-bit Adam off (fp32)
    ("abl_no_gradckpt", {"levers.gradient_checkpointing": "false"}),  # grad ckpt off
]


# Method combinations / tuning (--combos): how to spend the headroom the winning
# techniques leave free, and whether levers stack onto them. Mostly LOMO (the
# recommended route) since its ~17 GB free is what invites combination.
COMBO_MATRIX = [
    # throughput: spend LOMO's headroom on a bigger batch (fused = no accum)
    ("lomo_bs4", {"technique.name": "lomo", "hparams.batch_size": "4",
                  "levers.gradient_checkpointing": "true"}),
    ("lomo_bs8", {"technique.name": "lomo", "hparams.batch_size": "8",
                  "levers.gradient_checkpointing": "true"}),
    # speed: LOMO is light enough it may not need checkpointing — drop the recompute
    ("lomo_nockpt", {"technique.name": "lomo", "levers.gradient_checkpointing": "false"}),
    # combine both: bigger batch AND no checkpointing
    ("lomo_bs4_nockpt", {"technique.name": "lomo", "hparams.batch_size": "4",
                         "levers.gradient_checkpointing": "false"}),
    # block-coordinate + 8-bit base optimizer (stack two memory tricks)
    ("badam_8bit", {"technique.name": "badam", "technique.badam_switch_every": "5",
                    "levers.use_8bit_adam": "true", "levers.gradient_checkpointing": "true"}),
    # GaLore rank sweep: lower rank = less projection state (memory/quality knob)
    ("galore_rank64", {"technique.name": "galore", "technique.galore_rank": "64",
                       "levers.gradient_checkpointing": "true"}),
]


def run_one(label: str, overrides: dict, steps: int, timeout_s: int,
            name_suffix: str = "") -> dict:
    sets = [f"{k}={v}" for k, v in overrides.items()]
    # Distinct run_name per label (+ task suffix) so output dirs + mem-traces
    # don't overwrite each other (every technique otherwise derives the same
    # method-model-task name) — the per-technique plot overlays need distinct
    # trace files, and a second task must not clobber the first task's traces.
    sets += [f"run_name=p05_{label}{name_suffix}", f"hparams.max_steps={steps}"]
    cmd = [sys.executable, "scripts/benchmark.py", "--protocol", PROTOCOL,
           "--wandb-mode", "offline", "--set", *sets]
    # expandable_segments reduces allocator fragmentation — GaLore's SVD spike
    # OOMed at "30 MiB free" without it despite fitting overall.
    env = dict(os.environ, CUDA_HOME=ENV_PREFIX,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    print(f"\n===== {label} =====  ({' '.join(sets)})")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True,
                              text=True, timeout=timeout_s)
        out = proc.stdout
        row = None
        for line in out.splitlines():
            if line.startswith("ROW_JSON "):
                row = json.loads(line[len("ROW_JSON "):])
        if row is None:
            tail = "\n".join((out + proc.stderr).splitlines()[-8:])
            return {"label": label, "fits": False, "status": "no_row",
                    "exit_code": proc.returncode, "error": tail, **overrides}
        row["label"] = label
        row["status"] = "ok"
        row["run_seconds"] = round(time.time() - t0, 1)
        return row
    except subprocess.TimeoutExpired:
        return {"label": label, "fits": False, "status": "timeout", **overrides}
    except Exception as e:  # noqa: BLE001
        return {"label": label, "fits": False, "status": "error",
                "error": f"{type(e).__name__}: {e}", **overrides}


def _exit_137_note(row: dict) -> dict:
    """Annotate a likely OOM (subprocess killed) so the table reads honestly."""
    if row.get("status") == "no_row" and row.get("exit_code") in (137, -9):
        row["note"] = "OOM-killed (SIGKILL) — exceeded memory budget"
    return row


def write_outputs(rows: list[dict], basename: str = "feasibility_table") -> None:
    import pandas as pd

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [_exit_137_note(r) for r in rows]
    df = pd.DataFrame(rows)
    cols = [c for c in ["label", "technique", "fits", "peak_vram_gb", "peak_ram_gb",
                        "peak_ram_delta_gb", "wallclock_per_step_s",
                        "wallclock_per_epoch_s", "eval_score", "eval_metric",
                        "final_train_loss", "steps", "status", "run_seconds",
                        "note"] if c in df.columns]
    df = df[cols + [c for c in df.columns if c not in cols]]
    df.to_csv(OUT_DIR / f"{basename}.csv", index=False)  # CSV is the source of truth
    try:
        df.to_parquet(OUT_DIR / f"{basename}.parquet", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] parquet write failed ({e}); csv written")
    try:  # markdown is nice-to-have (needs tabulate); never let it crash the batch
        (OUT_DIR / f"{basename}.md").write_text(df[cols].to_markdown(index=False))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] markdown render failed ({e}); csv/parquet written")
    print(f"\n== wrote {len(rows)} rows -> {OUT_DIR}/{basename}.(csv|parquet|md)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 steps/run shake-out")
    ap.add_argument("--ablation", action="store_true", help="also run the lever ablation")
    ap.add_argument("--combos", action="store_true",
                    help="run the method-combination matrix -> combinations.* (instead of techniques)")
    ap.add_argument("--steps", type=int, default=None, help="override steps/run")
    ap.add_argument("--timeout-min", type=int, default=60, help="per-run timeout")
    ap.add_argument("--only", nargs="*", default=None, help="run only these labels")
    ap.add_argument("--task", default=None,
                    help="override task for all runs; writes <basename>_<tasktag>.*")
    ap.add_argument("--eval-samples", type=int, default=None,
                    help="override eval.max_eval_samples (held-out set size)")
    args = ap.parse_args()

    steps = args.steps if args.steps is not None else (3 if args.quick else 50)
    timeout_s = args.timeout_min * 60

    if args.combos:
        plan = list(COMBO_MATRIX)
        basename = "combinations"
    else:
        plan = list(TECHNIQUE_MATRIX)
        if args.ablation:
            for label, delta in ABLATION_CELLS:
                ov = dict(ABLATION_ANCHOR)
                ov.update(delta)
                plan.append((label, ov))
        basename = "feasibility_table"
    if args.only:
        plan = [(l, o) for (l, o) in plan if l in args.only]

    # Second-task support: inject the task override + a name suffix so a second
    # task writes its own table and its own traces (no clobbering task 1).
    name_suffix = ""
    if args.task:
        tag = args.task.split("_")[0]  # e.g. 'task1344'
        name_suffix = f"_{tag}"
        basename = f"{basename}_{tag}"
        plan = [(l, {**o, "task": args.task}) for (l, o) in plan]
    if args.eval_samples is not None:
        plan = [(l, {**o, "eval.max_eval_samples": str(args.eval_samples)}) for (l, o) in plan]

    print(f"== Phase 0.5 matrix ({basename}): {len(plan)} runs, {steps} steps each, "
          f"timeout {args.timeout_min}min/run")
    rows = []
    for label, overrides in plan:
        row = run_one(label, overrides, steps, timeout_s, name_suffix=name_suffix)
        rows.append(row)
        print(f"   -> {label}: fits={row.get('fits')} "
              f"vram={row.get('peak_vram_gb')} ram={row.get('peak_ram_gb')} "
              f"eval={row.get('eval_score')} status={row.get('status')}")
        write_outputs(rows, basename)  # checkpoint after every run so a crash keeps partials
    print("\n== MATRIX COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
