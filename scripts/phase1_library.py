#!/usr/bin/env python
"""Phase 1 — LoRA library pipeline (Sprints 1–4).

Subcommands (each resumable; per-task isolation — one bad task never aborts the
batch):

    # S1: build the versioned manifest from the Hub (1172 tasks, pilot flagged)
    conda run -n lora_lab python scripts/phase1_library.py manifest

    # S3: fill each task's NL description (the conditioning input)
    conda run -n lora_lab python scripts/phase1_library.py descriptions --pilot

    # S4: the quality gate — adapter-vs-base eval on the resident 7B
    conda run -n lora_lab python scripts/phase1_library.py gate --pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.library import manifest as M  # noqa: E402
from lora_lab.library.descriptions import description_for_task  # noqa: E402

GATE_JSONL = Path("results/phase1/gate_results.jsonl")


def cmd_manifest(args) -> int:
    entries = M.build_manifest()
    M.validate(entries)
    path = M.save_manifest(entries)
    rep = M.coverage_report(entries)
    print(f"[manifest] wrote {path}  version={M.manifest_hash(entries)}")
    print(json.dumps(rep, indent=2))
    return 0


def cmd_descriptions(args) -> int:
    entries = M.load_manifest()
    todo = [e for e in entries if e.status == "ok" and (not args.pilot or e.pilot)]
    print(f"[descriptions] filling {len(todo)} tasks "
          f"({'pilot' if args.pilot else 'full'})")
    for e in todo:
        if e.description and not args.force:
            continue
        try:
            desc = description_for_task(e.dataset_repo)
            if not desc:
                e.status, e.reason = "quarantined", "no_description"
                print(f"  {e.task_num}: NO DEFINITION -> quarantined")
                continue
            e.description = desc
            e.description_source = "sni_definition"
            e.description_hash = M._sha(desc)
            print(f"  {e.task_num}: {desc[:70]}...")
        except Exception as exc:  # per-task isolation
            e.status, e.reason = "quarantined", f"description_error:{type(exc).__name__}"
            print(f"  {e.task_num}: ERROR {exc!r} -> quarantined")
    M.save_manifest(entries)
    print(f"[descriptions] saved; with_description="
          f"{M.coverage_report(entries)['with_description']}")
    return 0


def cmd_gate(args) -> int:
    from lora_lab.library.gate import GateRunner, DEFAULT_TAU

    entries = M.load_manifest()
    done = set()
    if GATE_JSONL.exists() and not args.restart:
        for line in GATE_JSONL.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["task_num"])
    todo = [e for e in entries
            if e.status == "ok" and e.description
            and (not args.pilot or e.pilot) and e.task_num not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gate] {len(todo)} tasks to eval (skipping {len(done)} done), "
          f"tau={DEFAULT_TAU}")
    if not todo:
        return 0

    GATE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    runner = GateRunner(max_eval_samples=args.eval_samples)
    mode = "a" if (GATE_JSONL.exists() and not args.restart) else "w"
    with GATE_JSONL.open(mode) as fh:
        for i, e in enumerate(todo, 1):
            try:
                row = runner.run_task(e, tau=DEFAULT_TAU)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(f"  [{i}/{len(todo)}] {e.task_num} {row['metric']}: "
                      f"adapter={row['adapter_score']} base={row['base_score']} "
                      f"margin={row['margin']:+.3f} -> {row['gate'].upper()} "
                      f"({row['wallclock_s']}s)")
            except Exception as exc:  # per-task isolation
                err = {"task_num": e.task_num, "task_name": e.task_name,
                       "gate": "error", "reason": f"{type(exc).__name__}: {exc}"}
                fh.write(json.dumps(err) + "\n")
                fh.flush()
                print(f"  [{i}/{len(todo)}] {e.task_num} ERROR: {exc!r}")
                traceback.print_exc()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest").set_defaults(func=cmd_manifest)
    d = sub.add_parser("descriptions")
    d.add_argument("--pilot", action="store_true")
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_descriptions)
    g = sub.add_parser("gate")
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--eval-samples", type=int, default=150)
    g.add_argument("--restart", action="store_true")
    g.set_defaults(func=cmd_gate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
