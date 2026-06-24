#!/usr/bin/env python
"""Phase 1 — finalize the library (Sprints 5–6).

    # S6: aggregate the gate results into results/phase1/library_quality.{csv,md}
    conda run -n lora_lab python scripts/phase1_finalize.py report

    # S5: lock the frozen train/val/held-out split over gate-passing tasks
    conda run -n lora_lab python scripts/phase1_finalize.py split

    # S5: train our own rank-16 oracle LoRAs for the held-out tasks
    conda run -n lora_lab python scripts/phase1_finalize.py oracles

    # S6: stamp split roles + version hash into the manifest
    conda run -n lora_lab python scripts/phase1_finalize.py finalize
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.library import manifest as M  # noqa: E402
from lora_lab.library import split as SP  # noqa: E402
from lora_lab.library.report import write_quality_table, load_gate_rows  # noqa: E402

ORACLE_JSON = Path("results/phase1/oracles.json")


def cmd_report(args) -> int:
    res = write_quality_table()
    print(f"[report] {json.dumps(res['summary'], indent=2)}")
    for k, p in res["paths"].items():
        print(f"  {k}: {p}")
    return 0


def cmd_split(args) -> int:
    rows = load_gate_rows()
    passing = [r["task_num"] for r in rows if r.get("gate") == "pass"]
    names = {r["task_num"]: r["task_name"] for r in rows if "task_name" in r}
    sp = SP.make_split(passing, n_val=args.n_val, n_heldout=args.n_heldout, seed=42)
    path = SP.save_split(sp, names)
    print(f"[split] wrote {path}  lock_hash={sp.lock_hash()}")
    print(f"  train={len(sp.train)} val={len(sp.val)} held_out={len(sp.held_out)} "
          f"removed={len(sp.removed)}")
    print(f"  held_out: {sp.held_out}")
    sp.assert_valid()
    print("[split] leakage guard OK (held-out disjoint from train)")
    return 0


def cmd_oracles(args) -> int:
    from lora_lab.library.oracle import ensure_tasks_registered, train_oracle

    sp = SP.load_split()
    rows = {r["task_num"]: r for r in load_gate_rows() if "task_name" in r}
    entries = {e.task_num: e for e in M.load_manifest()}

    to_register, targets = [], []
    for num in sp.held_out:
        g = rows.get(num, {})
        e = entries.get(num)
        name = g.get("task_name") or (e.task_name if e else num)
        metric = g.get("metric", "exact_match")
        to_register.append({
            "name": name, "hf_repo": e.dataset_repo if e else "",
            "kind": g.get("kind", "classification"), "metric": metric,
            "description": e.description if e else "",
        })
        targets.append((num, name, metric, g.get("base_score")))
    ensure_tasks_registered(to_register)
    print(f"[oracles] training {len(targets)} held-out oracle LoRAs")

    results = []
    for num, name, metric, base in targets:
        print(f"[oracles] -> {num} ({name}) metric={metric}")
        try:
            r = train_oracle(name, metric, max_train_samples=args.train_samples,
                             max_steps=args.max_steps)
            r["task_num"] = num
            r["base_score"] = base
            r["margin_vs_base"] = round(r["oracle_score"] - (base or 0), 4)
            r["gate"] = "pass" if r["margin_vs_base"] >= SP_DEFAULT_TAU else "fail"
            results.append(r)
            print(f"  {num}: oracle={r['oracle_score']} base={base} "
                  f"margin={r['margin_vs_base']:+.3f} -> {r['gate'].upper()}")
        except Exception as exc:
            print(f"  {num}: ERROR {exc!r}")
            results.append({"task_num": num, "task_name": name, "gate": "error",
                            "reason": f"{type(exc).__name__}: {exc}"})
    ORACLE_JSON.write_text(json.dumps(results, indent=2))
    print(f"[oracles] wrote {ORACLE_JSON}")
    return 0


def cmd_finalize(args) -> int:
    entries = M.load_manifest()
    sp = SP.load_split()
    role = {}
    for t in sp.train: role[t] = "train"
    for t in sp.val: role[t] = "val"
    for t in sp.held_out: role[t] = "held-out"
    # fold gate outcomes into the manifest (kind/metric/rank + quarantine fails)
    gate = {r["task_num"]: r for r in load_gate_rows() if r.get("gate") in ("pass", "fail")}
    for e in entries:
        if e.task_num in role:
            e.split_role = role[e.task_num]
        g = gate.get(e.task_num)
        if g:
            e.kind, e.metric, e.rank = g["kind"], g["metric"], g.get("rank", e.rank)
            if g["gate"] == "fail":
                e.status, e.reason = "quarantined", f"below_tau:margin={g['margin']}"
    M.validate(entries)
    path = M.save_manifest(entries)
    print(f"[finalize] manifest version={M.manifest_hash(entries)} -> {path}")
    print(json.dumps(M.coverage_report(entries), indent=2))
    return 0


SP_DEFAULT_TAU = 0.05


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report").set_defaults(func=cmd_report)
    s = sub.add_parser("split")
    s.add_argument("--n-val", type=int, default=1)
    s.add_argument("--n-heldout", type=int, default=3)
    s.set_defaults(func=cmd_split)
    o = sub.add_parser("oracles")
    o.add_argument("--train-samples", type=int, default=500)
    o.add_argument("--max-steps", type=int, default=250)
    o.set_defaults(func=cmd_oracles)
    sub.add_parser("finalize").set_defaults(func=cmd_finalize)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
