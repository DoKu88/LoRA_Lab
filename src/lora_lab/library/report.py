"""Sprint 6 — aggregate gate results into the library quality table.

Reads ``results/phase1/gate_results.jsonl`` (one row per task, written by the
S4 gate) and emits ``results/phase1/library_quality.{csv,parquet,md}`` — the
gate evidence. Also folds gate outcomes (kind/metric/rank/pass-fail) back into
the manifest so a quarantined adapter is recorded, never silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

GATE_JSONL = "results/phase1/gate_results.jsonl"
OUT_DIR = "results/phase1"

COLUMNS = [
    "task_num", "task_name", "rank", "kind", "metric", "n_eval",
    "base_score", "adapter_score", "margin", "tau", "gate", "wallclock_s",
]


def load_gate_rows(path: str | Path = GATE_JSONL) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render_markdown(rows: list[dict]) -> str:
    ok = [r for r in rows if r.get("gate") in ("pass", "fail")]
    ok.sort(key=lambda r: -r.get("margin", -9))
    headers = ["task_num", "task", "rank", "metric", "base", "adapter",
               "margin", "gate", "s"]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in ok:
        name = r["task_name"]
        name = name[:38] + "…" if len(name) > 39 else name
        out.append("| " + " | ".join([
            r["task_num"], name, str(r.get("rank", "")), r["metric"],
            f"{r['base_score']:.3f}", f"{r['adapter_score']:.3f}",
            f"{r['margin']:+.3f}", r["gate"].upper(), f"{r['wallclock_s']:.0f}",
        ]) + " |")
    errs = [r for r in rows if r.get("gate") == "error"]
    if errs:
        out.append("\n**Errored/quarantined:** " +
                   ", ".join(f"{r['task_num']} ({r.get('reason','?')})" for r in errs))
    return "\n".join(out) + "\n"


def write_quality_table(out_dir: str | Path = OUT_DIR) -> dict:
    import pandas as pd

    rows = load_gate_rows()
    scored = [r for r in rows if r.get("gate") in ("pass", "fail")]
    out_dir = Path(out_dir)
    df = pd.DataFrame([{c: r.get(c) for c in COLUMNS} for r in scored], columns=COLUMNS)
    df = df.sort_values("margin", ascending=False)

    paths = {}
    csv_path = out_dir / "library_quality.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    try:
        pq = out_dir / "library_quality.parquet"
        df.to_parquet(pq, index=False)
        paths["parquet"] = pq
    except Exception as e:  # parquet optional
        print(f"[report] parquet skipped ({type(e).__name__})")
    md = out_dir / "library_quality.md"
    n_pass = sum(1 for r in scored if r["gate"] == "pass")
    header = (f"# Phase 1 — Library Quality Gate (adapter vs. base)\n\n"
              f"Gate-passing: **{n_pass}/{len(scored)}** tasks "
              f"(τ = margin ≥ {scored[0]['tau'] if scored else 0.05}). "
              f"Eval = exact-match (classification) / ROUGE-L (generation) on the "
              f"held-out test split; base vs. adapter on the *same* resident "
              f"Mistral-7B via PEFT `disable_adapter`.\n\n")
    md.write_text(header + render_markdown(rows))
    paths["md"] = md

    summary = {
        "scored": len(scored),
        "pass": n_pass,
        "fail": sum(1 for r in scored if r["gate"] == "fail"),
        "error": sum(1 for r in rows if r.get("gate") == "error"),
        "pass_task_nums": [r["task_num"] for r in scored if r["gate"] == "pass"],
        "mean_margin": round(sum(r["margin"] for r in scored) / len(scored), 4) if scored else 0,
    }
    return {"paths": paths, "summary": summary}
