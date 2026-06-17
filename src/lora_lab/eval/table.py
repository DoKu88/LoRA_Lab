"""Build the Phase 0 comparison table/dataset from run summaries."""

from __future__ import annotations

import json
from pathlib import Path

# The machine-readable results schema (one row per method x model x task run).
COLUMNS = [
    "method",
    "base_model",
    "task",
    "trainable_params",
    "pct_params",
    "peak_vram_gb",
    "wallclock_per_epoch_s",
    "final_train_loss",
    "eval_metric",
    "eval_metric_name",
    "checkpoint_size_mb",
]

# method ordering for stable, readable tables
_METHOD_ORDER = {"qlora": 0, "lora": 1, "full_ft": 2}


def _model_slug(model: str) -> str:
    return model.split("/")[-1]


def collect_rows(runs_dir: str | Path = "results/runs") -> list[dict]:
    """Read every results/runs/*/summary.json into a list of schema rows."""
    runs_dir = Path(runs_dir)
    rows: list[dict] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        if s.get("dry_run"):
            continue
        rows.append({c: s.get(c) for c in COLUMNS})
    rows.sort(
        key=lambda r: (
            _model_slug(r.get("base_model") or ""),
            r.get("task") or "",
            _METHOD_ORDER.get(r.get("method"), 9),
        )
    )
    return rows


def render_markdown(rows: list[dict]) -> str:
    headers = [
        "method", "base_model", "task", "trainable_params", "pct_params",
        "peak_vram_gb", "wallclock/epoch (s)", "final_loss", "eval", "metric",
        "ckpt MB",
    ]
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join([
            str(r.get("method", "")),
            _model_slug(r.get("base_model") or ""),
            str(r.get("task", "")),
            f"{(r.get('trainable_params') or 0):,}",
            f"{r.get('pct_params', '')}",
            _fmt(r.get("peak_vram_gb")),
            _fmt(r.get("wallclock_per_epoch_s")),
            _fmt(r.get("final_train_loss")),
            _fmt(r.get("eval_metric")),
            str(r.get("eval_metric_name") or ""),
            _fmt(r.get("checkpoint_size_mb")),
        ]) + " |")
    return "\n".join(out) + "\n"


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_table(rows: list[dict], out_dir: str | Path = "results") -> dict[str, Path]:
    """Write comparison.csv (+ .parquet if pyarrow) + comparison.md."""
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLUMNS)

    paths: dict[str, Path] = {}
    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    try:
        pq_path = out_dir / "comparison.parquet"
        df.to_parquet(pq_path, index=False)
        paths["parquet"] = pq_path
    except Exception as e:  # noqa: BLE001 - parquet optional
        print(f"[table] parquet skipped ({type(e).__name__}: {e})")

    md_path = out_dir / "comparison.md"
    md_path.write_text("# Phase 0 — Three-Way Fine-Tuning Comparison\n\n" + render_markdown(rows))
    paths["md"] = md_path
    return paths
