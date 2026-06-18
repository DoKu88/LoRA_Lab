"""Overlay the per-method GPU-memory-vs-iteration traces (Sprint 5).

Reads the raw traces persisted by the trainer (results/mem_trace/{method}-
{model}-{task}.csv) and renders one figure per (model, task) overlaying the
three methods on shared axes — x = training step, y = GPU memory (GB) — so the
memory profiles are directly comparable.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

_METHODS = ("qlora", "lora", "full_ft")
_COLORS = {"qlora": "tab:green", "lora": "tab:blue", "full_ft": "tab:red"}
# trace filename: {method}-{model}-{task}.csv  (method has no '-', model/task may)
_NAME_RE = re.compile(r"^(qlora|lora|full_ft)-(.+?)-(task\d+.*)$")


def _read_trace(path: Path) -> tuple[list[int], list[float]]:
    steps, mem = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            mem.append(float(row["gpu_mem_gb"]))
    return steps, mem


def _parse_name(stem: str) -> tuple[str, str, str] | None:
    m = _NAME_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)  # method, model, task


def plot_memory_traces(
    trace_dir: str | Path = "results/mem_trace",
    out_dir: str | Path = "results/plots",
) -> list[Path]:
    """Render one overlay PNG per (model, task). Returns written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace_dir = Path(trace_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # group traces by (model, task)
    groups: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for csv_path in sorted(trace_dir.glob("*.csv")):
        parsed = _parse_name(csv_path.stem)
        if not parsed:
            continue
        method, model, task = parsed
        groups[(model, task)][method] = csv_path

    written: list[Path] = []
    for (model, task), method_paths in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        peak = 0.0
        for method in _METHODS:
            if method not in method_paths:
                continue
            steps, mem = _read_trace(method_paths[method])
            if not steps:
                continue
            peak = max(peak, max(mem))
            ax.plot(steps, mem, label=method, color=_COLORS[method], marker="o", ms=3)
        ax.set_xlabel("training iteration (optimizer step)")
        ax.set_ylabel("GPU memory (GB)")
        ax.set_title(f"GPU memory vs. iteration — {model} / {task}")
        ax.legend(title="method")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        out_path = out_dir / f"gpu_mem_vs_iter_{model}-{task}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        written.append(out_path)
        print(f"[plot] {out_path}  (peak {peak:.2f} GB)")
    return written
