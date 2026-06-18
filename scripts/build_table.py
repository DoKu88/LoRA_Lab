#!/usr/bin/env python
"""Assemble the Phase 0 deliverable: comparison table/dataset + memory plots.

  * results/comparison.csv / .parquet / .md  (one row per method x model x task)
  * results/plots/gpu_mem_vs_iter_{model}-{task}.png  (overlay of the 3 methods)

    conda run -n lora_lab python scripts/build_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.eval.plot import plot_memory_traces  # noqa: E402
from lora_lab.eval.table import collect_rows, write_table  # noqa: E402


def main() -> int:
    rows = collect_rows()
    if not rows:
        print("[table] no run summaries found under results/runs/ — nothing to do")
    else:
        paths = write_table(rows)
        print(f"[table] {len(rows)} rows written:")
        for k, p in paths.items():
            print(f"   {k}: {p}")
        print("\n" + (Path("results/comparison.md").read_text()))

    plots = plot_memory_traces()
    if not plots:
        print("[plot] no memory traces found under results/mem_trace/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
