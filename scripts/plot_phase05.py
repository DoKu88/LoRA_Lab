#!/usr/bin/env python
"""Phase 0.5 plots (Sprint 7).

Renders, from the matrix outputs in results/phase05/:
  * gpu_mem_vs_iter.png  — peak VRAM vs step, one curve per technique
  * ram_vs_iter.png      — peak RAM vs step, one curve per technique
  * speed_vs_memory.png  — wall-clock/step vs peak VRAM scatter (the Pareto view)
  * lever_contribution.png — freed VRAM per ablation lever (if ablation present)

    conda run -n lora_lab python scripts/plot_phase05.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P05 = ROOT / "results" / "phase05"
TRACE = P05 / "mem_trace"
PLOTS = P05 / "plots"


def _read_traces(suffix: str, value_col: str):
    import pandas as pd

    series = {}
    for f in sorted(TRACE.glob(f"*{suffix}")):
        name = f.name[: -len(suffix)]
        try:
            df = pd.read_csv(f)
            if value_col in df.columns and "step" in df.columns and len(df):
                series[name] = df
        except Exception:  # noqa: BLE001
            continue
    return series


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    PLOTS.mkdir(parents=True, exist_ok=True)

    # --- memory-vs-iteration overlays ----------------------------------
    for suffix, col, ylabel, fname in [
        (".csv", "gpu_mem_gb", "GPU memory (GiB)", "gpu_mem_vs_iter.png"),
        (".ram.csv", "ram_gb", "Host RAM (GiB)", "ram_vs_iter.png"),
    ]:
        # avoid matching .ram.csv when we want plain .csv
        series = {n: d for n, d in _read_traces(suffix, col).items()
                  if not (suffix == ".csv" and n.endswith(".ram"))}
        if not series:
            continue
        plt.figure(figsize=(9, 5))
        for name, df in series.items():
            plt.plot(df["step"], df[col], marker=".", ms=3, label=name)
        plt.xlabel("training step")
        plt.ylabel(ylabel)
        plt.title(f"Phase 0.5 — {ylabel} vs iteration (Mistral-7B full FT)")
        plt.legend(fontsize=7, ncol=2)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS / fname, dpi=120)
        plt.close()
        print(f"  wrote {PLOTS / fname}")

    # --- speed vs peak memory scatter (Pareto) -------------------------
    table = P05 / "feasibility_table.csv"
    if table.exists():
        df = pd.read_csv(table)
        d = df[df.get("fits") == True] if "fits" in df.columns else df  # noqa: E712
        if "peak_vram_gb" in d.columns and "wallclock_per_step_s" in d.columns:
            d = d.dropna(subset=["peak_vram_gb", "wallclock_per_step_s"])
            if len(d):
                plt.figure(figsize=(8, 6))
                plt.scatter(d["peak_vram_gb"], d["wallclock_per_step_s"])
                for _, r in d.iterrows():
                    plt.annotate(str(r.get("technique", r.get("label", ""))),
                                 (r["peak_vram_gb"], r["wallclock_per_step_s"]),
                                 fontsize=8, xytext=(4, 4), textcoords="offset points")
                plt.axvline(32, ls="--", c="r", alpha=0.5, label="32 GB VRAM limit")
                plt.xlabel("peak VRAM (GiB)")
                plt.ylabel("wall-clock / step (s)")
                plt.title("Phase 0.5 — speed vs peak memory (lower-left = better)")
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(PLOTS / "speed_vs_memory.png", dpi=120)
                plt.close()
                print(f"  wrote {PLOTS / 'speed_vs_memory.png'}")

        # --- lever contribution (from ablation rows) -------------------
        abl = df[df["label"].astype(str).str.startswith("abl_")] if "label" in df.columns else df.iloc[0:0]
        if len(abl) >= 2 and "peak_vram_gb" in abl.columns:
            anchor = abl[abl["label"] == "abl_anchor"]
            if len(anchor):
                base_vram = float(anchor.iloc[0]["peak_vram_gb"])
                deltas = []
                for _, r in abl.iterrows():
                    if r["label"] == "abl_anchor":
                        continue
                    if pd.notna(r.get("peak_vram_gb")):
                        deltas.append((r["label"].replace("abl_no_", "−"),
                                       float(r["peak_vram_gb"]) - base_vram))
                if deltas:
                    plt.figure(figsize=(7, 4))
                    names = [n for n, _ in deltas]
                    vals = [v for _, v in deltas]
                    plt.bar(names, vals)
                    plt.ylabel("Δ peak VRAM vs anchor (GiB)")
                    plt.title("Lever ablation — VRAM cost of removing each lever")
                    plt.grid(alpha=0.3, axis="y")
                    plt.tight_layout()
                    plt.savefig(PLOTS / "lever_contribution.png", dpi=120)
                    plt.close()
                    print(f"  wrote {PLOTS / 'lever_contribution.png'}")
    print("== plots done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
