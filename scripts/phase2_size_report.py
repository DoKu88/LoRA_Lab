#!/usr/bin/env python
"""Phase 2 Sprint 2 — hypernetwork size report (table T3 inputs).

Builds the generator for the real Mistral-7B target set (q/k/v over 32 layers,
GQA: k/v out=1024) under each output parameterization and reports the learned
param count — the size half of table T3 (the VRAM half is measured in S4). Logs
the chosen parameterization + param counts to W&B (best-effort) and writes
results/phase2/hypernet_sizes.{csv,md}. No model download needed (specs are the
known Mistral attention shapes).

    python scripts/phase2_size_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.hypernet.config import HyperConfig  # noqa: E402
from lora_lab.hypernet.heads import estimate_params  # noqa: E402
from lora_lab.hypernet.logging import build_run_logger  # noqa: E402
from lora_lab.hypernet.model import HyperLoRAGenerator  # noqa: E402

# Mistral-7B-Instruct-v0.2 attention shapes (hidden 4096, GQA kv 1024), 32 layers.
N_LAYERS, D_MODEL, D_KV = 32, 4096, 1024
D_TASK = 384  # all-MiniLM-L6-v2 embedding dim (the default encoder)
RANK = 16


def mistral_target_specs() -> dict[str, tuple[int, int]]:
    specs: dict[str, tuple[int, int]] = {}
    for layer in range(N_LAYERS):
        p = f"model.layers.{layer}.self_attn"
        specs[f"{p}.q_proj"] = (D_MODEL, D_MODEL)
        specs[f"{p}.k_proj"] = (D_MODEL, D_KV)
        specs[f"{p}.v_proj"] = (D_MODEL, D_KV)
    return specs


TRUNK_HIDDEN, D_LAYER, D_MODULE = 256, 16, 8  # match configs/phase2/sft-mistral.yaml
D_COND = TRUNK_HIDDEN + D_LAYER + D_MODULE     # the real conditioning vector the heads see


def size_rows() -> list[dict]:
    """Param count per parameterization at the generator's *real* head input dim.

    estimate_params builds one head at a time at ``D_COND`` (= the conditioning
    vector the real generator feeds its heads — NOT the raw task embedding), so it
    matches the built generator's head params without instantiating a multi-billion-
    param 'full' module. The small shared trunk/embeddings (~0.2 M) are extra on top.
    """
    specs = mistral_target_specs()
    rows = []
    for param in ("vera", "lowrank", "full"):
        n = estimate_params(param, specs, D_COND, RANK)
        rows.append({"parameterization": param, "hypernet_params_M": round(n / 1e6, 2),
                     "n_targets": len(specs), "rank": RANK, "d_cond": D_COND})
    return rows


def main() -> int:
    rows = size_rows()
    # Validate the committed default (VeRA) actually builds + forwards on the real spec.
    gen = HyperLoRAGenerator(mistral_target_specs(), task_dim=D_TASK, rank=RANK,
                             parameterization="vera", trunk_hidden=TRUNK_HIDDEN,
                             layer_dim=D_LAYER, module_dim=D_MODULE)
    adapter = gen(torch.randn(D_TASK))
    assert len(adapter) == N_LAYERS * 3
    assert all(torch.isfinite(a).all() and torch.isfinite(b).all() for a, b in adapter.values())
    measured_vera_M = round(gen.num_params() / 1e6, 2)

    out = Path("results/phase2")
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "hypernet_sizes.csv", index=False)
    md = ["# T3 (size half) — hypernet params per output parameterization\n",
          f"Mistral-7B target set: q/k/v x {N_LAYERS} layers = {N_LAYERS*3} targets, "
          f"rank {RANK}, d_task {D_TASK}. **Committed default: VeRA** "
          f"(measured generator: {measured_vera_M} M params).\n",
          "| parameterization | hypernet params (M) |", "|---|--:|"]
    for r in rows:
        md.append(f"| {r['parameterization']} | {r['hypernet_params_M']} |")
    (out / "hypernet_sizes.md").write_text("\n".join(md) + "\n")

    # Log T3 inputs to W&B (online for the S2 verification handoff) via a HyperConfig run.
    cfg = HyperConfig(parameterization="vera", objective="reconstruction", device="cpu",
                      load_in_4bit=False, run_name="s2-size-report",
                      output_root="results/phase2/runs", wandb_mode="online")
    logger = build_run_logger(cfg, stage="S2-architecture")
    logger.set_summary(parameterization="vera", measured_vera_params_M=measured_vera_M,
                       **{f"params_M_{r['parameterization']}": r["hypernet_params_M"] for r in rows},
                       n_targets=N_LAYERS * 3, rank=RANK, d_task=D_TASK)
    logger.finish()

    print("[size-report] T3 (size half):")
    for r in rows:
        print(f"  {r['parameterization']:8} -> {r['hypernet_params_M']:8} M params")
    print(f"[size-report] VeRA generator built + forwarded on real Mistral spec "
          f"({measured_vera_M} M params, {N_LAYERS*3} targets); wrote {out}/hypernet_sizes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
