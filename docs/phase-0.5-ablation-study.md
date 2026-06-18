# Phase 0.5 — Ablation Study: Per-Optimization Contribution & Hypernetwork Headroom

*Objective B of the Phase 0.5 spike (see [`phase-0.5-sprint-plan.md`](./phase-0.5-sprint-plan.md)).
Each stackable memory optimization is toggled **one at a time** against a fixed
working anchor, so we know its isolated contribution to VRAM / RAM / speed —
even when an earlier optimization already made the run fit. The point is to
budget the **VRAM headroom** the hypernetwork (Phase 2+) will need on top of the
7B full fine-tune.*

**Anchor:** the working baseline — bf16 Mistral-7B on the GPU + paged 8-bit
AdamW + gradient checkpointing (the Sprint 2 feasibility recipe). Each cell
changes **exactly one lever** vs. this anchor. Fixed protocol: seq 512, batch 1
× grad-accum 8, 50 steps, seed 42, `task843`.

Raw data: `results/phase05/lever_ablation.{csv,parquet}` /
`results/phase05/feasibility_table.csv` (rows prefixed `abl_`).
Chart: `results/phase05/plots/lever_contribution.png`.

---

## Ablation table

*(populated by `scripts/run_phase05_matrix.py --ablation` then this writeup;
numbers below are filled in Sprint 7.)*

| Cell | Lever changed vs anchor | Fits | Peak VRAM | Δ VRAM vs anchor | Peak RAM | s/step | Interpretation |
|---|---|---|---|---|---|---|---|
| `abl_anchor` | — (full stack) | | | 0 (ref) | | | reference |
| `abl_no_8bit` | 8-bit Adam → fp32 | | | | | | cost of dropping 8-bit Adam |
| `abl_no_gradckpt` | gradient checkpointing off | | | | | | cost of dropping grad checkpointing |

---

## What each lever buys (to be written from the numbers)

- **8-bit Adam (vs fp32 AdamW).** Optimizer state 4 bytes → 1 byte/param. Expected
  to be load-bearing on the GPU path: fp32 Adam state (~56 GB) + bf16 grads can't
  coexist with the model in 32 GB, so `abl_no_8bit` is expected `fits=no`.
- **Gradient checkpointing.** Trades recompute for activation memory; expected a
  large VRAM cut at a modest step-time cost.

## Headroom recommendation for the hypernetwork

*(Sprint 7: from the anchor's free-VRAM headroom (32 − peak) and the lever
deltas, state which lever stack leaves the most room for the Phase 2
hypernetwork while keeping step time acceptable.)*
