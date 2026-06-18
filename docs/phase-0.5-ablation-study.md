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

Anchor = paged 8-bit AdamW + gradient checkpointing (27.64 GB). Each cell
flips exactly one lever. Source: `results/phase05/feasibility_table.csv`
(`abl_*` rows); chart: `results/phase05/plots/lever_contribution.png`.

| Cell | Lever changed vs anchor | Fits | Peak VRAM | Δ VRAM | s/micro-batch | Interpretation |
|---|---|---|---|---|---|---|
| `abl_anchor` | — (full stack) | ✅ | 27.64 GB | 0 (ref) | 0.282 | reference |
| `abl_no_8bit` | 8-bit Adam → fp32 AdamW | ❌ **OOM** | — | **load-bearing** | — | without 8-bit Adam the run **does not fit** |
| `abl_no_gradckpt` | gradient checkpointing off | ✅ | 29.92 GB | **+2.28 GB** | 0.246 | costs 2.28 GB VRAM; saves ~13% step time |

---

## What each lever buys (measured)

- **8-bit Adam — LOAD-BEARING.** Removing it (fp32 AdamW) **OOMs** during the
  first optimizer step (allocating `exp_avg_sq`): fp32 Adam state (~56 GB for m+v)
  plus bf16 weights+grads cannot coexist in 32 GB. This is the single most
  important lever — it's the difference between feasible and infeasible for the
  8-bit-Adam route. (The VRAM-direct optimizers — LOMO/GaLore/BAdam — achieve the
  same end differently, so they don't depend on this lever.)
- **Gradient checkpointing — a real but modest VRAM lever with a speed cost.**
  Turning it off *raises* peak VRAM by **2.28 GB** (29.92 vs 27.64) and *lowers*
  step time by ~13% (0.246 vs 0.282 s/micro-batch) — the classic recompute-for-
  memory trade. With it on we stay comfortably under 32 GB; off, we're at 29.9 GB
  with little margin.

## Headroom recommendation for the hypernetwork

The anchor (paged 8-bit) leaves only **~4.4 GB** free — too tight to stack a
hypernetwork. The ablation says you can't recover much from the levers on this
route (8-bit is mandatory; dropping grad-checkpointing *costs* VRAM). So **the
headroom must come from the choice of *technique*, not lever-tuning**: **LOMO
(14.6 GB → ~17 GB free)** or **BAdam (17.6 GB → ~14 GB free)** are the routes
that leave room for the Phase 2 hypernetwork, while keeping gradient
checkpointing on (cheap insurance) and — for LOMO/BAdam — not needing 8-bit Adam
at all. Net: **train the 7B with LOMO + gradient checkpointing; spend the freed
~17 GB on the hypernetwork.**
