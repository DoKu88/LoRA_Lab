# Phase 0.5 — Full-FT Memory Optimizations: VRAM / RAM / Time Trade-offs

*Reference table for the Phase 0.5 feasibility spike — "can we full-finetune Mistral-7B on this box?" The **execution plan that fills in this table** is [`./phase-0.5-sprint-plan.md`](./phase-0.5-sprint-plan.md). Pairs with [`../notes.md`](../notes.md) §C2 (Phase 0.5) and §B (practical tips). Section refs (§2.x) point at the lit-review entries in `../summaries.md`.*

---

## The wall — and why 96 GB RAM changes it

bf16 + standard Adam on Mistral-7B ≈ **14 GB weights + 14 GB grads + ~56 GB optimizer states ≈ 84 GB** — far over the **32 GB VRAM** budget. The box has **32 GB VRAM + 96 GB system RAM**.

The old blocker was system RAM: CPU offload (the usual escape hatch) needs tens of GB of RAM, which we used to lack. **With 96 GB RAM that escape hatch is now open** — ZeRO-Offload / FSDP CPU-offload can hold the offloaded optimizer + gradient state (~70 GB with fp32 Adam, ~28 GB with 8-bit) in the CPU pool while the 32 GB VRAM carries weights + activations. So there are now **two routes** to 7B full FT:

1. **Offload** (newly viable) — push optimizer/grads to the 96 GB RAM pool; simplest path to a *working* full FT, but PCIe-bound (slower steps).
2. **VRAM-direct** (GaLore/LOMO/BAdam/MeZO) — shrink the optimizer/gradient footprint so everything stays on-GPU; faster, no offload tax.

The job below is to either keep state on-GPU (route 2) **or** spill it cleanly to the now-ample RAM (route 1). NVMe offload (ZeRO-Infinity) is no longer needed.

---

## Techniques vs. VRAM / RAM / training-time

| Technique | VRAM effect | System-RAM effect | Training-time effect | Notes |
|---|---|---|---|---|
| **GaLore** (§2.9) | **Large direct cut** — low-rank-projects optimizer state; the ~56 GB optimizer pool collapses to a small fraction. Full-param FT claimed in ~24 GB class. | Neutral (stays on GPU) | Moderate slowdown — periodic SVD for the projection subspace adds overhead | *VRAM-direct; most promising.* Keeps real full-param updates. |
| **Q-GaLore** (§2.11) | **Largest direct cut** of the GaLore family — adds quantization on top of low-rank projection | Neutral | Moderate (SVD + quant overhead) | Best single bet to actually fit 7B full-FT in 32 GB |
| **LOMO / AdaLOMO** (§2.12) | **Large direct cut** — fuses gradient compute with the update, so full gradients *and* optimizer state are never materialized (SGD-like footprint: ~weights + activations only) | Neutral | Low-to-moderate; AdaLOMO adds adaptive state back, slightly higher | *VRAM-direct; designed for exactly this.* LOMO is closest to "free" memory-wise |
| **BAdam** (§2.13) | **Large direct cut** — only one transformer block holds grads/optimizer state at a time; cycles through all blocks | Neutral | **Higher wall-clock** — full-param coverage requires many block-cycles | *VRAM-direct.* Trades time for memory cleanly |
| **MeZO** (§2.14) | **Largest cut** — forward-only (zeroth-order), no backprop ⇒ inference-level memory (~weights + activations, no grads/optimizer) | Neutral | **Severe slowdown** — noisy, needs many forward passes to converge | *VRAM-direct; last resort.* Fits easily but slow/noisy |
| **8-bit / paged AdamW** (§2.8) | **Moderate cut** — optimizer state from fp32→8-bit (~56 GB → ~14 GB); paging spills to CPU on demand | Slight (paged states use pinned RAM) | Low overhead | Stackable lever, not a standalone fix for 7B |
| **Gradient checkpointing** (§2.7) | **Moderate cut** — recomputes activations instead of storing them; targets the *activation* pool | Neutral | ~20–30% slower steps | Cheapest big win; stack with everything |
| **Activation offload** | Moderate cut — pushes activations to CPU | Raises RAM use (affordable against 96 GB) | Slower (PCIe traffic) | Stackable; now cheap given the RAM headroom |
| **Drop fp32 master copy** | Small-moderate cut — removes the fp32 weight shadow | Neutral | Negligible | Cheap stackable lever; minor stability risk |
| **ZeRO-Offload / FSDP CPU-offload** (§2.6 / §2.10) | **Large cut** — moves optimizer/grads to CPU; VRAM holds only weights + activations | Holds ~70 GB (fp32 Adam) / ~28 GB (8-bit) — **fits in 96 GB** | Slower (PCIe-bound) | ✅ **Now viable thanks to 96 GB RAM.** Simplest route to a working 7B full FT; pair with 8-bit Adam for margin |
| **ZeRO-Infinity (NVMe offload)** (§2.6) | **Largest cut** — streams states to SSD | Low RAM (uses disk) | **Large slowdown; SSD-dependent** | No longer needed — RAM now holds the state; keep only as a fallback |

---

## How to read this for the spike

- **Two viable families, different trade-offs.** *Offload* (ZeRO-Offload/FSDP CPU-offload) is now unblocked by the 96 GB RAM and is the **simplest path to a working 7B full FT** — but PCIe-bound, so slower per step. *VRAM-direct* (GaLore/Q-GaLore, LOMO/AdaLOMO, BAdam, MeZO) keeps the optimizer on-GPU and should be **faster**, at the cost of approximations to the update. The spike's real job is to measure the **speed gap**, not to prove feasibility.
- **Stackable levers** (8-bit Adam, checkpointing, drop-fp32) multiply the headroom of whichever primary technique you pick — and 8-bit Adam in particular shrinks the offloaded state enough to leave comfortable RAM margin.
- **Suggested order to benchmark:** establish the baseline with **ZeRO-Offload + 8-bit Adam** first (most likely to *just work*) → then chase speed with **Q-GaLore / GaLore** and **LOMO** (on-GPU, no offload tax) → **BAdam** if you'll trade time → **MeZO** as last resort → **NVMe offload** only if RAM unexpectedly pinches.

---

## Results to fill in (per technique)

> Populated by **Sprint 6** of [`./phase-0.5-sprint-plan.md`](./phase-0.5-sprint-plan.md); the raw rows + plots land in `results/phase05/`.

Measured on Mistral-7B-Instruct-v0.2, seq 512, batch 1 × grad-accum 8, 50 steps,
seed 42, grad-checkpointing on, `expandable_segments:True`. Full writeup:
[`./phase-0.5-findings.md`](./phase-0.5-findings.md). Speed is **s/micro-batch**
(the apples-to-apples metric — LOMO/AdaLOMO update per micro-batch, others
accumulate 8). **Quality = held-out exact-match** on task843 / task1344.

| Technique | Config / flags | Fits (≤32 GB VRAM / ≤96 GB RAM)? | Measured peak VRAM | Peak system RAM | Wall-clock / micro-batch | Quality (EM: t843 / t1344) | Notes |
|---|---|---|---|---|---|---|---|
| **Q-GaLore** | GaLoreAdamW8bit, rank 128 | ✅ / ✅ | 28.59 GB | 1.9 GB | 0.75 s | **0.88 / 0.83** | best quality; ~28 GB so little headroom |
| **GaLore** | rank 128, gap 200 | ✅ / ✅ | 30.41 GB | 1.9 GB | 0.75 s | 0.87 / 0.84 | tightest VRAM + slowest (SVD/projection) |
| **BAdam** | BlockOptimizer, switch 5 | ✅ / ✅ | 17.60 GB | 1.9 GB | **0.13 s** | 0.86 / 0.75 | ★ quality *and* headroom; +8-bit → 15.7 GB |
| paged 8-bit AdamW (baseline) | bf16 + PagedAdamW8bit + grad-ckpt | ✅ / ✅ | 27.64 GB | 1.9 GB | 0.28 s | 0.61 / 0.53 | simplest; mid quality; 8-bit Adam load-bearing |
| AdaLOMO | fused, adaptive | ✅ / ✅ | 15.10 GB | 1.8 GB | 0.60 s | 0.45 / 0.43 | LR-sensitive; weak at shared 1e-5 |
| LOMO | fused backward, **no clip** | ✅ / ✅ | **14.60 GB** | 1.7 GB | 0.29 s | **0.00 / 0.59** | lightest, but does NOT learn at lr 1e-5 (SGD-like; needs higher LR + clipping) |
| ZeRO-Offload + fp32 Adam | DeepSpeed ZeRO-2, CPU optimizer offload | ❌ (RAM) / ❌ | — | ~95 GB (OOM) | — | — | fp32 CPUAdam ~87 GB > avail RAM; no 8-bit CPU optimizer |
| FSDP CPU-offload | — | not run | | | | | fp32 offload would OOM RAM like DeepSpeed |
| MeZO | — | not run | | | | | needs custom zeroth-order loop |
| ZeRO-Infinity (NVMe) | — | not run | | | | | fallback only — RAM didn't pinch the on-GPU routes |

**Combinations** (task843, `results/phase05/combinations.csv`): **BAdam + 8-bit base optimizer → 15.7 GB at 0.80 EM** (memory tricks stack: LOMO-class headroom *with* quality); GaLore rank 64 ≈ rank 128 quality at less state; LOMO LR sweep {1e-4,5e-4,1e-3} all fail (chance/divergence) → needs gradient clipping. See [`./phase-0.5-findings.md`](./phase-0.5-findings.md) for the full analysis.
