# Phase 0.5 — Full-FT Memory Optimizations: VRAM / RAM / Time Trade-offs

*Reference table for the Phase 0.5 feasibility spike — "can we full-finetune Mistral-7B on this box?" Pairs with [`../notes.md`](../notes.md) §C2 (Phase 0.5) and §B (practical tips). Section refs (§2.x) point at the lit-review entries in `../summaries.md`.*

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

Measure with `torch.cuda.max_memory_allocated()` per phase (§B), on Mistral-7B-Instruct-v0.2, fixed batch/seq-len/seed.

| Technique | Config / flags | Fits (≤32 GB VRAM / ≤96 GB RAM)? | Measured peak VRAM | Peak system RAM | Wall-clock / step | Quality (eval) | Notes |
|---|---|---|---|---|---|---|---|
| ZeRO-Offload + 8-bit Adam | | | | | | | baseline "just works" route |
| FSDP CPU-offload | | | | | | | |
| GaLore | | | | | | | |
| Q-GaLore | | | | | | | |
| LOMO | | | | | | | |
| AdaLOMO | | | | | | | |
| BAdam | | | | | | | |
| MeZO | | | | | | | |
| ZeRO-Infinity (NVMe) | | | | | | | fallback only |
