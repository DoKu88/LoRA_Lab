# Phase 0.5 — Full-FT Memory Optimizations: VRAM / RAM / Time Trade-offs

*Reference table for the Phase 0.5 feasibility spike — "can we full-finetune Mistral-7B on this box?" Pairs with [`../notes.md`](../notes.md) §C2 (Phase 0.5) and §B (32 GB practical tips). Section refs (§2.x) point at the lit-review entries in `../summaries.md`.*

---

## The wall

bf16 + standard Adam on Mistral-7B ≈ **14 GB weights + 14 GB grads + ~56 GB optimizer states ≈ 84 GB** — far over the hard ceiling of **32 GB VRAM + 32 GB system RAM**. CPU offload (the usual escape hatch) is largely blocked because we only have 32 GB system RAM; NVMe offload is the only offload path that survives, and it is slow + SSD-dependent.

The job of every technique below is to shrink that 84 GB into 32 GB of VRAM **without** leaning on system RAM we don't have.

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
| **Activation offload** | Moderate cut — pushes activations to CPU | **Raises RAM use** | Slower (PCIe traffic) | Stackable; eats your scarce 32 GB RAM |
| **Drop fp32 master copy** | Small-moderate cut — removes the fp32 weight shadow | Neutral | Negligible | Cheap stackable lever; minor stability risk |
| **ZeRO-Offload / FSDP CPU-offload** (§2.6 / §2.10) | **Large cut** — moves optimizer/grads to CPU | **Heavy — needs ~tens of GB RAM you don't have** | Slower (PCIe-bound) | ⚠️ **Largely blocked: only 32 GB system RAM.** The usual escape hatch is weak here |
| **ZeRO-Infinity (NVMe offload)** (§2.6) | **Largest cut** — streams states to SSD | Low RAM (uses disk) | **Large slowdown; SSD-dependent** | Only viable offload path given the RAM ceiling; needs a fast SSD |

---

## How to read this for the spike

- **VRAM-direct, RAM-neutral** (GaLore/Q-GaLore, LOMO/AdaLOMO, BAdam, MeZO) are the real candidates — they cut VRAM without leaning on the 32 GB RAM you don't have.
- **Offload paths** (ZeRO-Offload/FSDP) are the trap: they're the "obvious" answer but the 32 GB system-RAM ceiling neuters them. Only **ZeRO-Infinity (NVMe)** survives, at a heavy speed cost.
- **Stackable levers** (8-bit Adam, checkpointing, drop-fp32) don't fit 7B alone but multiply the headroom of whichever primary technique you pick.
- **Suggested order to benchmark:** try **Q-GaLore / GaLore** and **LOMO** first (most likely to fit at acceptable speed) → **BAdam** if you'll trade time → **MeZO** as last resort → **NVMe offload** only if all VRAM-direct methods fail.

---

## Results to fill in (per technique)

Measure with `torch.cuda.max_memory_allocated()` per phase (§B), on Mistral-7B-Instruct-v0.2, fixed batch/seq-len/seed.

| Technique | Config / flags | Fits in 32 GB? | Measured peak VRAM | Peak system RAM | Wall-clock / step | Quality (eval) | Notes |
|---|---|---|---|---|---|---|---|
| GaLore | | | | | | | |
| Q-GaLore | | | | | | | |
| LOMO | | | | | | | |
| AdaLOMO | | | | | | | |
| BAdam | | | | | | | |
| MeZO | | | | | | | |
| ZeRO-Infinity (NVMe) | | | | | | | |
