# Phase 0.5 — Findings: Full Fine-Tuning Mistral-7B on 32 GB VRAM + 96 GB RAM

*Live results log for the Phase 0.5 spike. Pairs with the sprint plan
([`phase-0.5-sprint-plan.md`](./phase-0.5-sprint-plan.md)) and the technique
reference ([`llm_optimizations.md`](./llm_optimizations.md)). Numbers are filled
in as each technique is benchmarked; the final trade-off table + ablation study
are assembled in Sprint 7.*

**Hardware:** RTX 5090 (32 GB, sm_120) + 96 GB system RAM (~87 GB available),
torch 2.10+cu128, conda env `lora_lab`. Base: `Mistral-7B-Instruct-v0.2` (7.24 B).

---

## Headline result (feasibility)

**Yes — Mistral-7B can be full-parameter fine-tuned on this box, and the fast
route keeps everything on the GPU.** The working recipe is **bf16 weights on the
GPU + bitsandbytes paged 8-bit AdamW + gradient checkpointing**: ~27 GB VRAM,
~1.7 s/step, negligible RAM. The classic CPU-offload route (DeepSpeed
ZeRO-Offload) is *not* viable here because its fp32 optimizer state exceeds the
RAM budget.

---

## Smoke-test results (Sprint 2 feasibility spike)

Mistral-7B, seq-len 512, micro-batch 1, gradient checkpointing on, 3 steps,
random-token batches (feasibility only — loss values not meaningful).

| Path | Optimizer | Fits? | Peak VRAM | Peak RAM | s/step | Verdict |
|---|---|---|---|---|---|---|
| **8-bit paged AdamW** (model on GPU) | `bnb.PagedAdamW8bit` | ✅ **yes** | **27.24 GB** | 4.5 GB | **1.70** | working fast baseline |
| DeepSpeed ZeRO-2 + CPU offload | `DeepSpeedCPUAdam` (fp32) | ❌ **no** | — | OOM-killed @ init | — | fp32 state ~84 GB > ~87 GB avail |

### Why fp32 ZeRO-Offload fails here (the memory math)
DeepSpeed CPU offload mandates the fp32 `DeepSpeedCPUAdam`; for 7.24 B params the
CPU must hold fp32 **master (29 GB) + momentum (29 GB) + variance (29 GB) ≈ 87 GB**
of optimizer state, plus pinned buffers and the model — over the ~87 GB of
available RAM. Process is SIGKILLed (exit 137) during `deepspeed.initialize()`.
DeepSpeed has **no 8-bit CPU optimizer**, so the "ZeRO-Offload + 8-bit Adam" idea
can't be realized through DeepSpeed; the 8-bit win comes from bitsandbytes paged
AdamW instead (states in 8 bits, paged to CPU on demand — but at 7B they mostly
stay GPU-resident, hence the tiny 4.5 GB RAM use and no offload tax).

### Implication for the technique taxonomy
The "offload anchor" that actually *works* on this box is **8-bit paged AdamW**,
not DeepSpeed offload. fp32 DeepSpeed offload is recorded as a real `fits=no`
row. This also gives a clean ablation point: on the offload path, toggling the
8-bit lever flips feasibility (fp32 → OOM, 8-bit → fits in 27 GB).

*(Full fixed-protocol runs on real SNI data — 50 steps, with eval quality — and
the remaining techniques follow; this table is updated as they land.)*
