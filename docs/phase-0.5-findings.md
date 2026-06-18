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

**Yes — Mistral-7B can be full-parameter fine-tuned on this box, entirely on the
GPU; the CPU-offload route is the only one that fails.** Every VRAM-direct
technique fits in 32 GB and uses negligible system RAM (~2 GB). But **memory/speed
alone is misleading — the held-out eval is what picks the winner:**
- **Best memory-headroom + quality (for stacking a hypernetwork): BAdam + 8-bit
  Adam — ~15.7 GB, ~0.80 EM, ~16 GB of VRAM free.**
- **Best raw quality: Q-GaLore / GaLore (~0.83–0.88 EM)**, but ~28–30 GB and 2.7× slower.
- **LOMO looks best on paper (14.6 GB, fast) but does NOT learn at the shared LR
  (0.0 EM on the 3-class task)** — it's SGD-like and needs a different LR +
  gradient clipping. The earlier "LOMO" recommendation is *withdrawn* on the eval.
- The classic CPU-offload route (DeepSpeed ZeRO-Offload) is not viable: its fp32
  optimizer state (~87 GB) exceeds available RAM.

---

## Smoke-test results (Sprint 2 feasibility spike)

Mistral-7B, seq-len 512, micro-batch 1, gradient checkpointing on, 3 steps,
random-token batches (feasibility only — loss values not meaningful).

| Path | Optimizer | Fits? | Peak VRAM | Peak RAM | s/step | Verdict |
|---|---|---|---|---|---|---|
| **8-bit paged AdamW** (model on GPU) | `bnb.PagedAdamW8bit` | ✅ **yes** | **27.24 GB** | 4.5 GB | **1.70** | working fast baseline |
| DeepSpeed ZeRO-2 + CPU offload | `DeepSpeedCPUAdam` (fp32) | ❌ **no** | — | OOM-killed @ init | — | fp32 state ~84 GB > ~87 GB avail |

### Why fp32 ZeRO-Offload fails here (the memory math)
**fp32 is the killer, and DeepSpeed gives no way around it.** DeepSpeed CPU
offload runs the Adam step on CPU via its `DeepSpeedCPUAdam` kernel, which is
**fp32-only** (DeepSpeed has no 8-bit CPU optimizer). For 7.24 B params the CPU
must then hold three fp32 copies:

| fp32 state on CPU (ZeRO-Offload) | size |
|---|---|
| master weights | 29 GB |
| Adam momentum (m) | 29 GB |
| Adam variance (v) | 29 GB |
| **optimizer state total** | **~87 GB** |
| + bf16 model copy during init | ~14 GB |
| + pinned transfer buffers | several GB |

The box has 96 GB but only **~87 GB available** — so the fp32 triple-copy alone
fills RAM, and the process is SIGKILLed (exit 137) during
`deepspeed.initialize()`, before it even takes a step. The 96 GB upgrade that
`notes.md` assumed would unblock offload is *just barely* not enough.

**The irony:** the one thing that would rescue offload — 8-bit optimizer state
(~28 GB instead of 87 GB) — is exactly what DeepSpeed's CPU path lacks. And the
moment you *have* 8-bit Adam (bitsandbytes paged AdamW), the state shrinks so
much that you **don't need to offload at all** — it fits on the 32 GB GPU
(27 GB), with states mostly GPU-resident (hence the tiny 4.5 GB RAM and no
offload tax). So offloading wasn't conceptually wrong; it lost a footrace to the
on-GPU 8-bit path that the same idea (quantize the optimizer) enables. Routes
that *could* make offload work — an 8-bit CPU optimizer, or NVMe spill
(ZeRO-Infinity, much slower) — were unnecessary once the on-GPU methods won.

### Implication for the technique taxonomy
The "offload anchor" that actually *works* on this box is **8-bit paged AdamW**,
not DeepSpeed offload. fp32 DeepSpeed offload is recorded as a real `fits=no`
row. This also gives a clean ablation point: on the offload path, toggling the
8-bit lever flips feasibility (fp32 → OOM, 8-bit → fits in 27 GB).

## Fixed-protocol trade-off tables (real SNI data, with held-out eval)

Protocol: Mistral-7B, seq 512, batch 1 × grad-accum 8, **50 opt-steps**, seed 42,
lr 1e-5, grad-checkpointing on, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
**Held-out eval** = exact-match on each task's full test split (181 / 248 examples),
generated greedily from the just-trained model. `s/micro-batch` is the
apples-to-apples speed metric (LOMO/AdaLOMO fuse the update so 1 opt-step = 1
micro-batch; the others accumulate 8). Sources:
`results/phase05/feasibility_table.csv` (task843),
`results/phase05/feasibility_table_task1344.csv`; plots in `results/phase05/plots/`.

### Table 1 — task843 (financial sentiment, 3-class)

| Technique | Fits | Peak VRAM | **Peak RAM** | s/micro-batch | VRAM headroom (32−peak) | **eval EM** | final loss |
|---|---|---|---|---|---|---|---|
| **Q-GaLore** | ✅ | 28.59 GB | 1.9 GB | 0.75 | 3.4 GB | **0.878** | 0.35 |
| **GaLore** | ✅ | 30.41 GB | 1.9 GB | 0.75 | 1.6 GB | 0.867 | 0.42 |
| **BAdam** | ✅ | 17.60 GB | 1.9 GB | **0.13** | 14.4 GB | 0.856 | 0.25 |
| baseline (paged 8-bit) | ✅ | 27.64 GB | 1.9 GB | 0.28 | 4.4 GB | 0.608 | 0.49 |
| AdaLOMO | ✅ | 15.10 GB | 1.8 GB | 0.60 | 16.9 GB | 0.453 | 1.92 |
| LOMO | ✅ | **14.60 GB** | 1.7 GB | 0.29 | **17.4 GB** | **0.000** | 3.77 |
| fp32 ZeRO-Offload | ❌ | — | ~95 GB (OOM) | — | — | — | — |

### Table 2 — task1344 (RTE entailment, binary — the harder task)

| Technique | Fits | Peak VRAM | **Peak RAM** | s/micro-batch | VRAM headroom | **eval EM** | final loss |
|---|---|---|---|---|---|---|---|
| **GaLore** | ✅ | 30.43 GB | 2.0 GB | 0.78 | 1.6 GB | **0.835** | 0.15 |
| **Q-GaLore** | ✅ | 28.57 GB | 1.9 GB | 0.78 | 3.4 GB | 0.831 | 0.09 |
| **BAdam** | ✅ | 17.61 GB | 1.9 GB | 0.15 | 14.4 GB | 0.746 | 0.20 |
| LOMO | ✅ | 14.65 GB | 1.7 GB | 0.32 | 17.4 GB | 0.589 | 0.33 |
| baseline (paged 8-bit) | ✅ | 27.65 GB | 1.9 GB | 0.31 | 4.4 GB | 0.528 | 0.23 |
| AdaLOMO | ✅ | 15.15 GB | 1.8 GB | 0.62 | 16.9 GB | 0.431 | 5.25 |

### What the numbers say (and how the two tasks agree)
- **Feasibility:** every VRAM-direct technique fits; the offload route is the only
  family that fails (fp32 CPU-Adam state > RAM — see the breakdown above).
- **Peak RAM is trivial for every on-GPU method (~1.7–2.0 GB).** That's just
  Python + the data/model-load working set — these techniques **don't touch the
  96 GB pool at all** (states live on the GPU). System RAM is a non-constraint
  here; it only mattered for the offload route, which needed ~95 GB and OOMed.
- **Quality leaders are consistent across both tasks: Q-GaLore / GaLore / BAdam
  (0.83–0.88 EM).** baseline (paged 8-bit) is mid (0.53–0.61). **LOMO and AdaLOMO
  underperform** — and that only shows up in *eval*, not in memory/speed.
- **The LOMO trap:** memory/speed alone rank LOMO #1 (14.6 GB, fast). But it
  scored **0.000** on the 3-class task and only ~chance (0.589) on the binary
  task. LOMO is SGD-like, so the shared **lr 1e-5 (an Adam LR) is far too small**
  for it — it barely updates. This is the single most important reason the spike
  ran evals: the memory/speed table would have recommended a technique that
  doesn't learn. The combinations study below tried to rescue it with an LR sweep.
- **Speed (per micro-batch):** BAdam fastest (0.13 s — only one block's optimizer
  is live), baseline ~0.28 s, GaLore/Q-GaLore ~2.7× slower (per-step low-rank
  projection + periodic SVD).

## Method combinations — how these stack (`results/phase05/combinations.csv`)

Can we combine the levers/techniques to get *both* memory headroom and quality?
All on task843, same protocol.

| Combination | Peak VRAM | Peak RAM | s/micro | eval EM | What it tells us |
|---|---|---|---|---|---|
| **BAdam + 8-bit base optimizer** | **15.68 GB** | 1.9 GB | 0.13 | **0.801** | ★ memory tricks *stack*: BAdam (one live block) + 8-bit shrinks it 17.6→15.7 GB while keeping ~0.80 quality |
| GaLore rank 64 (vs 128) | 29.88 GB | 2.0 GB | 0.75 | 0.873 | rank is a near-free knob — half the projection state, same quality |
| LOMO lr 1e-4 | 14.60 GB | 1.7 GB | 0.29 | 0.331 | higher LR → only chance (0.33 = 1/3) |
| LOMO lr 5e-4 | 14.60 GB | 1.7 GB | 0.29 | 0.000 | diverged |
| LOMO lr 1e-3 | 14.60 GB | 1.7 GB | 0.29 | 0.000 | diverged |
| LOMO bs8 / no-ckpt (lr 5e-4) | 15.7 / 16.7 GB | 1.8 GB | — | 0.331 | headroom *can* buy batch/▼ckpt, but quality still chance |
| AdaLOMO lr 1e-3 | 15.10 GB | 1.8 GB | 0.60 | 0.331 | chance |

**How to combine — the practical guidance:**
1. **Memory tricks stack cleanly.** BAdam (block-coordinate) + 8-bit base optimizer
   compose: **15.7 GB at 0.80 EM** — i.e. *LOMO-class memory headroom (~16 GB free)
   with actual quality*. This is the combination to use when you need both room
   for a hypernetwork **and** a model that learns. Gradient checkpointing stacks
   on top of all of these (it's independent — see the ablation study).
2. **GaLore's rank is a free dial.** rank 64 ≈ rank 128 quality at less projection
   memory; drop it if you want a bit more headroom from the GaLore family.
3. **LOMO does not combine its way to quality in this budget.** No LR in
   {1e-5, 1e-4, 5e-4, 1e-3} clears chance on task843: too small → undertrained
   (0.33), too large → divergence (0.00). LOMO needs **gradient clipping** (its
   paper's two-pass `grad_norm`+`fused_backward`, which we disabled for speed)
   and/or many more steps. Spending its headroom on batch size / dropping
   checkpointing doesn't help until the optimizer itself is stabilized. Treat
   LOMO's memory win as *unrealized* until clipping is added (a follow-up).

## Recommendation (corrected by the eval results)
- **Best memory headroom *with* quality → BAdam + 8-bit Adam (~15.7 GB, ~0.80 EM,
  ~16 GB free).** This is the route to stack a hypernetwork on (Phase 2): it
  leaves roughly half the GPU free and actually learns. *(The earlier
  memory-only pick, LOMO, is withdrawn — it doesn't learn at this LR.)*
- **Best raw quality → Q-GaLore or GaLore (0.83–0.88 EM)**, but at ~28–30 GB they
  leave little headroom and are ~2.7× slower; pick these if quality is paramount
  and you don't need room on the GPU for anything else.
- **Simplest / most standard → paged 8-bit AdamW baseline** (27.6 GB), but its
  quality is mid (0.53–0.61) at 50 steps.
- **Avoid:** plain LOMO/AdaLOMO at the shared LR (don't learn), and fp32 DeepSpeed
  offload (OOM).

## Caveats
- 50-step benchmark — short. Quality numbers are *comparative under a fixed budget*,
  not converged accuracy; absolute EM would rise with more steps. The cross-task
  *ranking* is the trustworthy part.
- **LOMO/AdaLOMO ran without gradient clipping** (single fused pass, for speed);
  their poor quality is partly this. A clipped re-run is the obvious follow-up
  before writing LOMO off entirely.
- On-GPU 7B full FT trains **bf16 weights, no fp32 master** (29 GB master won't
  fit). **Scope:** affects only *full-finetuning the 7B* — **not** Text-to-LoRA,
  where the base is frozen and only the small hypernetwork/LoRA trains (full fp32
  there). See the dedicated note above.
- **BAdam** covers ~10 of 32 blocks in 50 steps (switch_every=5); full coverage
  needs a longer run. Memory/speed/quality profile is representative.
- **MeZO** and **FSDP CPU-offload** not run (MeZO needs a custom zeroth-order
  loop; FSDP fp32 offload would OOM RAM like DeepSpeed).
