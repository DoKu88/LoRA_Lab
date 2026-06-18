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
- **Best memory-headroom + quality (for stacking a hypernetwork): LOMO with
  gradient clipping @ lr 5e-4 — 14.6 GB, 0.84 EM, ~17 GB of VRAM free.** The
  lightest route, and once clipped it reaches top-tier quality.
- **Fastest at competitive quality: BAdam (17.6 GB, 0.13 s/micro, 0.86 EM)**, or
  BAdam + 8-bit (15.7 GB, 0.80 EM) if you want BAdam at LOMO-class memory.
- **Best raw quality: Q-GaLore / GaLore (~0.83–0.88 EM)**, but ~28–30 GB and 2.7× slower.
- **The LOMO lesson:** *unclipped* LOMO does NOT learn at the shared lr 1e-5
  (0.0 EM) and diverges at higher LR — memory/speed alone would have mis-picked
  it. **Gradient clipping + lr ~5e-4 fixes it** (0.0 → 0.84), restoring LOMO as
  the headroom winner. This round-trip (picked → withdrawn on eval → restored
  with the right recipe) is the spike's clearest lesson: *measure quality, and
  tune the optimizer to its family.*
- **CPU offload is viable — but only via FSDP, and it's ~45× slower.** DeepSpeed
  ZeRO-Offload OOMs (its fp32 master copy pushes state to ~87 GB > available RAM),
  but **FSDP CPU-offload fits at 62 GB RAM** (bf16 params, no fp32 master) and
  trains — at **~9–13 s/step vs ~0.3 s on-GPU**. So offload works as a fallback,
  not a default.
- **MeZO is the memory floor (13.8 GB — no grads, no optimizer state) but does not
  converge in this budget** (eval 0.0 even at 500 steps; zeroth-order needs
  thousands). It *fits* easily; it doesn't *learn* fast enough to be useful here.

---

## Smoke-test results (Sprint 2 feasibility spike)

Mistral-7B, seq-len 512, micro-batch 1, gradient checkpointing on, 3 steps,
random-token batches (feasibility only — loss values not meaningful).

| Path | Optimizer | Fits? | Peak VRAM | Peak RAM | s/step | Verdict |
|---|---|---|---|---|---|---|
| **8-bit paged AdamW** (model on GPU) | `bnb.PagedAdamW8bit` | ✅ **yes** | **27.24 GB** | 4.5 GB | **1.70** | working fast baseline |
| **FSDP CPU-offload** (params→CPU) | fp32 AdamW | ✅ **yes** | 27.5 GB | **62 GB** | **~9–13** | offload *works* — but PCIe-bound, ~45× slower |
| DeepSpeed ZeRO-2 + CPU offload | `DeepSpeedCPUAdam` (fp32) | ❌ **no** | — | OOM-killed @ init | — | fp32 state ~87 GB > ~87 GB avail |

**The two offload frameworks diverge — and the reason is instructive.** Same idea
(optimizer/params on CPU), opposite outcome: **DeepSpeed OOMs, FSDP fits.**
DeepSpeed ZeRO-Offload keeps an **fp32 master copy** of the weights on CPU
(+29 GB) on top of fp32 m+v; FSDP with bf16 mixed precision keeps **bf16 params,
no separate fp32 master**, so its CPU footprint is ~62 GB — under the ~87 GB
ceiling. So "offload" isn't one thing: the *framework's* precision bookkeeping
decides feasibility here. FSDP proves offload is reachable on this box; its
~9–13 s/step (vs ~0.3 s on-GPU) is the PCIe tax that makes it a fallback, not a
default. (Single-GPU FSDP runs `NO_SHARD` + `CPUOffload` — nothing to shard, but
params still stream CPU↔GPU.)

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
| LOMO (no clip) | ✅ | 14.60 GB | 1.7 GB | 0.29 | 17.4 GB | **0.000** | 3.77 |
| MeZO | ✅ | **13.77 GB** | 5.2 GB | 0.22³ | **18.2 GB** | **0.000** | 12.0 |
| FSDP CPU-offload | ✅ | 27.50 GB | **62.5 GB** | ~9.4⁴ | 4.5 GB | n/a⁴ | ~0.47 |
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

*(MeZO and FSDP measured on task843 only — see Table 1. ³ MeZO's "step" is 2
forward passes (no backward); even at 500 steps eval stays 0.0 — zeroth-order
needs thousands. ⁴ FSDP s/step is offload-bound (~9–13 s); its quality wasn't
eval'd — generation under CPU-offload is impractically slow and FSDP isn't a
recommended route given the ~45× train slowdown; loss does decrease.)*

### What the numbers say (and how the two tasks agree)
- **Feasibility:** every VRAM-direct technique fits; **FSDP offload also fits**
  (62 GB RAM) but is ~45× slower; **only DeepSpeed offload fails** (fp32 master
  pushes it over RAM — see the breakdown above). **MeZO fits (13.8 GB floor) but
  doesn't converge** in budget.
- **Peak RAM splits the families cleanly:** on-GPU methods use **~1.7–5 GB**
  (just Python + working set — they never touch the 96 GB pool), while the
  **offload routes are RAM-bound** (FSDP 62 GB; DeepSpeed wanted ~87 GB and
  OOMed). So system RAM is a non-constraint *unless* you offload — which is the
  whole reason the on-GPU methods win here.
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

## LOMO with gradient clipping — the fix (`results/phase05/clipped_lomo.csv`)

Unclipped LOMO failed two ways (undertrained at lr 1e-5, diverged at 5e-4/1e-3).
LOMO's paper uses a two-pass clipped update (`grad_norm` then `fused_backward`
over a retained graph); we'd disabled it for speed. Re-running *with* clipping:

| Run (task843) | Peak VRAM | Peak RAM | s/micro-batch | eval EM |
|---|---|---|---|---|
| **LOMO clip, lr 5e-4** | **14.60 GB** | 1.8 GB | 0.47 | **0.840** |
| LOMO clip, lr 1e-3 | 14.60 GB | 1.8 GB | 0.47 | 0.840 |
| LOMO clip, lr 1e-4 | 14.60 GB | 1.8 GB | 0.47 | 0.812 |
| AdaLOMO clip, lr 5e-4 | 15.10 GB | 1.8 GB | 0.81 | 0.624 |
| AdaLOMO clip, lr 1e-3 | 15.10 GB | 1.8 GB | 0.81 | 0.337 |

**Clipping resolves the LOMO problem.** At lr 5e-4 it jumps **0.00 → 0.84 EM** —
tied with the quality leaders (Q-GaLore 0.88, BAdam 0.86) but at the **lowest
VRAM of any technique (14.6 GB, ~17 GB free)**. The clipped update is two
backward passes, so speed drops from 0.29 → 0.47 s/micro-batch — still faster
than GaLore (0.75) and the memory is unchanged (`grad_norm` clears grads, adds
nothing). **AdaLOMO does not benefit** — adaptive scaling + clipping interacts
poorly and stays LR-sensitive (0.62 / 0.34); use plain clipped LOMO.

## Recommendation (final, eval-driven)
- **★ Best for stacking a hypernetwork (max headroom + quality): LOMO + gradient
  clipping @ lr ~5e-4 — 14.6 GB, 0.84 EM, ~17 GB VRAM free, 0.47 s/micro.** The
  lightest route, top-tier quality once clipped. Use a per-technique LR (≈5e-4),
  not the Adam-scale 1e-5.
- **Fastest at competitive quality: BAdam (17.6 GB, 0.13 s/micro, 0.86 EM)** — or
  **BAdam + 8-bit (15.7 GB, 0.80 EM)** for BAdam at LOMO-class memory. Pick BAdam
  if per-step speed matters more than the last ~3 GB of headroom.
- **Best raw quality: Q-GaLore / GaLore (0.83–0.88 EM)** — but ~28–30 GB (little
  headroom) and ~2.7× slower. Choose only if quality is paramount and the GPU
  doesn't need room for anything else.
- **Simplest / most standard: paged 8-bit AdamW baseline** (27.6 GB) — mid quality
  (0.53–0.61) at 50 steps; fine if you just want vanilla Adam dynamics.
- **Avoid:** *unclipped* LOMO/AdaLOMO (don't learn), AdaLOMO generally (LR-fragile),
  and fp32 DeepSpeed offload (RAM OOM).

## Caveats
- 50-step benchmark — short. Quality numbers are *comparative under a fixed budget*,
  not converged accuracy; absolute EM would rise with more steps. The cross-task
  *ranking* is the trustworthy part.
- The unclipped LOMO/AdaLOMO rows in Tables 1–2 use a single fused pass (no
  clipping) — that's *why* they underperform. **Resolved:** the clipped re-run
  (above) restores LOMO to 0.84 EM; use clipped LOMO @ lr ~5e-4 in practice.
  AdaLOMO stays weak even clipped.
- On-GPU 7B full FT trains **bf16 weights, no fp32 master** (29 GB master won't
  fit). **Scope:** affects only *full-finetuning the 7B* — **not** Text-to-LoRA,
  where the base is frozen and only the small hypernetwork/LoRA trains (full fp32
  there). See the dedicated note above.
- **BAdam** covers ~10 of 32 blocks in 50 steps (switch_every=5); full coverage
  needs a longer run. Memory/speed/quality profile is representative.
- **MeZO** (Sprint 8) runs at the 13.8 GB memory floor but stays at 0.0 EM even
  at 500 steps — zeroth-order needs orders more steps; not usable in this budget.
- **FSDP CPU-offload** (Sprint 8) *fits* (62 GB RAM, no fp32 master — unlike
  DeepSpeed) and learns, but at ~9–13 s/step it's ~45× slower than on-GPU; a
  viable fallback, not a default. Its eval quality wasn't measured (generation
  under offload is impractically slow).
- **ZeRO-Infinity (NVMe)** still not run — unnecessary, since FSDP already shows
  the offload route fits in RAM (no need to spill to disk).
