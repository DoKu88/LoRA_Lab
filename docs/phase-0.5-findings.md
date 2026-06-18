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

## Fixed-protocol trade-off table (real SNI data)

Protocol: Mistral-7B, `task843_financial_phrasebank_classification`, seq 512,
batch 1 × grad-accum 8, **50 opt-steps**, seed 42, grad-checkpointing on. All
on-GPU techniques run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Source: `results/phase05/feasibility_table.csv`; plots in
`results/phase05/plots/`.

| Technique | Fits | Peak VRAM | Peak RAM | s/opt-step | **s/micro-batch** | Headroom (32−VRAM) | final loss |
|---|---|---|---|---|---|---|---|
| **LOMO** | ✅ | **14.60 GB** | 1.7 | 0.29¹ | **0.29** | **17.4 GB** | 0.61 |
| AdaLOMO | ✅ | 15.10 GB | 1.8 | 0.60¹ | 0.60 | 16.9 GB | 1.93² |
| BAdam | ✅ | 17.60 GB | 1.9 | 1.03 | **0.13** | 14.4 GB | 0.57 |
| Q-GaLore | ✅ | 28.59 GB | 1.9 | 6.03 | 0.75 | 3.4 GB | 0.35 |
| baseline (paged 8-bit) | ✅ | 27.64 GB | 1.9 | 2.26 | 0.28 | 4.4 GB | 0.44 |
| GaLore | ✅ | 30.41 GB | 1.9 | 6.03 | 0.75 | 1.6 GB | 0.44 |
| fp32 ZeRO-Offload | ❌ | — | ~95 GB | — | — | — | — |

¹ LOMO/AdaLOMO fuse the update into backward (no grad accumulation), so one
"opt-step" = **1** micro-batch; the others accumulate **8**. The **s/micro-batch**
column is the apples-to-apples speed metric. ² AdaLOMO's high loss is an LR
mismatch at the shared 1e-5 (its adaptive scaling wants a different LR), not an
instability — flagged for a per-technique LR if it's used for real.

### What the numbers say
- **Feasibility: every VRAM-direct technique fits; the offload route does not.**
  The plan expected offload to be the easy path; on this box it's the *only*
  family that fails (fp32 CPU-Adam state > RAM).
- **Memory:** LOMO/AdaLOMO are far the lightest (~15 GB — half the GPU free),
  then BAdam (17.6), then the 8-bit/GaLore family (~28–30 GB, near the ceiling).
- **Speed (per micro-batch, the fair metric):** BAdam fastest (0.13 s — only one
  block's optimizer is live), baseline and **LOMO tie at ~0.28 s** (LOMO's win is
  *memory*, not speed), AdaLOMO ~2×, **GaLore/Q-GaLore ~2.7× slower** (the
  per-step low-rank projection + periodic SVD is real overhead).

## Recommendation (fastest viable route + hypernetwork headroom)
- **For Phase 2 (hypernetwork on top of the 7B): LOMO.** ~14.6 GB leaves **~17 GB
  of VRAM headroom** for the hypernetwork + its activations, at baseline-level
  per-token speed. BAdam (17.6 GB, fastest/token) is the runner-up.
- **Simplest robust choice if headroom isn't the constraint: paged 8-bit AdamW**
  (the baseline) — standard Adam dynamics, 27.6 GB, no projection/cycling quirks.
- **Avoid:** GaLore/Q-GaLore here — they're both the tightest on VRAM *and* the
  slowest, the worst corner of the trade-off; and fp32 DeepSpeed offload (OOM).

## Caveats
- 50-step benchmark measures **memory + speed**, not convergence. `final_train_loss`
  is a learning signal, not a quality verdict. **Full held-out eval quality is the
  one remaining Sprint 7 item** (the runs use `save_checkpoint=False` to protect
  disk — eval needs an inline-eval pass or a targeted re-run of the chosen route).
- On-GPU 7B full FT trains **bf16 weights without an fp32 master copy** (a 29 GB
  master doesn't fit) — a known precision caveat for all rows. **Scope:** this
  applies only to *full-finetuning the 7B*. It does **not** affect the project's
  Text-to-LoRA workflow (§1.1), where the 7B base is *frozen* and only the small
  hypernetwork/LoRA is trained — those keep full fp32 Adam easily, and the
  caveat never bites. It would only matter if a later phase full-finetunes the
  base itself (e.g. an oracle baseline); the eval-quality column below measures
  how much it actually costs. LOMO/GaLore are designed for bf16 full-FT, so the
  hit is typically small.
- **BAdam** covers ~10 blocks in 50 steps (switch_every=5); full coverage of all
  32 blocks needs a longer run. Memory/speed profile is representative.
- **MeZO** and **FSDP CPU-offload** are not yet run (MeZO needs a custom
  zeroth-order loop; FSDP fp32 offload would OOM RAM like DeepSpeed).
