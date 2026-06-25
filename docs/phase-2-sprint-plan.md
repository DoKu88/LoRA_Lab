# Phase 2 — Sprint Plan: Train the T2L-style Hypernetwork

*Sprint-planning material for Phase 2. Pairs with [`../notes.md`](../notes.md) §C2 (timeline — Phase 2), §A (research questions), §B (memory tips), and the lit entries in [`../summaries.md`](../summaries.md): Text-to-LoRA §1.1, HyperNetworks §1.3, TAGI §1.10, HypeLoRA §1.6, HyperLoader §1.9. Consumes the Phase 1 library ([`phase-1-findings.md`](./phase-1-findings.md)): the train-split adapters + task descriptions (training data) and the locked held-out split + oracle LoRAs (eval baseline).*

---

## What we are trying to achieve in Phase 2

Train a **text-conditioned hypernetwork** that, in a **single forward pass**, generates a competent LoRA adapter for `Mistral-7B-Instruct-v0.2` from a natural-language **task description** — the T2L recipe (§1.1). Then evaluate generated adapters on **held-out tasks** against three honest baselines, and clear the phase's critical gate.

> **The gate is the whole point (§C, §A.10):** generated LoRAs must **beat a nearest-neighbor retrieval baseline** on held-out tasks. Matching the base model or losing to "just retrieve the closest library LoRA" means the hypernetwork isn't *generalizing* — it's memorizing or worse. Everything downstream (the Phase-3 interpretability comparison) is meaningless if the generated adapter isn't a real, functioning adapter. So Phase 2 is not "build a hypernetwork" — it's "build one that demonstrably generalizes."

**Training objective — SFT, not reconstruction (project-defining, carried from notes.md §C2):** T2L can be trained two ways (§1.1 Fig. 1):
- **Reconstruction** (Eq. 6) — L1-regress the generated ΔW directly onto a target library LoRA. *No base-model forward pass* → cheap. But it **fails to generalize to unseen tasks** (T2L Table 6) and would make the Phase-3 interp comparison **circular** (the hypernetwork is trained to *copy* hand-trained LoRAs in weight space, so "do they share feature geometry?" answers itself).
- **SFT** (Eq. 5) — apply the generated ΔW to the frozen base, run the task, and **backprop the task loss through the frozen 4-bit base into the hypernetwork**. This lets the hypernetwork find its *own* solution. **Every gate-clearing run is SFT.** Reconstruction is used only as cheap plumbing / a warmup init (Sprints 1, 3).

**Headline deliverables:**
1. A trained hypernetwork checkpoint that generates LoRAs for Mistral-7B from a task description (the T2L meta-model).
2. A **held-out eval table** vs. three baselines (trained-LoRA upper bound, base lower bound, retrieval), with the **gate verdict** (generated > retrieval?).
3. `docs/phase-2-findings.md` — the architecture, the SFT recipe, the memory profile of the backprop-through-base path, the gate result, and the failure-mode analysis (§A.12: bad task *identification* vs bad task *execution*).

---

## Hard constraints (these drive every design choice)

- **Single GPU: 32 GB VRAM.** Unlike Phase 1 (inference-only), Phase 2 **trains**, and the memory-critical step is **backprop through the frozen quantized base into the hypernetwork** (notes.md §B). The base is a **frozen 4-bit NF4 feature extractor** (QLoRA-style), so the situation is close to QLoRA, not full pretraining — but the activation/gradient path through a 7B still needs **gradient checkpointing** and possibly **activation offload**. Profile this first; it's where we OOM.
- **Output dimensionality dominates trainability *and* memory (§A.2).** A Mistral-7B LoRA over q/k/v at rank 16 across 32 layers is ~9.4 M parameters (from Phase 1's oracle: `trainable% = 0.13`). A hypernetwork that emits *all* of that per task is a large output space. Start with the **smallest viable target** and grow only if the gate needs it (see the output-parameterization decision in S2).
- **The held-out split is frozen once re-locked.** The original 9-task pilot split is replaced by the curated **~400 train / 10 val / 30 held-out** spec (see the Phase-2 run spec above); re-locking re-runs `make_split` → a new `lock_hash` in `heldout_split.yaml`, which is then the contract. Train **only** on the train-split tasks; the held-out tasks (and their descriptions) are never seen during meta-training. Leakage invalidates the gate.
- **Library quality is assumed, not re-litigated.** Phase 2 distills the **gate-passing** Phase-1 library only (quarantined adapters like task639 are excluded by construction). Garbage-in is already filtered.
- **Plumbing before scale (notes.md §B, §A — the single most repeated lesson).** Get the generate→apply→backprop loop green on a **tiny** model (SmolLM2-135M / Qwen-0.5B) with **3 toy tasks** before touching Mistral-7B. A working loop on a small model de-risks the whole phase.
- **Determinism / W&B.** Seeded; every run logs to W&B best-effort/non-blocking with offline fallback (project working pref), reusing `train/run_logger.py`.

### What we reuse from Phases 0 / 1 (don't rebuild)

| Need | Reuse |
|---|---|
| Task descriptions (conditioning input) + train/val/held-out split | Phase 1 `configs/phase1/{library,heldout_split}.yaml` |
| Per-task target LoRAs (reconstruction targets) | Phase 1 library adapters (train split) |
| **trained-LoRA upper bound** + **base lower bound** baselines | Phase 1 oracle LoRAs (held-out) + bare base |
| SNI data, chat-templating, prompt-masking, held-out eval view | `src/lora_lab/data/sni.py` |
| Held-out generation + EM/ROUGE-L scoring | `src/lora_lab/eval/{evaluate,metrics}.py` |
| VRAM/RAM tracing, peak memory | `src/lora_lab/utils/vram.py` |
| W&B logging | `src/lora_lab/train/run_logger.py` |
| 4-bit base load (NF4) + PEFT LoRA injection | `src/lora_lab/methods/build.py` (qlora path) |

---

## Phase-2 run spec — data scale, split, training, and outputs

> Supersedes the original **pilot** split (9 train / 1 val / 3 held-out, built from
> the 14 `pilot: true` tasks). See [`phase-2-hypernet-sizing-options.md`](./phase-2-hypernet-sizing-options.md)
> for the full rationale and the held-out-axis comparison table.

### 1. How much data are we training on?

| Split | Tasks | Notes |
|---|---|---|
| **Train** | **~400** | Drawn broadly across all families from the **1,037 gate-passing** library tasks (564 generation + 473 classification). T2L used 479 — we match that order of magnitude. |
| **Val** | **10** | Early-stop / checkpoint selection only. |
| **Held-out (gate)** | **30** | Curated for diagnostic power (see §2). Never seen in training — the leakage contract. |
| *(reserved)* | ~600 | Remaining passing tasks held in reserve for Phase-4 scaling ablations. |

- **SFT (S4):** batch 4 × **~6,000 steps** ≈ **24,000 example-passes** (~60 per train
  task, sampled with replacement across the ~400 tasks).
- **Reconstruction warmup (S3):** ~400 train-split library LoRA adapters as ΔW
  targets, ~2,000 steps (no base forward).

### 2. What is the held-out set?

A **curated 30-task held-out set** built for three generalization axes where the
retrieval baseline is *plausible but wrong* — so a pass is unambiguous (full
analysis + per-axis table in the sizing-options doc):

| Axis | Held-out picks | # | Why it proves generalization |
|---|---|---|---|
| **Format transfer** ⭐ | hold out the `*_answer_generation` form of datasets whose `*_classification` form is trained (31 paired datasets available) | 15 | retrieval lands on the same-topic, wrong-output-format LoRA → fails. |
| **Language transfer** | hold out translation pairs unseen in that direction | 8 | retrieval returns a different-language LoRA → fails. |
| **Domain transfer** | same skill, new domain (e.g. train Amazon/Yelp sentiment, hold out twitter_emotion / financial_phrasebank / bengali_reviews) | 7 | tougher retrieval competitor → measures skill-vs-domain. |

> The pilot **accidentally trained on** `twitter_emotion`, `financial_phrasebank`,
> and `amazon_reviews` — these become **held-out** domain-transfer targets here.

### 3. Training broadly across tasks

Training samples **uniformly across all ~400 train tasks and all families**
(classification, generation, translation, NER, QA, …) — not a single-family
diet. Breadth is what lets the hypernetwork learn the *description → adapter* map
rather than memorize a few tasks; it is the precondition for the held-out axes in
§2 to be reachable. Re-lock the split with `make_split` (curated variant) → new
`lock_hash` in `configs/phase1/heldout_split.yaml`.

### 4. Training-time estimate (single 32 GB GPU)

| Stage | Steps | Est. wall-clock | Notes |
|---|---|---|---|
| Recon warmup (S3) | ~2,000 | **~20–30 min** | no base forward → cheap; scales with hypernet size. |
| SFT meta-train (S4) | ~6,000 (batch 4) | **~5–7.5 hr** | base-backprop-bound (~3–4.5 s/step); independent of #tasks, scales with #steps. |
| Held-out eval (S5) | 30 tasks × 4 conditions | **~1–2 hr** | generation + scoring on the held-out test splits. |
| **Total (default VeRA)** | — | **< 10 hr** | T2L-M parameterization ≈ same; T2L-L adds OOM-risk/throttle (see sizing doc). |

### 5. Charts, figures & tables we will output (→ `results/phase2/`, `docs/phase-2-findings.md`)

**Tables**
- **T1 — Four-way held-out gate table** (primary): per-task `generated / oracle / base / retrieval` score, margin-vs-retrieval, margin-vs-base, verdict. `{csv,parquet,md}`.
- **T2 — Per-axis gate summary**: mean score + margin grouped by held-out axis (format / language / domain) + aggregate, with the pass/fail verdict per axis.
- **T3 — Hypernet size & memory**: param count + peak VRAM per parameterization (VeRA vs full), measured (replaces the predicted sizing-options table).
- **T4 — Data-scale summary**: tasks & example-passes per split (the §1 numbers, as run).

**Figures**
- **F1 — Generalization curve** (headline): score vs description-embedding distance to nearest train task, one line each for generated / retrieval / oracle / base. Generalization shows as generated staying high while retrieval decays with distance.
- **F2 — Generated-vs-retrieval scatter**: one point per held-out task; above the diagonal = a win. The visual gate.
- **F3 — Training curves**: recon + SFT loss vs step (W&B).
- **F4 — Peak-VRAM-per-phase bar**: the backprop-through-base memory profile (the 32 GB constraint).
- **F5 — Per-axis margin bars**: mean margin vs retrieval and vs base, per generalization axis.

### Proposed repo layout

```
src/lora_lab/hypernet/
  encoder.py       task description -> task embedding (frozen sentence encoder, e.g. the base's own embeddings or a small ST model)
  heads.py         shared trunk + per-(layer, module) output heads -> LoRA A/B factors
  model.py         the HyperLoRA module: (description, layer_id, module_id) -> ΔW; assembles a PEFT-applyable adapter
  apply.py         inject a generated LoRA into the (frozen) base as a live PEFT adapter
  recon.py         reconstruction objective (Eq. 6) — L1 on ΔW vs a target library LoRA (no base forward)
  sft.py           SFT objective (Eq. 5) — apply ΔW, task forward, backprop task loss through frozen base
  retrieval.py     nearest-neighbor retrieval baseline (description-embedding index over the train split)
  meta_train.py    the meta-training loop (sample task -> generate -> objective -> step), W&B + VRAM traces
configs/phase2/    tiny-plumbing.yaml, recon-warmup.yaml, sft-mistral.yaml (one per stage/run)
results/phase2/    held-out eval table {csv,parquet,md}, mem traces, hypernet checkpoints (gitignored; regenerable)
docs/phase-2-findings.md
```

---

## Sprints

Each sprint lists: **(1) Goal · (2) Requirements · (3) Definition of done · (4) Required testing.**

> **Pre-flight (before any Mistral-7B meta-training run):** Phase-1 gate complete and the train/held-out split finalized; the tiny-model plumbing loop (S1) green; a 4-bit Mistral base loads and a *hand-made* LoRA backprops a task loss into a dummy parameter without OOM (the S4 memory check, run on a single batch first).

### Sprint 1 — Tiny-model plumbing: generate → apply → backprop  *(BLOCKER — must finish first)*

1. **Goal:** A working end-to-end loop on a **tiny** base (SmolLM2-135M / Qwen2.5-0.5B) and **3 toy SNI tasks**: a stub hypernetwork emits a LoRA from a task embedding, the LoRA is applied to the base, a forward runs, and a loss backpropagates **into the hypernetwork**. Plumbing only — correctness of the wiring, not quality.
2. **Requirements:**
   - `apply.py`: take generated A/B tensors and inject them as a live LoRA on the base (PEFT custom adapter or manual hooks), so the base forward uses ΔW and grads flow back to A/B.
   - A stub `model.py` producing correctly-shaped A/B for every target (layer, module).
   - Run both objectives once: **reconstruction** (regress onto a toy target LoRA — no base forward) and a **one-batch SFT** (task loss through the base). Assert grads reach the hypernetwork params.
   - Log VRAM-per-phase (§B) even at tiny scale, to validate the tracer on this path.
3. **Definition of done:** one full generate→apply→forward→backprop step completes on the tiny base for both objectives; the hypernetwork's parameters receive non-zero gradients; shapes match the base's LoRA target modules exactly.
4. **Required testing:** assert generated A/B shapes match `(r × in)` / `(out × r)` for every targeted projection; assert `hypernet.grad` is non-None and finite after `backward()`; a no-op adapter (A·B = 0 at init) leaves base logits unchanged (apply-correctness); reconstruction loss decreases on a single overfit toy target.

### Sprint 2 — Hypernetwork architecture + output parameterization  *(needs S1; the design decision)*

1. **Goal:** The real `HyperLoRA` architecture and a **committed output-parameterization choice** — the single biggest lever on trainability and memory (§A.2).
2. **Requirements:**
   - **Encoder** (`encoder.py`): task description → fixed task embedding. Default: a frozen sentence encoder (small, cheap); record the choice. The embedding conditions generation (§1.1).
   - **Conditioning scheme** (§1.9 HyperLoader, §1.6 HypeLoRA): generate per **(layer, module)** via learned layer/module embeddings + the task embedding through a shared trunk and small output heads — so cross-layer structure is shared, not 32× independent generators.
   - **Output-parameterization options to pick from** (start smallest, grow only if the gate fails):
     - **(a) VeRA-style** — frozen random A shared across tasks, hypernetwork emits only the small B / per-layer scaling vectors (smallest output; §2.4).
     - **(b) Low-rank factored heads** — emit A and B but from a low-rank head (fewer hypernetwork params than dense heads).
     - **(c) Full A/B** — emit both matrices densely (largest; the T2L default).
     Commit to the smallest that the plumbing supports; note the fallback ladder.
   - Match the library's LoRA shape exactly (rank 16, q/k/v_proj) so generated and library/oracle adapters are directly comparable in Phase 3.
3. **Definition of done:** `model.py` generates a full Mistral-7B-shaped adapter from one description in a single forward; parameter count of the hypernetwork is reported; the output parameterization is committed in `configs/phase2/` with the fallback ladder documented.
4. **Required testing:** generated adapter loads + applies onto Mistral-7B via the Phase-1 `apply` path and produces finite logits; two different descriptions produce *different* adapters (conditioning actually conditions); hypernetwork param count is within the planned budget.

### Sprint 3 — Reconstruction warmup (cheap init)  *(needs S2; parallel-dev with S4 design)*

1. **Goal:** Pre-train the hypernetwork by **reconstruction** (Eq. 6) against the train-split library LoRAs — a cheap, no-base-forward warmup that gets it producing sane adapters before the expensive SFT pass.
2. **Requirements:**
   - `recon.py`: L1 (or smooth-L1) between generated ΔW and each train task's library LoRA ΔW, sampled across tasks; seeded.
   - Train on the **train split only**; never touch held-out targets.
   - Track reconstruction loss + a sanity eval: do reconstructed adapters *roughly* reproduce a few train tasks' behavior?
3. **Definition of done:** reconstruction loss converges; the warmup checkpoint, applied to a handful of **train** tasks, lifts those tasks above base (a sanity sign the generator works) — explicitly *not* a held-out claim.
4. **Required testing:** held-out targets are never loaded during reconstruction (leakage guard); reconstruction loss decreases; a reconstructed train adapter beats base on its own train task (sanity, not the gate).

### Sprint 4 — SFT meta-training on Mistral-7B (the real objective)  *(needs S2/S3; the VRAM-critical phase)*

1. **Goal:** Meta-train the hypernetwork with **SFT** (Eq. 5) — generate ΔW from a task description, apply to the **frozen 4-bit** Mistral-7B, run the task batch, and **backprop the task loss through the base into the hypernetwork** — over the train-split tasks.
2. **Requirements:**
   - `sft.py` + `meta_train.py`: per step, sample a train task → embed its description → generate ΔW → apply → forward a task batch → cross-entropy (prompt-masked, reuse `data/sni`) → backward into the hypernetwork only (base frozen).
   - **Memory levers (notes.md §B):** 4-bit NF4 base, **gradient checkpointing** on the base forward, bf16, 8-bit/paged AdamW for the hypernetwork; **activation offload** only if it still OOMs. Profile peak VRAM per phase; record the trace (reuse `utils/vram`).
   - Warm-start from the S3 reconstruction checkpoint.
   - W&B + VRAM/RAM traces per run; seeded; checkpoint the hypernetwork.
3. **Definition of done:** SFT meta-training runs end-to-end on Mistral-7B within 32 GB; loss decreases; a hypernetwork checkpoint is saved that generates adapters for held-out descriptions; the backprop-through-base memory profile is captured.
4. **Required testing:** peak VRAM ≤ 32 GB asserted; grads reach only the hypernetwork (base stays frozen — assert base params have no grad); loss decreases over the train tasks; generated adapter for a *train* task beats base (in-distribution sanity before the held-out gate).

### Sprint 5 — Baselines, held-out eval & the gate  *(needs S4; the critical gate)*

1. **Goal:** Evaluate generated adapters on the **held-out** tasks against three baselines and render the gate verdict.
2. **Requirements:**
   - **Three baselines (§A.10):** (a) **trained-LoRA upper bound** = the Phase-1 **oracle** LoRA for each held-out task; (b) **base lower bound** = bare Mistral-7B; (c) **nearest-neighbor retrieval** = embed the held-out description, retrieve the closest **train**-split task's library LoRA by description embedding (`retrieval.py`), apply it. Same eval harness, same metric, same split as Phase 1.
   - For each held-out task: score generated-LoRA, oracle, base, retrieval on the held-out test split; tabulate (table **T1**).
   - **The gate:** generated mean score **> retrieval** mean score across the 30 held-out tasks (and clearly > base). Report per-task (T1), **per generalization axis** (format / language / domain — table **T2**, figure **F5**), and aggregate. Emit the **generalization curve** (F1, score vs distance-to-nearest-train) and the **generated-vs-retrieval scatter** (F2) as the headline evidence.
   - **Failure-mode analysis (§A.12):** where generated loses, is it bad task *identification* (retrieval would've picked a better adapter → an encoder/conditioning problem) or bad *execution* (right intent, weak weights → an output-parameterization/training problem)? This decides what to fix.
3. **Definition of done:** the four-way held-out table is committed; the gate verdict (pass/fail) is stated with the margin; if it fails, the failure-mode analysis points to the specific fix (per notes.md: smaller VeRA target §2.4, DoRA §2.3, or better distillation) **before** any Phase-3 work.
4. **Required testing:** all four conditions eval'd on the *identical* held-out split (same split hash as Phase 1); retrieval never retrieves a held-out task (train-split index only); generated and oracle adapters share the exact LoRA shape (Phase-3 comparability); the gate comparison is computed over the same task set for all conditions.

### Sprint 6 — Findings, ablations entry & artifact  *(needs S5)*

1. **Goal:** Write up Phase 2 and stage the ablation knobs Phase 4 will sweep.
2. **Requirements:**
   - `docs/phase-2-findings.md`: architecture, output parameterization (and why), the SFT memory profile, and the full output set — tables **T1–T4** and figures **F1–F5** (see the Phase-2 run spec §5) — plus the failure-mode verdict and the recommendation.
   - Record the cheap ablation axes for later (rank, #train tasks, seed) — wire them as config knobs now even if not swept until Phase 4.
   - Versioned hypernetwork checkpoint + the exact configs to reproduce it; W&B report (best-effort).
3. **Definition of done:** findings committed; gate verdict + reproducibility recipe documented; ablation knobs exposed in `configs/phase2/`.
4. **Required testing:** the committed SFT config reproduces the hypernetwork checkpoint's held-out table within tolerance; table schema validation (no empty cells; every held-out task has all four conditions).

---

## Parallelism map

```
        ┌─────────────────────────────────────────────┐
        │ S1 — Tiny-model plumbing (generate→apply→bp) │  (BLOCKER, GPU-light)
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────┐
        │ S2 — Architecture + output parameterization  │
        └───────────────────────┬─────────────────────┘
                ┌───────────────┴───────────────┐
                ▼                                ▼
   ┌────────────────────────┐      ┌──────────────────────────┐
   │ S3 Reconstruction warm │      │ (S4 design / mem pre-flight│  ← parallel DEV
   │ (cheap, no base fwd)   │      │  on a single batch)        │
   └───────────┬────────────┘      └─────────────┬────────────┘
               └──────────────┬─────────────────┘
                              ▼
        ┌─────────────────────────────────────────────┐
        │ S4 — SFT meta-training on Mistral-7B (VRAM)  │  ← needs the freed GPU
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────┐
        │ S5 — Baselines + held-out eval + THE GATE    │
        └───────────────────────┬─────────────────────┘
                                ▼
        ┌─────────────────────────────────────────────┐
        │ S6 — Findings + ablation knobs + artifact    │
        └─────────────────────────────────────────────┘
```

- **S1 is the blocker** and is GPU-light (tiny model) — it can be built while Phase 1's gate still occupies the GPU.
- **S3/S4 design** proceed in parallel after S2; the **S4 SFT run** needs the GPU free (after the Phase-1 gate) and is the VRAM-critical, design-sensitive step — **launched only after a design review**, not autonomously.
- **Single-GPU caveat:** Phase-1 gate, S3 reconstruction, and S4 SFT all serialize on the one GPU.

---

## Run scope & failure fallback

- **S1–S3 are GPU-light or GPU-free** and can be developed/tested while the Phase-1 gate runs (tiny model, reconstruction has no base forward).
- **S4 (SFT on Mistral-7B) is the expensive, design-sensitive step.** It waits for (a) the GPU to free after the Phase-1 gate and (b) a review of the S2 output-parameterization choice — a wrong choice wastes a full training run. Stage it behind a committed config so it launches with one command.
- **If the gate (S5) fails:** that is a *result*, not a failure of the phase — the failure-mode analysis (identification vs execution) dictates the fix (smaller target / DoRA / better distillation) and we iterate S2/S4 before touching Phase 3. Per notes.md, **cut scope (fewer tasks, one rank) before cutting the gate.**

---

## Phase 2 exit gate

Carried from [`../notes.md`](../notes.md) §C2 Phase 2:

> **Gate (the critical one):** generated LoRAs beat the *retrieval* baseline on held-out tasks. If they don't, the hypernetwork isn't really generalizing — stop and fix the output parameterization before touching SAEs.

Concretely, the gate clears when: **(1)** the SFT-trained hypernetwork generates a Mistral-7B adapter for each held-out task from its description alone; **(2)** the four-way held-out table (generated / oracle / base / retrieval) is committed; **(3)** generated **> retrieval** (and ≫ base) in aggregate across held-out tasks, on the Phase-1-locked split; and **(4)** the findings note states the margin, the memory profile of the backprop-through-base path, and (if relevant) the failure-mode fix. Clearing this gate means the generated adapters are *real, generalizing* adapters — the precondition for the Phase-3 interpretability comparison to be meaningful.
