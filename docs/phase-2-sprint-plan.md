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
3. `docs/phase-2-findings.md` — the architecture, the SFT recipe, the memory profile of the backprop-through-base path, the gate result, the **SFT-vs-reconstruction comparison** (T5 / F6 — the empirical case for choosing the SFT objective), and the failure-mode analysis (§A.12: bad task *identification* vs bad task *execution*).

---

## Hard constraints (these drive every design choice)

- **Single GPU: 32 GB VRAM.** Unlike Phase 1 (inference-only), Phase 2 **trains**, and the memory-critical step is **backprop through the frozen quantized base into the hypernetwork** (notes.md §B). The base is a **frozen 4-bit NF4 feature extractor** (QLoRA-style), so the situation is close to QLoRA, not full pretraining — but the activation/gradient path through a 7B still needs **gradient checkpointing** and possibly **activation offload**. Profile this first; it's where we OOM.
- **Output dimensionality dominates trainability *and* memory (§A.2).** A Mistral-7B LoRA over q/k/v at rank 16 across 32 layers is ~9.4 M parameters (from Phase 1's oracle: `trainable% = 0.13`). A hypernetwork that emits *all* of that per task is a large output space. Start with the **smallest viable target** and grow only if the gate needs it (see the output-parameterization decision in S2).
- **The held-out split is frozen once re-locked.** The original 9-task pilot split is replaced by the curated **~400 train / 10 val / 30 held-out** spec (see the Phase-2 run spec above); re-locking re-runs `make_split` → a new `lock_hash` in `heldout_split.yaml`, which is then the contract. Train **only** on the train-split tasks; the held-out tasks (and their descriptions) are never seen during meta-training. Leakage invalidates the gate.
- **Library quality is assumed, not re-litigated.** Phase 2 distills the **gate-passing** Phase-1 library only (quarantined adapters like task639 are excluded by construction). Garbage-in is already filtered.
- **Plumbing before scale (notes.md §B, §A — the single most repeated lesson).** Get the generate→apply→backprop loop green on a **tiny** model (SmolLM2-135M / Qwen-0.5B) with **3 toy tasks** before touching Mistral-7B. A working loop on a small model de-risks the whole phase.
- **Determinism / W&B.** Seeded; **every run (S1–S6) logs everything to W&B** — best-effort/non-blocking with offline fallback (project working pref), reusing `train/run_logger.py`. The full logged-fields contract is the [W&B experiment tracking](#wb-experiment-tracking--log-everything) subsection below; nothing stays stdout-only.

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

### W&B experiment tracking — log everything

**Contract: every Phase-2 run, from the S1 tiny-plumbing smoke to the S5 gate, is a W&B run.** All logging goes through `RunLogger` (`train/run_logger.py`) — best-effort / non-blocking, offline fallback, mirrored to a local `metrics.jsonl` so nothing is lost if W&B is unreachable. One W&B run per launch; runs are **grouped by sprint/stage** and **tagged** with the config name, objective (`reconstruction` / `sft`), base model, and the split `lock_hash`. No metric is left to stdout-only.

**Logged on every run:**
- **Config snapshot** — the full resolved `HyperConfig` + YAML: seed, rank, target modules, output parameterization (VeRA / low-rank / full), steps, lr, batch, base model, `lock_hash`, `wandb_project`/`wandb_mode`.
- **Per-step scalars** (every step, stage-tagged) — `recon/loss` (S3) or `sft/loss` (S4), learning rate, grad norm, step time (s/step), and examples-per-second throughput.
- **Memory traces** (`utils/vram`) — peak & current VRAM and RAM, logged per eval interval and at peak; this is the series behind figure **F4** (peak-VRAM-per-phase).
- **Validation** — val-split metric at each checkpoint-selection interval + the early-stop signal.
- **Run summary** — final losses, hypernetwork param count, peak VRAM, wall-clock, and exit status as W&B summary fields.

**Logged at the eval/gate stage (S5):**
- Per-task four-way scores (generated / oracle / base / retrieval) **plus the reconstruction-checkpoint scores (T5)**, per-axis margins, and the **gate verdict** as summary fields.
- All tables **T1–T6** and figures **F1–F6** uploaded as W&B Tables / media (in addition to `results/phase2/`).

**Artifacts (W&B Artifacts, versioned):**
- Hypernetwork checkpoints (S3 warmup, S4 final), each linked to its producing config + run.
- The locked split (`heldout_split.yaml` + `lock_hash`) and the eval tables (`{csv,parquet,md}`).

**S6** consolidates the above into a W&B report (best-effort), linked from `docs/phase-2-findings.md`.

**Human verification handoff (every sprint that logs to W&B).** Silent logging breakage is exactly what this guards against, so the automated asserts are not sufficient on their own: as soon as a sprint's W&B logging is wired, **Claude notifies the user** (push notification) to open the W&B website (wandb.ai → `lora-lab-phase2`) and confirm the run, its metrics/curves, and any artifacts actually appear. The sprint's W&B portion of the Definition of done is checked off **only after the user confirms on the website** — not on the local `metrics.jsonl` / automated test alone.

---

## Phase-2 run spec — data scale, split, training, and outputs

> Supersedes the original **pilot** split (9 train / 1 val / 3 held-out, built from
> the 14 `pilot: true` tasks). See [`phase-2-hypernet-sizing-options.md`](./phase-2-hypernet-sizing-options.md)
> for the full rationale and the held-out-axis comparison table.

### 1. How much data are we training on?

| Split | Tasks (count) | Notes |
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

| Axis | Held-out picks | # tasks | Why it proves generalization |
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

| Stage | Steps (count) | Est. wall-clock (min / hr) | Notes |
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
- **T5 — SFT-vs-reconstruction-vs-base held-out comparison**: per-held-out-task score for the **SFT-trained** hypernetwork vs the **reconstruction-trained (S3 warmup)** hypernetwork vs the **base** lower bound (oracle + retrieval as reference columns); margins SFT−reconstruction and SFT−base; aggregate + per-axis means. The project's *direct empirical test* of the SFT-over-reconstruction thesis (notes.md §C2; T2L §5.4 / Table 6) on **our** library — not just the asserted claim. `{csv,parquet,md}`.
- **T6 — Cost comparison across the four methods**: **training time · trainable parameters · final training loss** for the four adapter methods of the SFT-vs-reconstruction comparison (SFT-generated / reconstruction-generated / oracle / base). Surfaces the cost story behind the thesis — both hypernetwork variants share one parameter footprint and emit an adapter in a single forward pass (differing only in objective + training time), while the oracle pays a fresh per-task LoRA run and base trains nothing. Losses are each method's *own* objective (SFT/oracle = token CE, reconstruction = L1 on ΔW) → not directly comparable in absolute terms. `{csv,parquet,md}`.

**Figures**
- **F1 — Generalization curve** (headline): score vs description-embedding distance to nearest train task, one line each for generated / retrieval / oracle / base. Generalization shows as generated staying high while retrieval decays with distance.
- **F2 — Generated-vs-retrieval scatter**: one point per held-out task; above the diagonal = a win. The visual gate.
- **F3 — Training curves**: recon + SFT loss vs step (W&B).
- **F4 — Peak-VRAM-per-phase bar**: the backprop-through-base memory profile (the 32 GB constraint).
- **F5 — Per-axis margin bars**: mean margin vs retrieval and vs base, per generalization axis.
- **F6 — SFT vs reconstruction vs base** (the thesis figure): paired bars of mean held-out score for SFT-generated / reconstruction-generated / base, per generalization axis + aggregate — the visual of "SFT generalizes, reconstruction doesn't" — plus an SFT-vs-reconstruction per-task scatter (SFT on y, reconstruction on x; above the diagonal = SFT wins).

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

Each sprint lists: **(1) Goal · (2) Requirements · (3) Definition of done · (4) Required testing**, plus a **Time estimate** (dev effort + run wall-clock) at the top. The estimates sum to the ~2-week Phase-2 budget (notes.md §C2) and are planning guesses, not commitments.

**Version control (every sprint).** Each sprint ends with a **commit + push to the Phase-2 branch** (`phase_2`) once its Definition of done and Required testing are green — one self-contained, descriptively-messaged commit per sprint. If we then find and fix bugs while debugging *after* a sprint is marked done, that fix is a **separate second commit + push** (sprint-completion commit first, debugging-fix commit second) — never amend or fold the fix into the sprint commit, so the history shows the sprint and its follow-up fixes distinctly. So a sprint that needs post-completion debugging produces **two commits and two pushes**.

> **Pre-flight (before any Mistral-7B meta-training run):** Phase-1 gate complete and the train/held-out split finalized; the tiny-model plumbing loop (S1) green; a 4-bit Mistral base loads and a *hand-made* LoRA backprops a task loss into a dummy parameter without OOM (the S4 memory check, run on a single batch first).

### Sprint 1 — Tiny-model plumbing: generate → apply → backprop  *(BLOCKER — must finish first)*

**Time estimate:** ~2 days (dev/plumbing; GPU-light — each loop run finishes in minutes).

1. **Goal:** A working end-to-end loop on a **tiny** base (SmolLM2-135M / Qwen2.5-0.5B) and **3 toy SNI tasks**: a stub hypernetwork emits a LoRA from a task embedding, the LoRA is applied to the base, a forward runs, and a loss backpropagates **into the hypernetwork**. Plumbing only — correctness of the wiring, not quality.
2. **Requirements:**
   - `apply.py`: take generated A/B tensors and inject them as a live LoRA on the base (PEFT custom adapter or manual hooks), so the base forward uses ΔW and grads flow back to A/B.
   - A stub `model.py` producing correctly-shaped A/B for every target (layer, module).
   - Run both objectives once: **reconstruction** (regress onto a toy target LoRA — no base forward) and a **one-batch SFT** (task loss through the base). Assert grads reach the hypernetwork params.
   - Log VRAM-per-phase (§B) even at tiny scale, to validate the tracer on this path.
   - **W&B:** wire `RunLogger` into the loop here (per the [tracking contract](#wb-experiment-tracking--log-everything)) — per-step loss for both objectives + VRAM trace + config snapshot — so logging is proven on the tiny model before Mistral scale (`wandb_mode: disabled` is fine for the CPU smoke, but the calls must fire).
3. **Definition of done:** one full generate→apply→forward→backprop step completes on the tiny base for both objectives; the hypernetwork's parameters receive non-zero gradients; shapes match the base's LoRA target modules exactly; **the loop logs through `RunLogger` — `metrics.jsonl` is written with per-step loss for both objectives plus a VRAM trace and the config snapshot** (the tracking contract is proven here before scale). **Claude then notifies the user to confirm these appear on the W&B website** (verification handoff).
4. **Required testing:** assert generated A/B shapes match `(r × in)` / `(out × r)` for every targeted projection; assert `hypernet.grad` is non-None and finite after `backward()`; a no-op adapter (A·B = 0 at init) leaves base logits unchanged (apply-correctness); reconstruction loss decreases on a single overfit toy target; **assert `RunLogger.log_metrics` fired and `metrics.jsonl` contains ≥1 per-step record per objective with `loss` + VRAM keys; a `wandb_mode: disabled` run still writes the local `metrics.jsonl` (offline-fallback path).**

### Sprint 2 — Hypernetwork architecture + output parameterization  *(needs S1; the design decision)*

**Time estimate:** ~2 days (architecture design + implementation; validated with a single forward, GPU-light).

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
   - **W&B:** log the committed architecture config + hypernetwork param count per parameterization (the measured inputs to table **T3**) as a W&B run/summary, so the size-vs-memory choice is tracked, not just noted in the doc.
3. **Definition of done:** `model.py` generates a full Mistral-7B-shaped adapter from one description in a single forward; parameter count of the hypernetwork is reported; the output parameterization is committed in `configs/phase2/` with the fallback ladder documented; **the param count + chosen parameterization (the T3 inputs) are logged to W&B as run/summary fields.** **Claude then notifies the user to confirm on the W&B website.**
4. **Required testing:** generated adapter loads + applies onto Mistral-7B via the Phase-1 `apply` path and produces finite logits; two different descriptions produce *different* adapters (conditioning actually conditions); hypernetwork param count is within the planned budget; **assert the architecture config + param count are recorded to the run summary / `metrics.jsonl`.**

### Sprint 3 — Reconstruction warmup (cheap init)  *(needs S2; parallel-dev with S4 design)*

**Time estimate:** ~1.5 days dev + **~20–30 min/training run** (§4 — no base forward, so cheap).

1. **Goal:** Pre-train the hypernetwork by **reconstruction** (Eq. 6) against the train-split library LoRAs — a cheap, no-base-forward warmup that gets it producing sane adapters before the expensive SFT pass.
2. **Requirements:**
   - `recon.py`: L1 (or smooth-L1) between generated ΔW and each train task's library LoRA ΔW, sampled across tasks; seeded.
   - Train on the **train split only**; never touch held-out targets.
   - Track reconstruction loss + a sanity eval: do reconstructed adapters *roughly* reproduce a few train tasks' behavior?
   - **Preserve this checkpoint for the S5 thesis test:** the same reconstruction checkpoint is later evaluated head-to-head on **held-out** vs the SFT checkpoint (table **T5**, figure **F6**). S3 itself never touches held-out — that eval happens in S5 — so keep it as a versioned artifact.
   - **W&B:** per-step `recon/loss`, VRAM trace, config snapshot, and the warmup checkpoint logged as a versioned W&B Artifact (per the [tracking contract](#wb-experiment-tracking--log-everything)) — this is the S4 warm-start, so it must be reproducibly tracked.
3. **Definition of done:** reconstruction loss converges; the warmup checkpoint, applied to a handful of **train** tasks, lifts those tasks above base (a sanity sign the generator works) — explicitly *not* a held-out claim; **per-step `recon/loss` + VRAM trace are in W&B and the warmup checkpoint is logged as a versioned W&B Artifact** (it is the S4 warm-start). **Claude then notifies the user to confirm the run + artifact on the W&B website.**
4. **Required testing:** held-out targets are never loaded during reconstruction (leakage guard); reconstruction loss decreases; a reconstructed train adapter beats base on its own train task (sanity, not the gate); **assert the `recon/loss` series is logged and the checkpoint Artifact exists with its producing config attached; an offline run still writes `metrics.jsonl`.**

### Sprint 4 — SFT meta-training on Mistral-7B (the real objective)  *(needs S2/S3; the VRAM-critical phase)*

**Time estimate:** ~3 days dev + memory-profiling + **~5–7.5 hr/training run** (§4); the long pole of Phase 2.

1. **Goal:** Meta-train the hypernetwork with **SFT** (Eq. 5) — generate ΔW from a task description, apply to the **frozen 4-bit** Mistral-7B, run the task batch, and **backprop the task loss through the base into the hypernetwork** — over the train-split tasks.
2. **Requirements:**
   - `sft.py` + `meta_train.py`: per step, sample a train task → embed its description → generate ΔW → apply → forward a task batch → cross-entropy (prompt-masked, reuse `data/sni`) → backward into the hypernetwork only (base frozen).
   - **Memory levers (notes.md §B):** 4-bit NF4 base, **gradient checkpointing** on the base forward, bf16, 8-bit/paged AdamW for the hypernetwork; **activation offload** only if it still OOMs. Profile peak VRAM per phase; record the trace (reuse `utils/vram`).
   - Warm-start from the S3 reconstruction checkpoint.
   - W&B + VRAM/RAM traces per run; seeded; checkpoint the hypernetwork.
3. **Definition of done:** SFT meta-training runs end-to-end on Mistral-7B within 32 GB; loss decreases; a hypernetwork checkpoint is saved that generates adapters for held-out descriptions; the backprop-through-base memory profile is captured; **per-step `sft/loss`, the per-phase peak-VRAM trace (F4 series), and the final checkpoint Artifact are in W&B, and the run summary (final loss, hypernet param count, peak VRAM, wall-clock, exit status) is set.** **Claude then notifies the user to confirm the curves, VRAM trace + checkpoint artifact on the W&B website.**
4. **Required testing:** peak VRAM ≤ 32 GB asserted; grads reach only the hypernetwork (base stays frozen — assert base params have no grad); loss decreases over the train tasks; generated adapter for a *train* task beats base (in-distribution sanity before the held-out gate); **assert the `sft/loss` series + VRAM trace + run-summary fields are logged and the final-checkpoint Artifact exists with its config linked.**

### Sprint 5 — Baselines, held-out eval & the gate  *(needs S4; the critical gate)*

**Time estimate:** ~1.5 days dev + **~1–2 hr/eval run** (§4 — 30 tasks × 4 conditions).

1. **Goal:** Evaluate generated adapters on the **held-out** tasks against three baselines, render the gate verdict, **and run the head-to-head SFT-vs-reconstruction comparison** that empirically justifies the project's SFT-over-reconstruction choice.
2. **Requirements:**
   - **Three baselines (§A.10):** (a) **trained-LoRA upper bound** = the Phase-1 **oracle** LoRA for each held-out task; (b) **base lower bound** = bare Mistral-7B; (c) **nearest-neighbor retrieval** = embed the held-out description, retrieve the closest **train**-split task's library LoRA by description embedding (`retrieval.py`), apply it. Same eval harness, same metric, same split as Phase 1.
   - For each held-out task: score generated-LoRA, oracle, base, retrieval on the held-out test split; tabulate (table **T1**).
   - **The gate:** generated mean score **> retrieval** mean score across the 30 held-out tasks (and clearly > base). Report per-task (T1), **per generalization axis** (format / language / domain — table **T2**, figure **F5**), and aggregate. Emit the **generalization curve** (F1, score vs distance-to-nearest-train) and the **generated-vs-retrieval scatter** (F2) as the headline evidence.
   - **Reconstruction-vs-SFT comparison (the thesis test).** Also load the **S3 reconstruction-warmup checkpoint** and evaluate it on the *same* held-out tasks (eval-only — no training, so the leakage contract is intact) head-to-head against the SFT hypernetwork and the base lower bound → table **T5**, figure **F6**. This is the empirical evidence for the project-defining "SFT, not reconstruction" decision (notes.md §C2; T2L §5.4 / Table 6): we expect reconstruction to sit near retrieval/base on held-out while SFT pulls clear. Report the SFT−reconstruction margin per task, per axis, and aggregate.
   - **Cost comparison table (T6).** Assemble a **training-time / trainable-params / final-loss** table across these same four methods (SFT-generated, reconstruction-generated, oracle, base) — pulling training time + final loss from the S3/S4 W&B runs, param counts from S2/S4, and the oracle's per-task cost from Phase 1. Base trains nothing ("—"); flag that the losses are different objectives (SFT/oracle token CE vs reconstruction L1 on ΔW) and **not directly comparable** in absolute terms. This quantifies the amortization argument: one hypernetwork run serves all tasks vs a fresh oracle LoRA per task.
   - **Failure-mode analysis (§A.12):** where generated loses, is it bad task *identification* (retrieval would've picked a better adapter → an encoder/conditioning problem) or bad *execution* (right intent, weak weights → an output-parameterization/training problem)? This decides what to fix.
   - **W&B:** log the four-way per-task scores, per-axis margins, and the **gate verdict** as W&B summary fields, and upload tables **T1–T6** + figures **F1–F6** as W&B Tables / media (per the [tracking contract](#wb-experiment-tracking--log-everything)) — the gate result **and the SFT-vs-reconstruction comparison** live in W&B, not only in the findings doc.
3. **Definition of done:** the four-way held-out table is committed; the gate verdict (pass/fail) is stated with the margin; if it fails, the failure-mode analysis points to the specific fix (per notes.md: smaller VeRA target §2.4, DoRA §2.3, or better distillation) **before** any Phase-3 work; **table T5 + figure F6 (SFT vs reconstruction vs base) and the T6 cost comparison (training time / params / loss across the four methods) are committed, with the SFT−reconstruction margin stated; the four-way per-task scores, per-axis margins, and the gate verdict are logged as W&B summary fields, and tables T1–T6 + figures F1–F6 are uploaded as W&B Tables / media.** **Claude then notifies the user to confirm the tables/figures + gate verdict on the W&B website.**
4. **Required testing:** all four conditions eval'd on the *identical* held-out split (same split hash as Phase 1); retrieval never retrieves a held-out task (train-split index only); generated and oracle adapters share the exact LoRA shape (Phase-3 comparability); the gate comparison is computed over the same task set for all conditions; **the reconstruction (S3) checkpoint is eval'd on the identical held-out split as the SFT checkpoint (same metric + harness), and T5 has all of {SFT, reconstruction, base} for every held-out task (no missing cell); **T6 has training-time + params + loss for all four methods (base's training cells are "—", not blank);** assert the gate verdict + all four condition scores are present in the W&B run summary, and every table (T1–T6) and figure (F1–F6) was uploaded (no missing artifact).**

### Sprint 6 — Findings, ablations entry & artifact  *(needs S5)*

**Time estimate:** ~1 day (write-up + artifact / W&B-report staging; no training).

1. **Goal:** Write up Phase 2 and stage the ablation knobs Phase 4 will sweep.
2. **Requirements:**
   - `docs/phase-2-findings.md`: architecture, output parameterization (and why), the SFT memory profile, and the full output set — tables **T1–T6** and figures **F1–F6** (see the Phase-2 run spec §5) — plus the failure-mode verdict and the recommendation.
   - Record the cheap ablation axes for later (rank, #train tasks, seed) — wire them as config knobs now even if not swept until Phase 4.
   - Versioned hypernetwork checkpoint + the exact configs to reproduce it; W&B report (best-effort).
3. **Definition of done:** findings committed; gate verdict + reproducibility recipe documented; ablation knobs exposed in `configs/phase2/`; **the W&B report is generated (best-effort) and linked from `docs/phase-2-findings.md`, referencing the S3/S4 runs + checkpoint Artifacts.** **Claude then notifies the user to confirm the report renders on the W&B website.**
4. **Required testing:** the committed SFT config reproduces the hypernetwork checkpoint's held-out table within tolerance; table schema validation (no empty cells; every held-out task has all four conditions); **assert the findings doc links to the W&B run(s)/report and that those references resolve to the S4/S5 runs + artifacts (skipped gracefully if `wandb_mode: offline`/`disabled`, with the local `metrics.jsonl` linked instead).**

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

---

## Proposed Figures & Tables

> ⚠️ **All numbers below are illustrative placeholders** — hand-picked synthetic values that show each artifact's **schema, columns, and expected shape**, not results. They encode the *expected story* (oracle ≥ SFT-generated > retrieval ≫ base; reconstruction stuck near base) so the layout reads correctly before real data exists. Scores are a held-out test metric (EM or ROUGE-L, 0–100 — abbreviated **pts** in column headers; Δ values are differences in pts), reusing the Phase-1 eval harness.

### T1 — Four-way held-out gate table *(primary)*

One row per held-out task; the gate is decided on the **aggregate** `Δ vs retrieval` row. `Δ vs retr` = generated − retrieval; `Δ vs base` = generated − base.

| Held-out task | Axis | Generated SFT (pts) | Oracle (pts) | Base (pts) | Retrieval (pts) | Δ vs retr (pts) | Δ vs base (pts) | Verdict |
|---|---|--:|--:|--:|--:|--:|--:|:--:|
| task893_quail_answer_gen | format | 63.2 | 70.1 | 39.8 | 45.0 | **+18.2** | +23.4 | ✅ |
| task1564_triviaqa_answer_gen | format | 58.7 | 66.4 | 41.2 | 47.9 | **+10.8** | +17.5 | ✅ |
| task645_en→es_translation | language | 51.3 | 59.8 | 33.1 | 40.2 | **+11.1** | +18.2 | ✅ |
| task1338_de→en_translation | language | 55.0 | 61.2 | 35.5 | 52.6 | **+2.4** | +19.5 | ✅ |
| task512_twitter_emotion | domain | 68.9 | 74.3 | 50.1 | 65.7 | **+3.2** | +18.8 | ✅ |
| task888_financial_phrasebank | domain | 60.4 | 72.0 | 48.6 | 63.1 | **−2.7** | +11.8 | ❌ |
| … *(24 more held-out tasks)* | … | … | … | … | … | … | … | … |
| **Aggregate (30 tasks)** | — | **63.8** | 70.9 | 42.4 | 55.2 | **+8.6** | +21.4 | ✅ **PASS** |

*Reading it:* generated clears retrieval overall (**+8.6**) → gate **passes**; the lone ❌ (financial_phrasebank) is where a strong same-domain retrieval competitor wins — exactly the kind of case the S5 failure-mode analysis dissects (identification vs execution).

### T2 — Per-axis gate summary

Aggregates T1 by generalization axis; the per-axis verdict surfaces *where* generalization holds.

| Axis | n (tasks) | Generated (pts) | Oracle (pts) | Base (pts) | Retrieval (pts) | Δ vs retr (pts) | Verdict |
|---|--:|--:|--:|--:|--:|--:|:--:|
| Format transfer ⭐ | 15 | 61.5 | 69.2 | 40.3 | 46.1 | **+15.4** | ✅ |
| Language transfer | 8 | 53.8 | 60.5 | 34.6 | 48.9 | **+4.9** | ✅ |
| Domain transfer | 7 | 64.2 | 73.1 | 49.0 | 64.0 | **+0.2** | ⚠️ marginal |
| **Aggregate** | 30 | **63.8** | 70.9 | 42.4 | 55.2 | **+8.6** | ✅ **PASS** |

*Reading it:* the win is biggest on **format transfer** (retrieval grabs the same-topic, wrong-format LoRA and fails); **domain transfer** is near-tied because retrieval's competitor is strongest there.

### T3 — Hypernet size & memory *(measured in S2/S4)*

| Output parameterization | Hypernet params (M) | Output / task (params) | Peak VRAM, S4 (GB) | Notes |
|---|--:|--:|--:|---|
| **VeRA-style** *(default)* | 4.8 | ~0.05 M (B + scales) | 24.6 | fits with headroom |
| Low-rank factored heads | 11.2 | full A/B (r=16) | 28.1 | fallback (b) |
| Full A/B (T2L-L) | 32.5 | full A/B dense | 31.4 | OOM-risk → throttle batch |

### T4 — Data-scale summary *(as run)*

| Split | Tasks (count) | Example-passes (count) | Steps (count) |
|---|--:|--:|--:|
| Train — SFT (S4) | ~400 | 24,000 | 6,000 |
| Train — reconstruction (S3) | ~400 | — (no base fwd) | 2,000 |
| Val | 10 | — | eval only |
| Held-out (gate) | 30 | — | eval only |

### T5 — SFT vs reconstruction vs base *(the thesis test)*

Same held-out tasks, three conditions side-by-side (oracle + retrieval as reference). `Δ SFT−recon` is the headline number: the value of the SFT objective over a pure reconstruction-trained generator.

| Held-out task | Axis | SFT (pts) | Recon (pts) | Base (pts) | Oracle (pts) | Retr (pts) | Δ SFT−recon (pts) | Δ SFT−base (pts) |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| task893_quail_answer_gen | format | 63.2 | 44.1 | 39.8 | 70.1 | 45.0 | **+19.1** | +23.4 |
| task645_en→es_translation | language | 51.3 | 38.0 | 33.1 | 59.8 | 40.2 | **+13.3** | +18.2 |
| task512_twitter_emotion | domain | 68.9 | 53.5 | 50.1 | 74.3 | 65.7 | **+15.4** | +18.8 |
| … *(27 more)* | … | … | … | … | … | … | … | … |
| **Aggregate (30 tasks)** | — | **63.8** | **47.6** | 42.4 | 70.9 | 55.2 | **+16.2** | +21.4 |

*Reading it:* reconstruction (47.6) lands just above **base** (42.4) and *below* **retrieval** (55.2) on held-out — it memorized weight-space copies and didn't generalize — while **SFT** (63.8) clears everything. That **+16.2** gap is the project's own evidence for "SFT, not reconstruction."

### T6 — Cost comparison across the four methods

Training cost & footprint for the four adapter methods in the SFT-vs-reconstruction comparison. Both hypernetwork rows share **one** 4.8 M-param model and emit an adapter in a single forward pass — only the **objective + training time** differ. The oracle pays a fresh 9.4 M-param LoRA run *per task*; base trains nothing.

| Method | Adapter produced via | Training time (wall-clock) | Trainable params (count) | Final training loss (native objective units) | Per-task inference (latency) |
|---|---|--:|--:|--:|:--|
| SFT-generated (hypernetwork) | 1 hypernet forward | ~5–7.5 hr · one-time, amortized over all tasks | 4.8 M (VeRA) | token CE ≈ 1.12 (`sft/loss`) | ~1 forward (ms) |
| Reconstruction-generated (hypernetwork) | 1 hypernet forward | ~20–30 min · one-time | 4.8 M (VeRA) | L1 on ΔW ≈ 2.4e-4 (`recon/loss`) | ~1 forward (ms) |
| Oracle (per-task LoRA) | train a LoRA per task | ~15 min/task × N tasks (Phase 1) | 9.4 M per adapter | token CE ≈ 0.78 | — (pre-trained; not zero-shot) |
| Base (no adapter) | — | — | 0 | — | — |

> **Loss caveat:** the loss column lists each method's *own* training objective — SFT and oracle minimize token cross-entropy, reconstruction minimizes L1 on the weight delta — so the absolute values are **not directly comparable**. The table's real point is the **training-time + parameter** asymmetry: one amortized 4.8 M hypernetwork run that generalizes, vs a 9.4 M oracle run *per task* that doesn't transfer.

---

### F1 — Generalization curve *(headline)*

Score vs description-embedding distance to nearest train task. Generalization = generated stays high while **retrieval decays** with distance.

```
score (pts, 0–100)
 75│ O    O    O    O    O    O          O = Oracle      (upper bound)
 65│ G    G    G    G    G    G          G = Generated    (SFT — holds up)
 55│ R    R    R                         R = Retrieval    (decays w/ distance)
 45│           ·    R    R    R
 42│ b    b    b    b    b    b          b = Base         (flat lower bound)
   └────┬────┬────┬────┬────┬────┬──► distance to nearest train task (cosine dist, 0–1)
       near                       far
```

### F2 — Generated-vs-retrieval scatter *(the visual gate)*

One point per held-out task. Above the diagonal = generated beats retrieval (a win). Points below = losses (e.g. financial_phrasebank).

```
generated (pts)
 70│                  •  •   ⟋
 65│              • • •    ⟋
 60│           •  • •   ⟋          ⟋ = y = x  (diagonal)
 55│        • • •    ⟋
 50│      • •     ⟋   ●  ← below diagonal = retrieval wins (1–2 tasks)
 45│          ⟋
   └────┬────┬────┬────┬────┬──► retrieval (pts)
       40   50   55   60   70
```

### F3 — Training curves *(W&B)*

Reconstruction warmup (S3) then SFT meta-train (S4, warm-started from it).

```
loss (objective units: CE in nats / L1)
 │•                         recon/loss (S3, 0–2k steps)
 │ ••                       sft/loss   (S4, 0–6k steps)
 │   •••___
 │         ••••____  ← warm-start hand-off (S3 ckpt → S4)
 │  sft:        •••••••__________
 │                            •••••••••______
 └────┬─────┬─────┬─────┬─────┬─────┬──► step (count)
      0    1k    2k    3k    4.5k   6k
```

### F4 — Peak-VRAM-per-phase bar *(the 32 GB constraint)*

Where the backprop-through-base path peaks; must stay under 32 GB.

```
phase   peak VRAM(GB)|0      8      16      24    32|
generate ΔW    9.2   ███▍
apply adapter  9.8   ███▋
base fwd+ckpt 18.4   ███████▎
backward      24.6   █████████▊            ◄ peak (headroom to 32)
optim step    21.0   ████████▍
                     └──────────────────────────── 32 GB cap
```

### F5 — Per-axis margin bars

Mean margin vs retrieval and vs base, per axis (positive vs retrieval = generalizing there).

```
axis        Δ vs retrieval (pts)    Δ vs base (pts)
Format      +15.4 ██████▏           +21.2 ████████▌
Language     +4.9 ██▏               +19.2 ███████▋
Domain       +0.2 ▏                 +15.2 ██████
Aggregate    +8.6 ███▌              +21.4 ████████▌
```

### F6 — SFT vs reconstruction vs base *(the thesis figure)*

Paired bars per axis + aggregate; the visual of "SFT generalizes, reconstruction doesn't."

```
axis        SFT (pts)       Recon (pts)     Base (pts)
Format     61.5 ██████▏     45.0 ████▌      40.3 ████
Language   53.8 █████▍      39.5 ███▉       34.6 ███▍
Domain     64.2 ██████▍     50.0 █████      49.0 ████▉
Aggregate  63.8 ██████▍     47.6 ████▊      42.4 ████▏
```

*Companion scatter (SFT y vs reconstruction x): every held-out task sits **above** the diagonal — SFT beats its reconstruction-only counterpart on all 30.*
