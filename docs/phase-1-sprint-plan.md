# Phase 1 — Sprint Plan: Build the LoRA Library (the hypernetwork's training data *and* the interp baseline)

*Sprint-planning material for Phase 1. Pairs with [`../notes.md`](../notes.md) §C2 (timeline — Phase 1), §C (the end goal), and §C1 (base-model decision). Phase 0 ([`phase-0-findings.md`](./phase-0-findings.md)) proved the toolchain; Phase 0.5 ([`phase-0.5-findings.md`](./phase-0.5-findings.md)) proved full-FT-of-7B is reachable here. Phase 1 builds the artifact every later phase consumes.*

---

## What we are trying to achieve in Phase 1

Assemble a **versioned, quality-gated library of per-task LoRA adapters** for `Mistral-7B-Instruct-v0.2`, each paired with its **natural-language task description**, and **lock a frozen held-out split** we will never train the hypernetwork on.

> **The artifact does double duty** (notes.md §C): the library is simultaneously **(a)** the *training data* for the Phase 2 hypernetwork (task description → LoRA), and **(b)** the *"hand-trained LoRA" comparison set* for the Phase 3 interpretability study. **Build it once, use it twice.** This is why the quality gate matters: a garbage library → a garbage hypernetwork → a meaningless interp comparison.

**The good news (notes.md §C1, Phase-1 source box):** this is largely a **download-and-organize** job, not a train-479-adapters-from-scratch job. The `Lots-of-LoRAs` release (Brüel-Gabrielsson et al. 2024, *"Compress then Serve"*, [arXiv:2407.00066](https://arxiv.org/abs/2407.00066)) ships **~1,268 ready-made rank-16 LoRA adapters trained for `Mistral-7B-Instruct-v0.2`** plus **~1,174 per-task datasets** — exactly our primary base. The matching natural-language **task descriptions** (the hypernetwork's conditioning input) live in the `SakanaAI/text-to-lora` repo's `tasks/` folder. So Phase 1's real work is *selection, verification, alignment, and versioning* — turning a pile of downloads into a trustworthy, reproducible library with a locked split — plus training our **own** oracle LoRAs for the handful of held-out tasks we want fully under our control.

**Headline deliverables:**
1. A **versioned library manifest** (`configs/phase1/library.yaml`) — one row per task: adapter source + hash, dataset source + split hashes, task description + hash, rank/target-modules, and split role (train / val / held-out).
2. A **per-task quality table** (`results/phase1/library_quality.{csv,parquet,md}`) — every adapter's eval vs. the base model on its own task's held-out test split, with the **margin** (adapter − base). This *is* the gate evidence.
3. A **frozen held-out split** (`configs/phase1/heldout_split.yaml`) — locked + hashed, with our **own-trained oracle LoRAs** for the held-out tasks (the Phase 3 baseline).
4. `docs/phase-1-findings.md` — what the library is, how many tasks cleared the gate, the split definition, and the reproducibility recipe.

**Comparability anchor:** follow T2L's task partition where practical — the **500-task English SNI subset** of Brüel-Gabrielsson et al. (**479 train / 11 val / 10 removed for contamination**), so our results stay comparable to the Text-to-LoRA paper (§1.1). Deviations from that partition are recorded explicitly in the manifest, never silent.

---

## Hard constraints (these drive every design choice)

- **Single GPU: 32 GB VRAM. System RAM: 96 GB.** The Phase-1 GPU cost is **inference-dominated** (eval each adapter vs. base) plus a *little* training (held-out oracle LoRAs). Mistral-7B + a rank-16 adapter is a LoRA/QLoRA-class footprint (~6–16 GB from Phase 0.5) — comfortable. Nothing here needs the full-FT routes.
- **Gated base model.** `Mistral-7B-Instruct-v0.2` is gated on HuggingFace — needs `HF_TOKEN` + an accepted license. This is the **most likely step-zero killer** (same lesson as Phase 0.5's pre-flight); verify with a real `from_pretrained`, not just "the token exists." See [`gated-models-setup.md`](./gated-models-setup.md).
- **Adapters must match the base exactly.** A `Lots-of-LoRAs` adapter is only valid on the *exact* base (`Mistral-7B-Instruct-v0.2`) and target-module set it was trained for. Loading a mismatched adapter "works" silently and produces garbage — so adapter↔base compatibility is an *asserted*, hashed invariant, not an assumption.
- **The held-out split is sacred.** Tasks in the held-out split are **never** seen by the Phase 2 hypernetwork during training. Once locked (hashed), the split is immutable for the rest of the project — leakage here invalidates every downstream gate. Adopt T2L's contamination removals so we don't re-import known leaks.
- **Disk.** ~1,268 adapters × rank-16 (~tens of MB each, hundreds of MB for the Mistral target set) + per-task datasets is **single-digit to low-tens of GB** — confirm headroom and decide retention (full mirror vs. cached subset) *before* the bulk pull, not after the disk fills.
- **Determinism / reproducibility.** Reuse the Phase-0 `data/sni.py` split-hashing discipline: every split, dataset, adapter, and description carries a content hash, so "the exact library we used" is reproducible and assert-able. No silently-empty or silently-substituted rows.

### What we reuse from Phases 0 / 0.5 (don't rebuild)

| Need | Reuse |
|---|---|
| SNI data loading, chat-templating, prompt-masking, split hashing, held-out eval view | `src/lora_lab/data/sni.py` |
| Held-out eval (exact-match / ROUGE-L), greedy generation | `src/lora_lab/eval/{evaluate,metrics}.py` |
| Build LoRA/QLoRA/full-FT for the oracle held-out adapters | `src/lora_lab/methods/build.py`, `src/lora_lab/train/trainer.py` |
| Results table + plots | `src/lora_lab/eval/{table,plot}.py` |
| W&B run logging (best-effort, non-blocking) | `src/lora_lab/train/run_logger.py` |
| Task manifest pattern (extend, don't replace) | `configs/tasks.yaml` → `configs/phase1/library.yaml` |

### Proposed repo layout (extends the Phase 0 / 0.5 layout)

```
configs/phase1/library.yaml          versioned library manifest (task → adapter, dataset, description, split role, hashes)
configs/phase1/heldout_split.yaml    the frozen train/val/held-out partition (locked + content-hashed)
src/lora_lab/library/
  acquire.py        pull adapters + per-task datasets (Lots-of-LoRAs) + descriptions (SakanaAI/text-to-lora)
  manifest.py       build/validate the versioned manifest; integrity hashes; coverage report
  adapters.py       load a Lots-of-LoRAs adapter onto Mistral-7B via PEFT; assert base/target-module match
  descriptions.py   align each task → its NL description (the hypernetwork conditioning input)
  oracle.py         train our own held-out oracle LoRAs (reuse methods/build + train/trainer)
  index.py          (forward-looking) task-description embedding index for Phase 2's retrieval baseline
results/phase1/library_quality.{csv,parquet,md}   per-task adapter-vs-base eval (the gate)
results/phase1/library/              downloaded adapters + datasets (or cache pointers, per retention decision)
docs/phase-1-findings.md             the library writeup + gate evidence + split definition
```

---

## Sprints

Each sprint lists: **(1) Goal/objective · (2) Requirements · (3) Definition of done · (4) Required testing.**

> **Pre-flight checklist (verify BEFORE the bulk download / eval run — this is where Phase 1 dies at step zero):**
> 1. **Gated-base access** — `HF_TOKEN` set *and* the `Mistral-7B-Instruct-v0.2` license accepted; verified with a real `from_pretrained` (per [`gated-models-setup.md`](./gated-models-setup.md)).
> 2. **`Lots-of-LoRAs` reachable** — at least one adapter repo + one per-task dataset download and load cleanly; confirm the adapter's `adapter_config.json` reports `base_model_name_or_path = Mistral-7B-Instruct-v0.2`, `r = 16`, and a known target-module set.
> 3. **`SakanaAI/text-to-lora` task descriptions present** — the `tasks/` folder is cloned/fetched and parseable; spot-check that a description exists for several library tasks.
> 4. **Disk headroom** — enough for the retention decision (full mirror vs. subset); decide *before* the pull.
> 5. **Per-task isolation** — the acquire/eval loop is wrapped so one bad task (missing adapter, broken dataset, OOM during eval) is logged `status=quarantined` + reason and the loop **moves on**, never aborting the whole batch.

### Sprint 1 — Define the task set & build the versioned manifest  *(BLOCKER — must finish first)*

1. **Goal:** A single source of truth for *which tasks are in the library* and *where every artifact for each comes from*, content-hashed so the exact library is reproducible.
2. **Requirements:**
   - Enumerate the **candidate task set**: start from T2L's 500-task English SNI subset (479/11/10) and intersect with what `Lots-of-LoRAs` actually ships (adapter *and* dataset present) and what `SakanaAI/text-to-lora` has a description for. Record the three-way coverage (have-adapter ∩ have-dataset ∩ have-description).
   - Write `configs/phase1/library.yaml` — one row per task: `hf_adapter_repo`, `hf_dataset_repo`, `description_source`, `rank`, `target_modules`, `split_role` (train/val/held-out, filled in S5), and `{adapter_hash, dataset_split_hashes, description_hash}` (filled as artifacts are fetched in S2–S3). Extend the `configs/tasks.yaml` schema; don't fork it.
   - A `manifest.py` that builds, validates, and **coverage-reports** the manifest (counts: total candidate, full-coverage, partial, dropped — with the reason per drop).
   - Decide + record the **retention policy** (mirror all adapters/datasets locally vs. cache-on-demand) given the disk headroom.
3. **Definition of done:** `library.yaml` lists every task that has full three-way coverage; every dropped/partial task is recorded with a reason (never silently absent); the coverage report prints counts that reconcile (candidate = full + partial + dropped); retention policy committed.
4. **Required testing:** manifest round-trips (load → validate → re-emit identical); coverage numbers reconcile; assert no duplicate task names; assert every `split_role` is one of {train, val, held-out, unassigned}; the candidate set's intersection with T2L's 500-task list is reported (how many of T2L's tasks we cover).

### Sprint 2 — Acquire & load adapters; assert base/target-module compatibility  *(needs S1)*

1. **Goal:** Prove a `Lots-of-LoRAs` adapter loads onto our exact Mistral-7B base via PEFT, applies, and generates — and that mismatches are *caught*, not silently tolerated.
2. **Requirements:**
   - `acquire.py` pulls adapters (per retention policy) and records each adapter's content hash into the manifest.
   - `adapters.py` loads an adapter onto `Mistral-7B-Instruct-v0.2` via PEFT, **asserts** `adapter_config.base_model_name_or_path` + `r` + `target_modules` match the expected base/rank/module set, and runs a generation smoke test.
   - Bulk-acquire across the manifest with **per-task isolation** (pre-flight item 5): a missing/broken adapter is quarantined with a reason, the loop continues.
3. **Definition of done:** every full-coverage task's adapter is fetched + hashed + loads cleanly onto Mistral-7B and generates; incompatible/broken adapters are `status=quarantined` with the mismatch recorded; manifest now carries every `adapter_hash`.
4. **Required testing:** a deliberately mismatched adapter (wrong base or wrong target modules) is **rejected** by the assertion (negative test — silent acceptance is the failure mode we're guarding against); a known-good adapter changes the model's output vs. the bare base on a probe prompt (proves it actually applied); adapter hash is stable across two fetches.

### Sprint 3 — Align task descriptions (the conditioning set)  *(needs S1; parallel-dev with S2)*

1. **Goal:** Every library task is paired with the **natural-language description** that will be the Phase-2 hypernetwork's input — with *complete* coverage, since a task with no description can't be a hypernetwork training example.
2. **Requirements:**
   - `descriptions.py` maps each library task → its description from `SakanaAI/text-to-lora`'s `tasks/` folder (and/or the SNI `Definition` field as a recorded fallback), hashes it into the manifest, and flags any task lacking a description.
   - Resolve mismatches (task-name normalization between `Lots-of-LoRAs` repo names and T2L's task ids) explicitly — a documented mapping, not a guess.
   - Emit a descriptions coverage report; tasks missing a description are dropped from the *trainable* set (recorded) rather than given a fabricated description.
3. **Definition of done:** every task in the trainable library has a hashed description recorded; the name-normalization map is committed; description-coverage gaps are listed with their resolution (mapped / fallback-used / dropped).
4. **Required testing:** spot-check N descriptions against the source repo (exact text match); assert no empty/placeholder descriptions in the trainable set; name-normalization map is bijective on the covered set (no two tasks collapse to one description by accident).

### Sprint 4 — Quality gate: every library LoRA beats the base on its own task  *(needs S2 + S3; the Phase-1 gate)*

1. **Goal:** The headline gate — prove each library adapter is *actually competent*, by eval'ing it against the bare base model on its own task's held-out test split, and **quarantine any adapter that doesn't clearly beat base.**
2. **Requirements:**
   - For each task, run the Phase-0 eval harness twice on the held-out test split: **adapter-on-base** vs. **bare base**, same greedy decoding, same metric (exact-match for classification, ROUGE-L for generation). Record `adapter_score`, `base_score`, and `margin = adapter − base`.
   - Define "clearly beats" as a committed threshold (e.g. margin ≥ τ on the task's metric, τ recorded in the findings) — and **quarantine** adapters below it (a valid negative result, not a silent pass).
   - Emit `results/phase1/library_quality.{csv,parquet,md}` (reuse `eval/table.py`); log runs to W&B best-effort/non-blocking.
   - **Scope control:** if eval'ing all full-coverage tasks is GPU-heavy, subsample the test split per task to a fixed, seeded N (recorded), and `log()` exactly what was capped — never silently truncate.
3. **Definition of done:** every full-coverage task has an adapter-vs-base row with a margin; adapters below threshold are `status=quarantined`; the quality table is committed; the count of gate-passing tasks is reported (this is the library size going into Phase 2).
4. **Required testing:** assert adapter and base were eval'd on the *identical* split (same split hash); assert the metric matches the task `kind`; sign sanity (a quarantined adapter genuinely scores ≤ base; a passing adapter genuinely scores > base by ≥ τ); a re-run of one task reproduces its margin within tolerance (greedy → deterministic).

### Sprint 5 — Lock the frozen held-out split & train the oracle held-out LoRAs  *(needs S4)*

1. **Goal:** Partition the gate-passing library into **train / val / held-out**, lock it immutably, and train **our own** oracle LoRAs for the held-out tasks — the adapters Phase 2 evaluates *against* and Phase 3 compares feature geometry *with*.
2. **Requirements:**
   - Define the partition following T2L (479 train / 11 val / 10 contamination-removed) where our coverage allows; record exactly which of our gate-passing tasks land in each bucket, and any deviation from T2L's list with a reason. Write `configs/phase1/heldout_split.yaml` and **content-hash the held-out task id set** (the lock).
   - **Apply T2L's contamination removals** so known-leaked tasks never enter train.
   - `oracle.py` trains a LoRA (rank 16, matching the library) on each **held-out** task using the Phase-0 training stack (`methods/build` + `train/trainer`), so the held-out comparison baseline is *ours*, fully reproducible and under our control (notes.md §C1: "train your own LoRAs only for the handful of held-out tasks you want fully under your control"). These also pass the same S4 quality gate.
   - Optionally stand up `index.py` — a task-description embedding index over the **train** split — so Phase 2's nearest-neighbor *retrieval* baseline (notes.md §C2 Phase 2 gate) has its data ready. (Forward-looking; can slip to Phase 2 if time-pressed.)
3. **Definition of done:** `heldout_split.yaml` is committed, hashed, and immutable; every held-out task has an **own-trained** oracle LoRA that clears the S4 gate; train/val/held-out buckets are disjoint and cover the gate-passing library; contamination removals applied + recorded.
4. **Required testing:** assert the three buckets are pairwise disjoint and their union = gate-passing library; assert **no held-out task id appears in the train manifest** (the leakage guard — the single most important Phase-1 test); the split hash is stable across re-emit; each oracle LoRA beats base by ≥ τ on its task (same gate as S4); contamination-removed tasks are absent from all three buckets.

### Sprint 6 — Library artifact, versioning & findings  *(needs S1–S5; the headline deliverable)*

1. **Goal:** Package the verified library into a single versioned, reproducible artifact and write the findings note — the thing Phase 2 and Phase 3 consume.
2. **Requirements:**
   - Finalize `configs/phase1/library.yaml` with every hash filled (adapter, dataset splits, description), every `split_role` assigned, and a top-level **library version + content hash** over the whole manifest.
   - Render a library summary: task counts per split, per-task quality margins (from S4), description coverage, and the quarantine list with reasons.
   - Write `docs/phase-1-findings.md`: what the library is, how many tasks cleared the gate (and which were quarantined + why), the frozen-split definition, the oracle-LoRA recipe, the τ threshold, and the **reproducibility recipe** (manifest version → re-fetch → re-verify).
   - Push a W&B report (best-effort, never a gate).
3. **Definition of done:** the versioned library artifact is committed with a top-level hash; the findings note states the gate-passing library size, the locked split, and the reproducibility recipe; every manifest row is either fully populated or explicitly quarantined (no silently-empty cells).
4. **Required testing:** re-fetch a sampled subset from the committed manifest and assert the artifact hashes match (the reproducibility claim); manifest schema validation (no empty required fields on non-quarantined rows); the library version hash changes iff a tracked artifact changes; findings counts reconcile with `library_quality.csv` and `heldout_split.yaml`.

---

## Parallelism map

```
        ┌─────────────────────────────────────────────────┐
        │ Sprint 1 — Task set + versioned manifest          │  (BLOCKER)
        │            + retention policy                     │
        └───────────────────────┬─────────────────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
   ┌────────────────────────┐        ┌──────────────────────────┐
   │ S2 Acquire + load       │        │ S3 Align task             │   ← parallel DEV
   │ adapters; assert base/  │        │ descriptions (conditioning│
   │ target-module match     │        │ set); name normalization  │
   └───────────┬─────────────┘        └─────────────┬────────────┘
               └──────────────────┬─────────────────┘
                                  ▼
                  ┌──────────────────────────────────────┐
                  │ S4 Quality gate — adapter beats base  │  ← the Phase-1 gate
                  │     on its own task (eval all)        │
                  └──────────────────┬───────────────────┘
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ S5 Lock frozen held-out split +        │
                  │     train OUR oracle held-out LoRAs    │
                  └──────────────────┬───────────────────┘
                                     ▼
                  ┌──────────────────────────────────────┐
                  │ S6 Versioned library artifact +        │
                  │     findings + reproducibility check   │
                  └──────────────────────────────────────┘
```

- **Sprint 1** is a hard blocker — nothing is fetched or eval'd until the manifest defines *what* the library is.
- **Sprints 2 & 3** are independent *development* tracks (adapters vs. descriptions) against the shared manifest.
- **Single-GPU caveat:** eval (S4) and oracle training (S5) **serialize** on the one GPU. S4's full adapter-vs-base sweep is the GPU-heaviest part — schedule it as the long unattended run; S2/S3 are mostly I/O + CPU.
- **Sprint 4** needs both adapters (S2) and descriptions are *not* strictly required for the eval itself, but the trainable library is defined by S2 ∩ S3, so run S3 first to avoid eval'ing tasks that'll be dropped for missing descriptions.

---

## Run scope & failure fallback (no fixed time-box)

**No time-box** — the work is mostly download + inference. The GPU cost is one eval pass per adapter (S4) + a small batch of oracle LoRA trainings (S5); both fit comfortably in an unattended window. So we **run everything**; nothing is dropped for time.

What this section governs is **resilience**:
- **Per-task isolation:** a missing adapter, broken dataset, absent description, or eval OOM logs `status=quarantined` + the reason and the loop **moves on** — one bad task never sinks the batch.
- **Order so the high-value parts finish first** (in case the run is cut short): manifest (S1) → descriptions (S3) → adapter load-smoke (S2) → quality gate (S4) → split-lock + oracles (S5). Even a truncated run yields a verified, partially-eval'd library that's honest and re-runnable from its committed manifest.
- **Anything incomplete** is marked `quarantined`/`unassigned` with a reason in the manifest, never silently blank — a partial library is still trustworthy and resumable.

**Minimum for the phase to count as done:** S1 (the manifest), S4 (the quality gate — proving the library is competent), S5 (the locked held-out split + own-trained oracles), and S6 (the versioned artifact + findings). A library without a locked, leakage-checked split cannot feed Phase 2.

---

## Phase 1 exit gate

Carried from [`../notes.md`](../notes.md) §C2 Phase 1:

> **Gate:** every library LoRA clearly beats the base model on its task, and the held-out split is locked. (Garbage library → garbage hypernetwork → meaningless interp comparison.)

Concretely, the gate clears when: **(1)** the versioned library manifest (`configs/phase1/library.yaml`) is committed with every task fully populated or explicitly quarantined; **(2)** the quality table (`results/phase1/library_quality.{csv,md}`) shows every non-quarantined adapter beats base by ≥ τ on its own task's held-out split; **(3)** the frozen held-out split (`configs/phase1/heldout_split.yaml`) is locked + content-hashed, disjoint from train, with contamination removals applied and **our own oracle LoRAs** trained for every held-out task; and **(4)** `docs/phase-1-findings.md` states the gate-passing library size, the split definition, the τ threshold, and the reproducibility recipe. Clearing this gate means Phase 2 has trustworthy training data (task description → competent LoRA) and a clean, leakage-free held-out set to generalize to.
