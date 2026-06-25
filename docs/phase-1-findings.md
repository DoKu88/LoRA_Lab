# Phase 1 — Findings: Building the Quality-Gated LoRA Library

*Live results log for Phase 1. Pairs with the sprint plan
([`phase-1-sprint-plan.md`](./phase-1-sprint-plan.md)) and `notes.md` §C2 (Phase 1).
The library does double duty — Phase-2 hypernetwork training data **and** the
Phase-3 hand-trained interp baseline (§C). Numbers below are from the **pilot
pass**; the full 1117-task sweep reuses the same manifest and is resumable (see
"Scope" at the end).*

**Hardware:** RTX 5090 (32 GB, sm_120) + 96 GB RAM, torch 2.10+cu128, conda env
`lora_lab`. Base: `Mistral-7B-Instruct-v0.2` (7.24 B). Source: `Lots-of-LoRAs`
(adapters + per-task SNI datasets) + the embedded SNI `Definition` (descriptions).

---

## Headline result

**The pipeline works end-to-end and the gate is meaningful.** On a 14-task pilot,
**13/14 library adapters clearly beat the bare base** on their own held-out split
(mean margin **+0.41** EM/ROUGE-L), and the **1 failure (task639) was caught and
quarantined** — a real negative result, not a silent pass. A frozen, leakage-checked
train/val/held-out split is locked, and **3 of our own rank-16 oracle LoRAs** were
trained for the held-out tasks (all clear the gate). Everything is versioned and
reproducible from a content-hashed manifest.

Three things the build surfaced that the plan only anticipated:
- **The join key is the task *number*, not the name.** Adapters are named
  `…-r16-task<NUM>`; datasets are `task<NUM>_<slug>`. We join on `<NUM>` and carry
  both (the S3 name-normalization, resolved cleanly: 1172 numbers intersect).
- **Descriptions come free from the data.** Every SNI example embeds the canonical
  task `Definition:` at the head of its `input` — so the hypernetwork's conditioning
  text is extractable programmatically at **100 % coverage**, no separate scrape.
- **The library is mixed-rank.** The repos are all named `r16`, but actual adapter
  rank is **16 *or* 43** (the rank-adaptive *Compress-then-Serve* variants). Rank is
  not a compatibility constraint — any LoRA rank applies to the same base — so we
  **record** it and assert only the load-critical invariants (base model + target
  modules). 9/14 pilot adapters are r43.

---

## Sprint 1–3 — Manifest, acquisition, descriptions (the substrate)

`configs/phase1/library.yaml` (version `fee497c5fa710c14`) is the source of truth.

| Coverage | Count |
|---|---|
| Candidate task numbers (Hub: `Lots-of-LoRAs` adapters ∪ datasets) | **1227** |
| **Full coverage** (adapter **and** dataset present) | **1116** |
| Quarantined — missing adapter | 55 |
| Quarantined — missing dataset | 55 |
| Quarantined — below-τ on the gate (task639) | 1 |
| Pilot subset (this pass) | 14 |

- **Adapter compatibility (S2)** is asserted on load: `base_model_name_or_path ==
  mistralai/Mistral-7B-Instruct-v0.2` and `target_modules == {q,k,v}_proj` (hard),
  rank recorded. A negative test confirms a wrong-base/wrong-target adapter is
  *rejected*, not silently accepted (`tests/test_phase1_library.py`).
- **Descriptions (S3)** are the SNI `Definition` span parsed from each task's
  `input` (`description_source: sni_definition`), hashed into the manifest. Pilot
  coverage: **13/13** trainable tasks (the quarantined task639 keeps its text but
  drops out of the trainable set).

Nothing is dropped silently — every partial/failed task carries a `status:
quarantined` + `reason` in the manifest.

---

## Sprint 4 — Quality gate: every library LoRA beats the base

**Protocol.** Per task, generate greedily on the held-out **test** split and score
adapter-on-base vs. bare base on the *same* split. Metric inferred per task
(short, low-cardinality gold ⇒ exact-match; else ROUGE-L). Efficiency: the 7B is
loaded **once**; adapters attach/detach on the resident model and the base score
comes from the *same* model under PEFT `disable_adapter()` — so base and adapter
differ by exactly the LoRA. τ ("clearly beats") = **margin ≥ 0.05**. 120 eval
examples/task. Source: `results/phase1/library_quality.{csv,parquet,md}`,
`results/phase1/gate_results.jsonl`.

| task | rank | metric | base | adapter | margin | gate |
|---|---|---|---|---|---|---|
| task280 stereoset_classification | 43 | EM | 0.075 | 0.983 | **+0.908** | ✅ |
| task1391 winogrande_easy | 43 | EM | 0.100 | 0.908 | **+0.808** | ✅ |
| task512 twitter_emotion | 16 | EM | 0.075 | 0.875 | +0.800 | ✅ |
| task190 snli_classification | 43 | EM | 0.350 | 0.892 | +0.542 | ✅ |
| task391 causal_relationship | 43 | EM | 0.317 | 0.833 | +0.517 | ✅ |
| task843 financial_phrasebank | 16 | EM | 0.400 | 0.908 | +0.508 | ✅ |
| task1344 glue_entailment | 16 | EM | 0.417 | 0.892 | +0.475 | ✅ |
| task620 ohsumed_medical_heading | 43 | ROUGE-L | 0.188 | 0.582 | +0.394 | ✅ |
| task442 com_qa_paraphrase | 43 | ROUGE-L | 0.503 | 0.727 | +0.224 | ✅ |
| task1564 triviaqa_answer | 16 | EM | 0.182 | 0.364 | +0.182 | ✅ |
| task290 tellmewhy_answerability | 43 | EM | 0.550 | 0.725 | +0.175 | ✅ |
| task379 agnews_topic | 16 | EM | 0.625 | 0.783 | +0.158 | ✅ |
| task1342 amazon_reviews_title | 43 | ROUGE-L | 0.070 | 0.144 | +0.074 | ✅ |
| **task639 multi_woz_utterance** | 16 | ROUGE-L | 0.090 | 0.051 | **−0.040** | ❌ **quarantined** |

**Result: 13/14 pass, mean margin +0.41.** The lift is largest on classification
tasks where the base is near-random (stereoset 0.08→0.98) and smallest where the
base is already decent (agnews 0.63→0.78) — exactly the expected shape.

**Why task639 fails (and why that's the gate working).** The MultiWOZ adapter
**collapses to a fixed output** — it generated `"I'd like to go to the city
center."` for *every* input (refs: `"How about a thai place then?"`,
`"Thank you. Can you please give me the reference number?"`). It scores *below*
base, so it's quarantined. This is the same open-ended dialogue task that scored
~chance in Phase 0; the lesson holds — **memory/availability of an adapter ≠
competence**, and only an eval catches it. Quarantining it keeps a non-functional
adapter out of the hypernetwork's training data.

---

## Sprint 5 — Frozen held-out split + our own oracle LoRAs

**The split (`configs/phase1/heldout_split.yaml`, lock_hash `b0143d2dad87c937`).**
Partitioned over the 13 gate-passing tasks (deterministic, seed 42), mirroring
T2L's train/val/held-out shape at pilot scale:

| Bucket | Count | Tasks |
|---|---|---|
| **held-out** | 3 | task1391, task290, task379 |
| val | 1 | task442 |
| train | 9 | task190, task280, task391, task512, task620, task843, task1342, task1344, task1564 |
| contamination-removed | 0 | *(none in the pilot; T2L's 10 removals apply at full scale)* |

**Leakage guard asserted:** held-out ∩ train = ∅, all three buckets pairwise
disjoint, lock hash stable across re-emit (`tests/test_phase1_library.py`). This is
the single most important Phase-1 invariant — the held-out set is now immutable for
the rest of the project.

**Oracle LoRAs (`results/phase1/oracles.json`).** For each held-out task we trained
**our own** rank-16 LoRA (q/k/v_proj, lr 2e-4, 250 steps, 500 train samples) on
Mistral-7B — the adapter Phase 2 evaluates *against* and Phase 3 compares geometry
*with*, fully under our control. All three clear the gate:

| held-out task | metric | base | **our oracle** | margin | library adapter | peak VRAM | ckpt |
|---|---|---|---|---|---|---|---|
| task379 agnews_topic | EM | 0.625 | **0.933** | +0.308 | 0.783 | 25.7 GB | 39 MB |
| task290 tellmewhy | EM | 0.550 | **0.758** | +0.208 | 0.725 | 23.2 GB | 39 MB |
| task1391 winogrande | EM | 0.100 | **0.433** | +0.333 | 0.908 | 20.6 GB | 39 MB |

Notable: our 250-step oracle **beats the downloaded library adapter on 2/3**
(agnews, tellmewhy) and trails on winogrande (0.43 vs 0.91 — that task wants more
steps). All fit comfortably under 32 GB (peak 20–26 GB; LoRA on a bf16 7B is a
QLoRA-class footprint, consistent with Phase 0.5). A longer LR/step sweep for the
oracles is a cheap follow-up if Phase 3 wants the tightest possible upper bound.

---

## Sprint 6 — Versioned artifact & reproducibility

**Committed artifacts:**
- `configs/phase1/library.yaml` — manifest, version `fee497c5fa710c14`; every task
  fully populated or explicitly quarantined (no silently-empty rows).
- `configs/phase1/heldout_split.yaml` — locked split, lock_hash `b0143d2dad87c937`.
- `results/phase1/library_quality.{csv,parquet,md}` — the gate table (S4).
- `results/phase1/oracles.json` + `results/phase1/oracles/*/checkpoint` — the
  held-out oracle LoRAs (S5).
- `results/phase1/gate_results.jsonl` — raw per-task gate rows (resumable log).
- Code: `src/lora_lab/library/{manifest,descriptions,gate,split,oracle,report}.py`;
  drivers `scripts/phase1_{library,finalize}.py`; tests `tests/test_phase1_library.py`
  (14 invariants, offline, all green).

**Reproducibility recipe.** `manifest` re-lists the Hub → same number↔name map;
`descriptions` re-extracts the same hashed `Definition`; `gate` re-evals greedily
(deterministic) → same margins within tolerance; `split` re-emits the same
`lock_hash` from seed 42. The manifest `version_hash` changes iff a tracked
artifact (adapter/description/split-role) changes.

**τ threshold:** margin ≥ 0.05 absolute on the task's own metric. Chosen as a
modest, defensible "clearly beats" bar; the pilot's pass-margins (median ~+0.39)
sit well clear of it, so the gate is not τ-sensitive here.

---

## Scope & what's left

This pass **piloted the full pipeline on 14 tasks** to produce genuine S4/S5/S6
numbers (per the sprint plan's "run everything, but partial is honest and
resumable" clause). The remaining work is *more runs of the same code*, not new
code:

- **Full gate sweep** — run `phase1_library.py gate` over all **1116** full-coverage
  tasks (~30–40 GPU-h, resumable via `gate_results.jsonl`; per-task isolation means
  one bad adapter never aborts the batch). The pilot's 1/14 fail-rate suggests a
  small quarantine tail to expect.
- **Full split at T2L scale** — once the full gate is in, re-partition to T2L's
  **479 train / 11 val / 10 contamination-removed** and apply the published
  contamination list (the `CONTAMINATION` hook is in place, empty at pilot scale).
- **Oracle LR/step sweep** (optional) — tighten the held-out upper bound if Phase 3
  needs it (winogrande in particular).

**Exit-gate status (pilot):** ✅ manifest committed & validated · ✅ gate run, every
non-quarantined adapter beats base by ≥ τ · ✅ frozen split locked + leakage-checked
+ own oracles trained · ✅ findings written. The gate **logic and artifacts** are
proven; clearing it at full 1117-task scale is the resumable batch above.
