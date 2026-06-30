# Phase 2 — Hypernetwork Sizing Options (T2L S/M/L vs. ours)

Comparison of the current hypernetwork parameterization against the three
Text-to-LoRA (T2L) variants from the paper (Sakana AI, arXiv 2506.06105). The
S/M/L axis is **output-head granularity**; this repo's `vera → lowrank → full`
ladder (`src/lora_lab/hypernet/heads.py`) is its own expressivity ladder, and
only the `full` rung lines up with a paper variant (T2L-L's full-A-and-B output).

All quality / time figures are **predicted, not measured** — extrapolated from
the paper's trends plus this repo's setup (4-bit Mistral-7B, q/k/v targets, SNI
held-out gate, single 32 GB GPU). They are hypotheses for the gate run to
replace with real numbers.

> **Decision (committed): low-rank LoRA (`LowRankABHead`).** A single-target
> overfit diagnostic settled the `vera → lowrank → full` ladder empirically: the
> **VeRA** rung *cannot reconstruct* a target ΔW (relative error 1.0000 → 0.9999 —
> its frozen random A/B only get reweighted, never reshaped), so it is **rejected**
> as the default even though it is the smallest. **Low-rank LoRA** fits the same
> target cleanly (1.0000 → 0.1966) and is the committed parameterization;
> **full** A/B OOMs (~2.65 B params with Adam states). VeRA/full stay in the code
> as the documented (rejected) ladder rungs.

| Variant | Output head / what defines it | Params (≈) | Predicted held-out quality (vs per-task LoRA oracle = 100%) | Train Time (2k-step SFT, 1×32 GB) | Pros | Cons |
|---|---|---|---|---|---|---|
| **Ours (committed)** — low-rank LoRA hypernet | Per-(layer,module) `LowRankABHead`s: hypernet generates **real** A and B through a low-rank bottleneck from task⊕layer⊕module conditioning. Shared trunk, per-target heads. | ~151M | ~80–90% | ~1.5–2.5 hr | Reconstructs a target ΔW (the VeRA rung can't); fits SFT on 32 GB; one forward per target. | Off T2L taxonomy; more params than the paper's S/M variants (per-target, not shared, heads — see Prerequisite). |
| Ours — VeRA hypernet *(rejected)* | Per-(layer,module) `VeRAHead`s: frozen random A/B, hypernet generates scaling vectors `(d,b)` from task⊕layer⊕module conditioning. | ~56M | n/a — fails reconstruction | ~1.5–2.5 hr | Cheapest; smallest output; strong inductive bias. | **Cannot reconstruct a target ΔW** (1.0000 → 0.9999): only reweights fixed random directions, never reshapes them → fails the warmup and dead-ends the SFT gate. |
| **T2L-S** | Single shared head + rank-index embedding; emits one rank slice at a time. | ~5M | ~80–88% | ~1.8–3 hr | Smallest; best reported held-out generalization; lowest memory. | Slow generation (per-rank loop → r× hypernet forwards/step); biggest code change. |
| **T2L-M** | One shared head emitting full A and B, conditioned on layer⊕module. | ~34M | ~85–92% | ~1.5–2.5 hr | Paper's expressivity/size sweet spot; one forward per target; no throughput hit. | Higher SFT memory; needs shared-head refactor. |
| **T2L-L** | Per-module-type heads (q/k/v), each emitting full A and B; shared across layers via `layer_emb`. | ~55M | ~80–90% (high variance) | ~2.5–4.5 hr | Max adapter expressivity; closest to `FullABHead`. | Highest memory (32 GB OOM risk); weakest inductive bias → overfit risk; most params. |

## How to read this

- **Quality is non-monotonic in size.** Expected held-out ranking is roughly
  **M ≳ ours ≳ S > L**. Smaller *shared-head* output parameterizations (S/M)
  impose an inductive bias that helps generalization to *unseen* tasks; our
  committed low-rank LoRA trades more params (per-target heads) for the ability
  to emit real, reshapeable A/B — the capability the rejected VeRA rung lacked.
  L's extra capacity buys seen-task fit, not held-out gains, and OOMs on 32 GB.
- **Train time is base-bound, not hypernet-bound.** The SFT wall-clock is
  dominated by the frozen 7B base forward/backward, so ours/S/M cluster together
  (~1.5–2.5 hr). Only **L** slows down — its larger activations/grads pressure
  VRAM, forcing smaller batches / more grad-accum and risking OOM restarts on
  32 GB. S's per-rank generation loop adds a small per-step tax.
- **Reconstruction warmup** (the optional `warmup_from` phase) has *no base
  forward*, so it's minutes not hours; there time scales with hypernet size, but
  it's cheap enough not to move the total.

### Full-run training-time estimate (the scaled ~400-task run)

The per-variant times above are for a 2k-step run. The actual Phase-2 run scales
steps (not tasks — wall-clock tracks #steps, not #train tasks) for coverage of
the ~400-task train pool:

| Stage | Steps | Est. wall-clock (1×32 GB) |
|---|---|---|
| Recon warmup (S3) | ~2,000 | ~20–30 min (no base forward) |
| SFT meta-train (S4) | ~6,000 (batch 4) | ~5–7.5 hr (base-backprop-bound, ~3–4.5 s/step) |
| Held-out eval (S5) | 30 tasks × 4 conditions | ~1–2 hr |
| **Total (low-rank LoRA default)** | — | **< 10 hr** |

T2L-L adds OOM-risk / batch-throttle on 32 GB (see the variant table); the per-step
rate is unchanged, but a forced smaller batch needs more steps for the same
coverage, pushing SFT toward the upper end. ~6,000 SFT steps × batch 4 ≈ **24,000
example-passes** (~60 per train task).
- **Ours = the committed default.** Low-rank LoRA is the smallest rung that can
  actually reconstruct a target ΔW (VeRA can't) while still fitting SFT on 32 GB
  (full A/B OOMs) — the reason it's the committed parameterization.
- **Cheapest upgrade if the gate underperforms:** dedup the per-target heads into
  **one shared head** queried by layer/module embeddings ≈ **T2L-M** — the paper's
  best predicted quality at fewer params and roughly the same train time as the
  current low-rank LoRA default.

## Prerequisite for any T2L variant

Today `HyperLoRAGenerator` builds **one head per (layer,module) target**
(`model.py:137–142`). All three paper variants instead use a **single shared
output head** queried with layer/module(/rank) embeddings. So step 0 for S, M,
or L is to collapse `self.heads` (ModuleDict) into one shared head and route the
per-target distinction entirely through `layer_emb` / `module_emb`.

## Run context (as of 2026-06-24)

- **Training data:** the locked `configs/phase1/heldout_split.yaml` assigns
  **9 train tasks** (1 val, 3 held-out). SFT samples those 9 tasks' SNI sets at
  `batch_size=4 × max_steps=2000` ≈ 8k examples seen; reconstruction uses the 9
  library LoRA adapters as targets. This is a pilot-scale split — task diversity
  for generalization is thin, so expanding the (locked) train split is a likely
  high-leverage change before the real gate.
- **Hardware:** single 32 GB VRAM GPU (consistent with an RTX 5090). The
  memory-critical step is backprop through the frozen quantized base into the
  hypernet.
- **Quantization does NOT degrade the produced hypernet.** Only the **frozen
  base** is 4-bit NF4 (QLoRA-style feature extractor); the hypernet trains in
  bf16/fp32 and its kept weights are never quantized. The one caveat: the
  hypernet is specialized to the 4-bit base (it backprops through it), so eval
  must use the same 4-bit base (it does). Other memory levers (gradient
  checkpointing, bf16, 8-bit/paged AdamW, activation offload) are throughput/
  memory trade-offs, not quality degraders.

---

# Choosing the train/held-out split (which generalization to test)

The committed 9-train-task split is a **pilot** (only 14 tasks carried
`pilot: true` and were fed into `make_split`). The library actually has **1,037
gate-passing tasks** ready (564 generation + 473 classification); T2L itself
trained on 479. Two changes are needed before the gate means anything:

1. **Scale the train pool** to a few hundred tasks across all families — the
   hypernet needs task diversity to learn the description→adapter map at all.
2. **Curate the held-out set** for diagnostic power instead of the random 3, so
   we can actually tell generalization from memorization/retrieval.

The gate is `generated > nearest-neighbor retrieval`. So a good held-out task is
**far enough from every train task that retrieval grabs the wrong LoRA, yet
compositionally reachable** from the training distribution — the
"plausible-but-wrong-to-retrieve" zone where generalization is unambiguous.

| Held-out axis (family) | Train on → hold out | Tasks available (≈) | Defeats retrieval baseline? | Diagnostic clarity | Pros | Cons |
|---|---|---|---|---|---|---|
| **Format transfer** ⭐ (recommended) | Train the `*_classification` form, hold out the `*_answer_generation` form of the same dataset (and vice-versa). | 31 paired datasets (sentiment140, amazonreview_polarity, imdb, financial_phrasebank, europarl, …) | **Yes (strong)** — retrieval lands on the same-topic LoRA with the wrong output format → fails. | **Highest** — isolates "did it parse the task spec," not topic. | Cleanest single demonstration; retrieval can be embedding-near yet wrong; paired data already exists. | Needs the description to state output format clearly; fewer held-out tasks than translation. |
| **Language transfer** | Train many `X-en`/`en-X` pairs, hold out a language pair unseen in that direction. | ~249 translation tasks | **Yes** — retrieval returns a different-language LoRA → fails. | **High** — easy to narrate ("translated an unseen pair"). | Large pool; intuitive; strong retrieval-defeat. | Generation scoring (ROUGE/EM) noisier; some pairs scarce (1 task each). |
| **Domain transfer** | Same skill, new domain — train sentiment on Amazon/Yelp/poems, hold out Twitter/financial/Bengali. | ~30 sentiment + 271 classification | **Partial** — retrieval may grab a same-skill other-domain LoRA that does okay → weaker margin. | **Medium** — confounds skill vs. domain. | Realistic use case; abundant classification data. | Tougher retrieval competitor → smaller/ambiguous gate margin. |
| **Leave-one-family-out** | Train all families except one (all NER, or coreference); hold out that whole family. | 242 NER / 4 coreference / 17 NLI | **Yes (strong)** — no nearby LoRA exists. | **High but binary** — likely a clear pass or clear fail. | Unarguable if it works; tests true compositional reach. | Hardest; a flat fail tells you little about why. |
| **Random held-out** (current pilot) | Random N tasks held out. | n/a | **No** — a near-duplicate train task often exists → retrieval wins. | **Low** — can't separate generalize from memorize. | Zero design effort. | Weak/uninformative gate; the reason to move off it. |

**Recommendation:** train **broadly across all families** (scale to a few
hundred tasks), but set the **held-out axis = Format transfer** as the headline
diagnostic, with **Language transfer** and **Domain transfer** tasks as
secondary held-outs. Format transfer is strongest because retrieval can be near
in embedding space yet still wrong.

**Make it measurable:** for each held-out task, record the description-embedding
distance to its nearest train task (the retrieval baseline already computes it),
then plot score-vs-distance for generated / retrieval / oracle / base.
Generalization is "obvious" when generated stays high as distance grows while
retrieval decays — a curve, not a single number.

> Implementation note: re-locking the split means re-running `make_split` (or a
> curated variant) → new `lock_hash` in `configs/phase1/heldout_split.yaml`.
> Explicitly hold out the format-pair / domain-transfer tasks rather than letting
> them leak into training (the current pilot accidentally trains on
> twitter_emotion, financial_phrasebank, amazon reviews — those should be
> held-out targets).
