# Project Notes — Questions to Ask & Practical Tips

For: a first-year CS PhD student building a hypernetwork-that-outputs-LoRA on a single RTX 5090 (32 GB), with a mechanistic-interpretability angle. Pairs with [summaries.md](./summaries.md) and [themes.md](./themes.md).

---

## A. Research questions to ask yourself (in priority order)

### Scoping — answer these before writing any code
1. **What is the conditioning signal?** Text description (like T2L), a document (Doc-to-LoRA), a few prompts (Drag-and-Drop), or a demonstration/goal-image (VLA)? Pick **one** for v0. Text-conditioned LLM adaptation is the most reproducible starting point.
2. **What exactly does the hypernetwork output?** Full LoRA A/B matrices? Only B given a frozen random A (VeRA-style)? Only per-layer scalings? The output dimensionality dominates both trainability and memory — start small.
3. **Where does supervision come from?** The cleanest recipe (T2L, DnD) is **distillation**: train a library of ordinary per-task LoRAs first, then teach the hypernetwork to reproduce them. Decide your task set and how many LoRAs you can afford to pre-train on a 5090.
4. **What is the base model?** Quantized 7–8B (Llama/Qwen/Gemma class) is the sweet spot for 32 GB with QLoRA. Going to 13B is possible but tightens everything. Decide early — it sets the LoRA shapes.
5. **What's the held-out generalization test?** Unseen *tasks*? Unseen *task descriptions* for seen tasks? Unseen *documents*? Your eval split design is the experiment.

### The interpretability angle — your differentiator
6. **Are generated LoRAs more or less interpretable than trained ones?** Train an SAE on the activation deltas induced by both; compare monosemanticity, sparsity, feature-geometry (cf. Feature Geometry of LoRA Adapters, §4.10).
7. **Does the shared hypernetwork induce a structured weight basis?** Do generated adapters for related tasks live near each other in weight space (task-arithmetic style, §4.8)? Is there a low-dimensional, interpretable latent the hypernetwork uses?
8. **Can you steer generation interpretably?** Condition on (or edit toward) a known SAE feature or a ROME-style rank-one direction (§4.7) and check the behavioral effect is predictable.
9. **Can the generated adapter describe itself?** Diff Interpretation Tuning (§4.9) applied to *generated* (not hand-trained) diffs is, as far as these 35 papers show, unexplored.

### Method/scientific rigor
10. **What are the honest baselines?** (a) the per-task trained LoRA (upper bound), (b) zero-shot base model (lower bound), (c) a nearest-neighbor retrieval of an existing LoRA from your library. Beating (c) is the real bar — it's easy to forget.
11. **Does it generalize or memorize?** A hypernetwork can overfit to its LoRA library. Test interpolation (held-out tasks between seen ones) vs. extrapolation (genuinely new task types).
12. **What's the failure mode?** When generated LoRAs fail, is it bad task *identification* (wrong adapter) or bad task *execution* (right intent, wrong weights)? These need different fixes.

### Questions for your advisor / a meeting
13. Is the contribution the **method** (better weight generation) or the **science** (interpretability of generated weights)? They imply different experiments and venues. (Recommendation: lead with the science — it's less crowded.)
14. What's the minimum viable result that's publishable as a first-year? (Likely: "generated LoRAs are interpretable in way X, and that interpretability predicts/improves Y" — a focused, single-GPU story.)
15. Compute reality check: is one 5090 enough for the *meta-training* (not just inference), given you must first build a LoRA library? Budget the library cost explicitly.

---

## B. Practical tips for training on a 32 GB RTX 5090

### The memory budget, conceptually
Your trainable parameters are tiny (the hypernetwork + emitted LoRA). The base model is a **frozen 4-bit feature extractor**. So your situation is close to QLoRA, not full pretraining — this is the good news. Memory goes to four pools, attacked by different tools:

| Pool | Lever | Paper |
|---|---|---|
| Base weights | 4-bit quant (NF4) | QLoRA (§2.2) |
| Activations | gradient checkpointing | Chen 2016 (§2.7) |
| Optimizer state | 8-bit Adam / GaLore | (§2.8 / §2.9) |
| Spillover | CPU offload, per-layer streaming | ZeRO-Offload (§2.6), FSDP (§2.10) |

### Concrete stack to start with
- **`bitsandbytes` + PEFT + Transformers** (the QLoRA stack): load the base in 4-bit NF4, double-quantization on, `bnb` 8-bit paged AdamW for the hypernetwork's optimizer. This alone gets a 7–8B base comfortably into 32 GB.
- **Enable gradient checkpointing** on the base model's forward pass (`model.gradient_checkpointing_enable()`). It's the cheapest big win; expect ~20–30% slower steps for a large activation-memory cut.
- **Mixed precision: prefer bf16** (the 5090/Blackwell handles it well) over fp16 — fewer loss-scaling headaches.
- **Watch the backprop-through-base path.** Generating a LoRA and then backpropagating task loss through the *frozen quantized base* into the *hypernetwork* is the memory-critical step — this is the gradient-checkpointing + (if needed) activation-offload regime. Profile it first; it's where you'll OOM.
- **CPU offload is your pressure valve, not your default.** ZeRO-Offload / FSDP CPU-offload trades PCIe bandwidth for VRAM. Reach for it only when quant + checkpointing + 8-bit optimizer still OOM, because it slows you down.
- **If you ever want to train the base too:** GaLore (§2.9) enables full-parameter learning in ~24 GB by low-rank-projecting the optimizer state — relevant only if adapters prove insufficient. See Phase 0.5 for the full menu of full-FT-on-a-budget tricks (GaLore/Q-GaLore, LOMO, BAdam, MeZO).
- **Full fine-tuning has a hard ceiling in *VRAM*.** Even a **2–3B** model full-FT is near the 32 GB VRAM limit (needs 8-bit Adam + gradient checkpointing + small batch); **7B full FT won't fit on-GPU** with naive methods (≈84 GB). But with **96 GB system RAM**, CPU offload (ZeRO-Offload / FSDP CPU-offload) now holds the offloaded optimizer+gradient state (~70 GB) comfortably off-GPU — so 7B full FT is plausibly reachable here, at a PCIe-bandwidth speed cost (this is the Phase 0.5 spike). The Phase 0 three-way comparison still uses a small common base so that full FT fits *natively* (no offload tax) and the comparison stays apples-to-apples.

### 5090 / Blackwell-specific gotchas
- It's an **sm_120 (Blackwell)** card — you need a recent CUDA (12.8+) and current PyTorch nightly/stable built for it. Older `bitsandbytes`/`flash-attn`/`xformers` wheels may not have sm_120 kernels yet; budget time for the toolchain, and check each library's Blackwell support before committing. **This is the most likely thing to eat your first week — verify the stack runs end-to-end on a tiny model before scaling.**
- FlashAttention / fused-attention kernels matter for activation memory at longer context (relevant for Doc-to-LoRA-style long inputs) — confirm a Blackwell-compatible build.
- 32 GB is generous for QLoRA-on-7B but tight if you also keep a **LoRA library + SAE training** resident. Stage these: generate the library, checkpoint to disk, train the hypernetwork, then do interp as a separate pass.

### Engineering hygiene that saves PhD-months
- **Make a tiny end-to-end harness first** (125M–1B base, 3 toy tasks) that runs the full generate-LoRA → apply → eval → backprop loop. Get the *plumbing* right before the *scale*.
- **Log VRAM per phase** (`torch.cuda.max_memory_allocated()`), not just at the end — you need to know which pool blew up.
- **Version your LoRA library and eval splits** like data — reproducibility of the meta-training set is the experiment's backbone.
- **Deterministic seeds + a fixed held-out task set** from day one, or your interpretability comparisons won't be trustworthy.
- **Log every run to Weights & Biases** (loss, VRAM-per-phase, throughput, trainable-param %, and the full config snapshot) in a versioned W&B project, so the Phase 0 three-way comparison is auditable and reproducible. Keep it best-effort with an offline fallback — don't let W&B setup block a training run.

---

## C. The end goal (read this first)

> **Deliverable:** a workshop paper (NeurIPS/ICLR workshop, 4–8 pp) + an open artifact, supporting one claim:
> **"Hypernetwork-generated LoRA adapters occupy a measurably different (or measurably similar) feature geometry than hand-trained LoRAs for the same task — and here is the trustworthy measurement."**
>
> **Why this is a safe first project:** the result is publishable *either way*. A clean "they're different" is a finding (generated weights aren't just compressed trained weights); a clean "they're the same" is also a finding (the hypernetwork recovers the same solution). The risk is **not** "what if the answer is boring" — it's "what if the measurement isn't trustworthy." So the whole plan is built to make the measurement trustworthy: a working hypernetwork that *actually generates competent adapters*, compared against *properly trained* ones, with a *fixed held-out split*.
>
> **The artifact does double duty:** the LoRA library you build in Phase 1 is simultaneously (a) the hypernetwork's training data and (b) the "hand-trained LoRA" comparison set for the interp study. Build it once, use it twice.

## C1. Which base LLM? (decide this before Phase 0)

**We are an LLM project first** — VLAs and diffusion models are parked for later (see the parked-scope note at the bottom of `summaries.md`). The question is just: which language model is the base that your hypernetwork generates LoRAs *for*?

Sizes below are the **base model weights only**. On your 32 GB 5090 under QLoRA (4-bit frozen base + a small trainable hypernetwork/LoRA), the **~4-bit column is roughly what sits in VRAM** during meta-training, leaving plenty of room for activations/optimizer — anything up to ~8B fits comfortably.

| Model | Params | ~bf16 | ~4-bit (NF4) | Strengths / abilities | Role in this project |
|---|---|---|---|---|---|
| GPT-2 / SmolLM2-135M | 0.12–0.14B | ~0.3 GB | — | trivial to run; weak quality | **Phase 0 plumbing only** — get the generate→apply→backprop loop green fast |
| Qwen2.5-0.5B-Instruct | 0.5B | ~1.0 GB | ~0.4 GB | surprisingly coherent tiny chat | fast unit-test base for the harness |
| Llama-3.2-1B / 3B-Instruct | 1.2 / 3.2B | 2.5 / 6.4 GB | ~1.0 / ~2.2 GB | decent small instruct models | quick iteration if you want something between tiny and 2B |
| **Gemma-2-2B-Instruct** | 2.6B | ~5.2 GB | ~2.0 GB | strong for its size; **a T2L base** | **iteration / ablation base** — cheap, fast SAE runs and hyperparameter sweeps |
| **Mistral-7B-Instruct-v0.2** | 7.2B | ~14.5 GB | ~4.5 GB | solid 7B instruct; **T2L's primary base** | **★ PRIMARY base (recommended)** — the `Lots-of-LoRAs` library *and* T2L's main results both target it, so you get a ready-made LoRA library + matching task descriptions |
| Qwen2.5-7B-Instruct | 7.6B | ~15 GB | ~5.0 GB | stronger modern 7B | optional alt primary — only if you want a non-T2L base (but no ready LoRA library exists for it) |
| **Llama-3.1-8B-Instruct** | 8.0B | ~16 GB | ~5.5 GB | strong 8B; **a T2L base** | **secondary base** — Phase 4 cross-model robustness check (show the interp result isn't Mistral-specific) |

**Recommendation (this answers "which LLM"):** build on **Mistral-7B-Instruct-v0.2** as your primary base — it's the one decision that makes Phase 1 nearly free (the `Lots-of-LoRAs` adapters and SNI task descriptions both exist for it). Use **Gemma-2-2B-Instruct** for fast iteration and the cheaper SAE experiments, a **tiny model (GPT-2 / SmolLM2 / Qwen-0.5B)** for Phase-0 plumbing, and only add **Llama-3.1-8B-Instruct** in Phase 4 as a "does it generalize across base models?" check. All four are exactly the bases T2L itself reports on, so your results stay comparable to the paper.

## C2. Timeline (full-time, ~1 quarter / 10–12 weeks), built backward from that goal

Each phase ends in a **gate** — a concrete yes/no you must clear before the next phase is worth starting. Gates exist so you fail in week 3, not week 10.

**Phase 0 — De-risk the hardware *and* run the three-way fine-tuning comparison (Week 1).** *(This is the project's first engineering phase. Detailed sprint breakdown: [`docs/phase-0-sprint-plan.md`](./docs/phase-0-sprint-plan.md).)*
Stand up the Blackwell/sm_120 toolchain (in a **dedicated conda env** for this repo), then reproduce not just QLoRA but a controlled **three-way comparison — full fine-tuning vs. regular LoRA vs. QLoRA** — on a **common small base** (start at the smallest model for plumbing, then ladder up to **Gemma-2-2B-Instruct**), across **3–5 SNI tasks**, logged to **Weights & Biases**, producing a **comparison table + machine-readable results dataset** (quality, peak VRAM, wall-clock, trainable-param count). Seeing these differences empirically — on *this* hardware — is the point. Log VRAM per phase (§B).
*Why a small common base and not Mistral-7B: full FT of a 7B does not fit in 32 GB VRAM *natively* (≈84 GB for bf16+Adam). With 96 GB system RAM, CPU offload can now push the optimizer/gradient state off-GPU — but that adds a PCIe-bandwidth tax that would muddy a clean three-way comparison — so the apples-to-apples comparison still uses a base small enough that **full FT fits natively**. Whether 7B full FT is reachable on this box (now: likely yes, via offload) is the separate Phase 0.5 question.*
→ **Gate:** you can train *and merge* a LoRA on the 5090 with comfortable memory headroom, **and** the three-way comparison table is produced and reproducible across the model ladder. If not, the rest of the project is blocked — fix this before anything else.

**Phase 0.5 — Can we full-finetune Mistral-7B *here*? (feasibility spike, time-boxed).**
A focused investigation answering: given **32 GB VRAM + 96 GB system RAM**, is a true **full-parameter** fine-tune of `Mistral-7B-Instruct-v0.2` achievable, by which route, and at what cost in speed/quality? With 96 GB of RAM the answer is *likely yes* — the open questions are now **which path is fastest** and **what the throughput penalty is**, not whether it's possible at all. This still matters because Phase 0's three-way comparison is deliberately forced onto a *small* base (so full FT fits natively, no offload tax) — Phase 0.5 is where we confirm the 7B ceiling can actually be lifted.

*The wall — and why 96 GB RAM changes it:* bf16 + standard Adam on 7B ≈ 14 GB weights + 14 GB grads + ~56 GB optimizer states ≈ **84 GB** — far over 32 GB VRAM. The old blocker was that **CPU offload needs tens of GB of system RAM we used to lack**; with **96 GB RAM that escape hatch is now open** — ZeRO-Offload / FSDP CPU-offload can hold the offloaded optimizer + gradient state (~70 GB, less with 8-bit Adam) off-GPU while the 32 GB VRAM carries weights + activations. NVMe offload (ZeRO-Infinity) is no longer needed.

*Two families to benchmark, now both on the table:*

**A. Offload paths (newly viable — likely the simplest route to a *working* 7B full FT):**
- **ZeRO-Offload (§2.6) / FSDP CPU-offload (§2.10)** — move optimizer state (and gradients) to the 96 GB CPU pool; run the optimizer step on CPU. Fits, but pays a PCIe-bandwidth tax (slower steps). Pair with **8-bit / paged AdamW (§2.8)** to shrink the offloaded state (~70 GB → ~28 GB) and leave RAM margin.
- **ZeRO-Infinity NVMe offload (§2.6)** — no longer required now that RAM holds the state; keep only as a fallback if RAM still pinches.

**B. VRAM-direct methods (keep everything on-GPU — the *fast* route, no offload tax):**
- **GaLore / Q-GaLore** (§2.9 / §2.11) — low-rank gradient projection enabling full-parameter training in far less VRAM; Q-GaLore adds quantization. *Most promising VRAM-direct option.*
- **LOMO / AdaLOMO** (§2.12) — fuse the gradient computation with the parameter update to avoid storing full gradients/optimizer state (SGD-like footprint). *Designed for exactly this.*
- **BAdam** (§2.13) — block-coordinate optimization: only one transformer block carries gradients/optimizer state at a time, cycling through all blocks (full-param over the run).
- **MeZO** (§2.14) — zeroth-order, forward-only optimization (no backprop ⇒ inference-level memory); can full-FT large models but is slow/noisy. *Last resort.*
- **Stackable levers:** gradient checkpointing (§2.7), activation offload (now affordable against 96 GB RAM), dropping the fp32 master copy.

*The real trade-off is now speed, not feasibility:* offload is the easy "it works" path but PCIe-bound; the VRAM-direct methods keep the optimizer on-GPU and should be faster. Benchmark both and report the wall-clock gap.
→ **Deliverable / Gate:** a short feasibility note with **the memory math + measured peak VRAM *and* peak system RAM + wall-clock per technique**, and a **working config that full-finetunes Mistral-7B end-to-end here** (expected, via offload at minimum), plus a recommendation on the fastest viable route. This decides whether — and how cheaply — full FT can stay an option later in the project rather than being small-model-only.

**Phase 1 — Build the LoRA library (Weeks 2–3).**
Assemble a per-task LoRA for your conditioning set. Version the tasks, the LoRAs, and a **frozen held-out split** of tasks you will *never* train the hypernetwork on. These trained LoRAs are also your interp comparison baseline.
→ **Gate:** every library LoRA clearly beats the base model on its task, and the held-out split is locked. (Garbage library → garbage hypernetwork → meaningless interp comparison.)

> **Where to get the data (you probably don't need to train LoRAs from scratch).** T2L (§1.1) trains its library on the **Super-Natural Instructions (SNI)** task pool — specifically a 500-task English subset following Brüel-Gabrielsson et al. 2024 (479 train / 11 val / 10 removed for contamination). The three concrete sources:
>
> 1. **Pre-trained LoRA library + per-task datasets — `Lots-of-LoRAs` on HuggingFace** → https://huggingface.co/Lots-of-LoRAs — ~1,268 ready-made LoRA adapters and ~1,174 per-task datasets. These are the Brüel-Gabrielsson *"Compress then Serve"* release ([arXiv:2407.00066](https://arxiv.org/abs/2407.00066)) that T2L's subset is drawn from. **Crucially: the adapters are trained for `Mistral-7B-Instruct-v0.2` at rank 16** — so if you use that exact base model, you can *download* your whole LoRA library (and your hand-trained interp baseline) instead of training 479 adapters. That can collapse Phase 1 from weeks to days.
> 2. **The raw SNI task pool — `allenai/natural-instructions`** → https://github.com/allenai/natural-instructions (1,600+ tasks; HF mirror: `Muennighoff/natural-instructions`). Use this if you want to define your own task subset or generate fresh data.
> 3. **T2L's own code + task descriptions — `SakanaAI/text-to-lora`** → https://github.com/SakanaAI/text-to-lora. The natural-language task descriptions (your hypernetwork's conditioning input) are in the repo's `tasks/` folder; `./scripts/train_lora_baselines.sh` trains the oracle LoRAs; and pre-trained T2L checkpoints are at `huggingface.co/SakanaAI/text-to-lora`.
>
> **Recommended path:** base your project on **Mistral-7B-Instruct-v0.2**, pull adapters + per-task data from `Lots-of-LoRAs`, and grab the matching task *descriptions* from the SakanaAI repo. Train your own LoRAs (path 3's script) only for the handful of held-out tasks you want fully under your control, or if you later switch base models. *(Alternative library if you want non-SNI tasks: LoRA Land / Predibase, Zhao et al. 2024, [arXiv:2405.00732](https://arxiv.org/abs/2405.00732).)*

**Phase 2 — Train the T2L-style hypernetwork (Weeks 4–6).**
Implement a minimal text-conditioned hypernetwork; distill the library (the T2L/TAGI recipe, §1.1/§1.10); evaluate generated adapters on held-out tasks against three baselines: trained-LoRA (upper bound), base model (lower bound), and **nearest-neighbor retrieval of an existing library LoRA**.

> **Training objective — SFT, *not* reconstruction (a project-defining choice).** T2L can be trained two ways (§1.1, Fig. 1): *reconstruction* (Eq. 6 — L1-regress the generated ΔW directly onto a target LoRA; **no base-model forward pass**) or *SFT* (Eq. 5 — apply the generated ΔW to the frozen base, run the task, and **backprop the task loss through the frozen 4-bit base into the hypernetwork**). We commit to **SFT**, for two reasons:
> 1. **Generalization.** Reconstruction-trained T2L **fails to generalize to unseen tasks** (Table 6: 61.8 vs 66.3 avg; §5.4 / App. D — pre-trained LoRAs for similar tasks don't cluster in weight space), so only SFT can clear the held-out gate below.
> 2. **The interp study depends on it.** Reconstruction trains the hypernetwork to *copy* hand-trained LoRAs in weight space — so a "generated vs. hand-trained feature geometry" comparison (the §C deliverable, run in Phase 3) would be **circular**: they'd match by construction, not by discovery. SFT lets the hypernetwork find its *own* solution, which is what makes "do generated adapters occupy the same/different feature geometry?" a real, non-trivial measurement.
>
> This is the memory-critical backprop-through-base path flagged in §B — a **VRAM** cost (gradient checkpointing / activation offload), *not* a system-RAM cost. *(Reconstruction is still useful as cheap Phase-0 plumbing — it gets the generate→apply loop green on a tiny model with no base-model backprop — but every gate-clearing run must be SFT.)*

→ **Gate (the critical one):** generated LoRAs beat the *retrieval* baseline on held-out tasks. If they don't, the hypernetwork isn't really generalizing, and any interp comparison is comparing against a non-functional adapter. **Stop and fix the output parameterization** (VeRA-style smaller target §2.4, DoRA decomposition §2.3, better distillation) before touching SAEs.

**Phase 3 — The actual science (Weeks 7–9).**
Now run the comparison the paper is about: train SAEs on the activation deltas induced by *generated* vs. *hand-trained* LoRAs for the same held-out tasks, and quantify the difference with the established toolkit — feature-geometry / principal angles / CKA (§4.10), monosemanticity scoring (§4.4), and a task-arithmetic check on whether generated adapters compose like trained ones (§4.8).
→ **Gate:** you have a *measurement* (a number with error bars across tasks/seeds), not a vibe — whichever direction it points.

**Phase 4 — Lock it down & write (Weeks 10–12).**
Ablations (rank, task count, seeds), confirm the result is robust, write the 4–8pp workshop paper, clean and release the artifact. Target the next ML workshop deadline.
→ **Done:** submission + public repo.

**Where the slack is:** Phases 0, 2, and 3 are the ones that slip. If something runs over, cut *scope* (fewer tasks, one model, one rank) before cutting a *gate* — the gates are what keep the final result trustworthy.

## D. Things to read first (don't read all 35 at once)
- **Must-read core:** Text-to-LoRA (§1.1), HyperNetworks (§1.3), LoRA (§2.1), QLoRA (§2.2).
- **For the interp angle:** Towards Monosemanticity (§4.3) or Cunningham SAE (§4.4), Task Arithmetic (§4.8), and the two LoRA-interp bridges (§4.9, §4.10).
- **For VLAs (later):** OpenVLA (§3.1) + HyperVLA (§3.5).
- **Skim as references:** the hypernetwork survey (§1.8), the efficient-VLA survey (§3.7), the systems papers (§2.6–2.10).

---

*Caveats on the lit review itself:* a few entries are very recent or thin-provenance preprints (Doc-to-LoRA §1.2, HypeLoRA §1.6, Feature Geometry of LoRA §4.10 — single-author). Treat their specific numbers as provisional and verify against the PDFs before citing. Three Anthropic interpretability articles (§4.2, §4.3, §4.5) are web-only and not in `pdfs/`.
