# Phase 0.5 — Sprint Plan: Full Fine-Tune of Mistral-7B on This Box (Feasibility Spike)

*Sprint-planning material for the Phase 0.5 feasibility spike (run unattended overnight — no fixed time-box; see "Run scope & failure fallback"). Pairs with [`./llm_optimizations.md`](./llm_optimizations.md) (the technique × VRAM/RAM/time reference table), [`../notes.md`](../notes.md) §C2 (Phase 0.5) and §B (memory-budget practical tips), and the lit-review entries `§2.x` in [`../summaries.md`](../summaries.md).*

---

## What we are trying to achieve in Phase 0.5

Two questions, answered with measurements:

> **(A) Feasibility & fastest route** — Given 32 GB VRAM + 96 GB system RAM, can we run a *true full-parameter* fine-tune of `Mistral-7B-Instruct-v0.2` on this box — by which route, at what cost in speed and quality, and which route is fastest?
>
> **(B) Ablation study — per-optimization contribution & headroom** — A formal **ablation study**: *how much* does each individual memory optimization contribute (VRAM/RAM/speed), measured **in isolation** by toggling one lever at a time against a fixed reference config — even when an earlier optimization already made the run fit? We are not just trying to get Mistral-7B to *run*; we will stack a **hypernetwork on top of it later** (Phases 2+), so we need to know the **VRAM headroom** each lever frees, to budget for that addition. This is a named end-of-phase deliverable (see below).

With 96 GB RAM the answer to (A) is *likely yes* (CPU offload is now unblocked), so the spike's real job is **not** to prove feasibility but to **quantify the trade-offs**: which primary route is fastest, and — via the ablation study — **what each stackable lever individually contributes** to memory headroom and speed.

**Why run an ablation study (B) even if we already fit:** the natural temptation is to stop applying optimizations once a config fits in 32 GB. But "it fits" with zero headroom is useless here — the hypernetwork's parameters, its activations, and the backprop-through-base path (flagged in [`../notes.md`](../notes.md) §B) all need VRAM that the bare 7B full-FT doesn't account for. An ablation study is exactly the right instrument: hold everything fixed, toggle one optimization at a time, and attribute the standalone ΔVRAM / ΔRAM / Δspeed to each — turning "does it fit" into "*how much room is left, and which lever bought it*."

**The headline deliverables** are two artifacts:
1. A filled-in **technique × trade-off table** — one row per technique, measuring its effect on **peak VRAM**, **peak system RAM**, **training speed (wall-clock / step + tokens/s)**, and **free-VRAM headroom (32 − peak)**, plus whether it fits and an eval-quality number. This fills the skeleton already in [`llm_optimizations.md`](./llm_optimizations.md) ("Results to fill in (per technique)").
2. An **ablation study** (objective B) — `docs/phase-0.5-ablation-study.md` + `results/phase05/lever_ablation.{csv,parquet}` — reporting each stackable optimization's *isolated marginal* contribution and the resulting headroom recommendation for the hypernetwork.

Phase 0.5 populates both with real numbers from this hardware.

**Secondary deliverable:** at least one **working, committed config** that full-finetunes Mistral-7B end-to-end here, plus a short **feasibility note** with the memory math, the measured table, and a recommendation: *fastest viable route*, and *the speed-vs-quality trade-off* (so we can later decide whether to pay for a slower-but-better full FT or stick with a faster approximate one).

**Outcome artifacts:** `results/phase05/feasibility_table.{csv,parquet}` + a rendered Markdown table; the **ablation study** (`docs/phase-0.5-ablation-study.md` + `results/phase05/lever_ablation.{csv,parquet}`); per-technique **VRAM-and-RAM-vs-iteration traces/plots** (mirroring Phase 0's memory-vs-iteration plot); a W&B report (best-effort); and `docs/phase-0.5-findings.md`.

> Phase 0 deliberately forced its three-way comparison onto a *small* base so full FT fits natively (no offload tax). Phase 0.5 is the separate spike that confirms the **7B ceiling can be lifted** on this exact box. This document covers Phase 0.5 only.

---

## Hard constraints (these drive every design choice)

- **Single GPU: 32 GB VRAM. System RAM: 96 GB.** Both are measured ceilings, not just VRAM.
- **The wall:** bf16 + standard Adam on Mistral-7B ≈ 14 GB weights + 14 GB grads + ~56 GB optimizer states ≈ **84 GB** — far over 32 GB VRAM. So *every* technique here is about either keeping state on-GPU in a shrunken form (VRAM-direct) or spilling it to the 96 GB RAM pool (offload).
- **Two technique families, different trade-offs** (the spike measures the gap between them):
  - **Offload** (ZeRO-Offload / FSDP CPU-offload, §2.6/§2.10) — simplest path to a *working* full FT; PCIe-bound, so slower per step. Holds ~70 GB (fp32 Adam) / ~28 GB (8-bit Adam) in RAM — **fits in 96 GB**.
  - **VRAM-direct** (GaLore/Q-GaLore §2.9/§2.11, LOMO/AdaLOMO §2.12, BAdam §2.13, MeZO §2.14) — keeps the optimizer on-GPU; should be *faster*, at the cost of approximations to the true update.
  - **Stackable levers** (8-bit/paged Adam §2.8, gradient checkpointing §2.7, activation offload, drop fp32 master copy) multiply the headroom of whichever primary technique is chosen. **Each is also measured in isolation** (Sprint 6 ablation) so we know its standalone contribution — not just whether the stack happened to fit.
- **"Full-parameter" is non-negotiable for this spike.** LoRA/QLoRA are out of scope here (that was Phase 0). Every gate-clearing run must update *all* model weights (BAdam cycling through all blocks over the run still counts; LoRA does not).
- **Single GPU ⇒ training runs serialize.** "Parallel sprints" means parallel *development*; the actual benchmark runs queue on the one GPU. Schedule the long offload runs accordingly.
- **Fixed measurement protocol** so rows are comparable: same base (`Mistral-7B-Instruct-v0.2`), same 1–2 SNI task(s), **fixed batch size, sequence length, and seed** across every technique. Measure peak VRAM with `torch.cuda.max_memory_allocated()` per phase (§B); measure peak RAM with a sampled RSS probe.

### Technique benchmark order (cheapest-to-stand-up → hardest)

Mirrors the suggested order in [`llm_optimizations.md`](./llm_optimizations.md):

| # | Technique | Family | Why this order |
|---|---|---|---|
| 1 | **ZeRO-Offload + 8-bit Adam** | Offload | most likely to *just work* — establish the feasibility baseline first |
| 2 | **FSDP CPU-offload** | Offload | second offload data point; compare against ZeRO |
| 3 | **GaLore** | VRAM-direct | most promising on-GPU route; chase speed |
| 4 | **Q-GaLore** | VRAM-direct | best single bet to fit fully on-GPU (low-rank + quant) |
| 5 | **LOMO** | VRAM-direct | closest to "free" memory-wise; SGD-like footprint |
| 6 | **AdaLOMO** | VRAM-direct | LOMO + adaptive state; quality/speed comparison vs LOMO |
| 7 | **BAdam** | VRAM-direct | trades wall-clock for memory cleanly |
| 8 | **MeZO** | VRAM-direct | last resort; forward-only, slow/noisy |
| — | **ZeRO-Infinity (NVMe)** | Offload | **fallback only** — run *only if* 96 GB RAM unexpectedly pinches |

For the *primary*-technique rows above, stackable levers are applied as needed to make the run fit. Their **isolated marginal contributions** are measured separately in the **Sprint 6 lever-ablation study** — so "we didn't need lever X to fit" never means "we don't know what lever X is worth."

### Proposed repo layout (extends the Phase 0 layout)

```
src/lora_lab/methods/fullft/   one module per technique (galore.py, lomo.py, badam.py, mezo.py, offload.py …)
src/lora_lab/utils/            extend the VRAM helper with a host-RAM (RSS) probe
configs/phase05/               one YAML per technique run + one per lever-ablation cell (technique × stackable-levers × hparams)
results/phase05/               feasibility_table.{csv,parquet} + lever_ablation.{csv,parquet} + table.md · mem_trace/ · plots/
docs/phase-0.5-ablation-study.md  the ablation study writeup (objective B; written in Sprint 7)
docs/phase-0.5-findings.md     the feasibility note (written in Sprint 7)
```

---

## Sprints

Each sprint lists: **(1) Goal/objective · (2) Requirements (what needs to be accomplished) · (3) Definition of done · (4) Required testing.**

> **W&B applies to every run.** Sprints 2–6 (technique runs *and* lever-ablation runs) all execute through the Sprint 1 `benchmark()` harness, so each run **logs to W&B by construction** (live VRAM/RAM/loss curves + final summary row + config snapshot, named `MM_DD_YYYY_HR_MM_SEC_mistral7B_<technique-or-lever>`). It stays **best-effort/non-blocking** with an offline-local fallback — never a gate (project working pref). The Sprint 7 report aggregates these runs.

### Sprint 1 — Memory Math, Measurement Harness & RAM Probe  *(BLOCKER — must finish first)*

1. **Goal:** A fixed, trustworthy measurement protocol and the instrumentation to capture **both** peak VRAM **and** peak system RAM **and** wall-clock per technique — so every later row is apples-to-apples.

> **Pre-flight checklist (run/verify BEFORE any overnight benchmark kicks off — this is where an unattended run dies at step zero):**
> 1. **Gated-model access** — `HF_TOKEN` is set *and* the `Mistral-7B-Instruct-v0.2` license is accepted on its HF page. Verify with an actual `from_pretrained` (or `huggingface-cli download --dry-run`), **not** just "the token exists." (See [`gated-models-setup.md`](./gated-models-setup.md).)
> 2. **Blackwell/sm_120 wheels** — `torch.cuda.get_device_capability()` reports `(12, 0)`; PyTorch, `bitsandbytes`, and (if used) `flash-attn`/`xformers` all have sm_120 kernels. Older wheels silently lack them — confirm a real 4-bit forward + a bf16 backward run, per §B.
> 3. **Per-technique libraries import + smoke-run** — `deepspeed`, `accelerate` (FSDP), `galore-torch`, `lomo-optim`/`badam`/`mezo` (whichever a sprint needs) all import *and* run one step on a tiny model. A missing/ABI-broken optimizer lib is the second-most-likely overnight killer after the token.
> 4. **Disk headroom** — full-weights Mistral-7B checkpoints are ~14 GB each; confirm enough free space for the checkpoints you intend to keep (and decide retention before the run, not after it fills the disk).
> 5. **Unattended-run hygiene** — each technique run is wrapped so an OOM/crash in one technique logs `fits=no` + the failure and **moves on** to the next, rather than aborting the whole overnight batch.

2. **Requirements:**
   - Write the **memory-math reference** for Mistral-7B (weights / grads / optimizer / activations, per family) into the findings draft, so measured numbers can be checked against theory.
   - Extend the Sprint-1 VRAM helper from Phase 0 with a **host-RAM (RSS) probe** — a sampled background thread recording process + children RSS (use `psutil`) so offload's CPU-side footprint is captured, not just GPU.
   - Define the **fixed measurement config**: `Mistral-7B-Instruct-v0.2`, 1–2 SNI tasks, locked batch size, sequence length, grad-accum, seed, and step count. Commit it as `configs/phase05/_fixed_protocol.yaml`.
   - A `benchmark(technique_config)` entrypoint that runs N steps, samples `gpu_mem_gb` and `ram_gb` per step, and emits a per-run trace + a summary row (`peak_vram_gb`, `peak_ram_gb`, `wallclock_per_step_s`, `tokens_per_s`, `fits`).
   - **W&B logging (best-effort, non-blocking — reuse the Phase 0 logging layer):** every `benchmark()` run inits a W&B run and logs, per step, the live `gpu_mem_gb` / `ram_gb` curves + `train_loss` / `tokens_per_s`, and at the end the summary row + full config snapshot. **Run name:** `MM_DD_YYYY_HR_MM_SEC_mistral7B_<technique>` (matching Phase 0's scheme, with `<technique>` as the varying axis). Project: a dedicated Phase 0.5 W&B project. If creds are absent, fall back to `WANDB_MODE=offline` and persist locally under `results/phase05/runs/*/` (sync later with `wandb sync`) — **W&B never blocks or gates a run** (per project working prefs).
   - Confirm the Mistral-7B base loads (gated model — see [`gated-models-setup.md`](./gated-models-setup.md); needs `HF_TOKEN` + accepted license).
3. **Definition of done:** the harness runs an arbitrary technique config for N steps and writes a trace (VRAM **and** RAM vs. step) + a summary row; **the run logs to W&B online when creds are present and falls back to offline-local without blocking when they aren't**; the RAM probe returns sane non-zero numbers; the fixed protocol is committed; Mistral-7B loads on this box.
4. **Required testing:** RAM probe validated against a known allocation (allocate ~X GB, probe reads ~X GB); VRAM helper matches `nvidia-smi` within tolerance; the fixed-protocol config round-trips (load → run → reproduce identical seed/batch/seq); harness emits a schema-valid summary row on a trivial dummy run; **an offline (`WANDB_MODE=offline`) dummy run produces the expected local W&B artifacts and a disabled/absent-cred run still completes.**

### Sprint 2 — Offload Baseline: ZeRO-Offload + 8-bit Adam  *(needs S1; the "just works" feasibility proof)*

1. **Goal:** Prove a *working* full-parameter fine-tune of Mistral-7B exists on this box via the simplest route, and capture its trade-off row.
2. **Requirements:**
   - Stand up **ZeRO-Offload (DeepSpeed ZeRO-2/3 with `offload_optimizer` to CPU)** with **8-bit / paged AdamW (§2.8)** to shrink the offloaded state (~70 GB → ~28 GB) for RAM margin.
   - Stack **gradient checkpointing (§2.7)** and bf16; optionally **activation offload** if VRAM still pinches.
   - Run the fixed protocol end-to-end; record the VRAM+RAM-vs-iteration trace and the summary row.
   - Save a reloadable full-weights checkpoint and confirm loss decreases.
3. **Definition of done:** Mistral-7B trains end-to-end (all params updated) without OOM and without exceeding 32 GB VRAM / 96 GB RAM; trace + summary row persisted; checkpoint reloads and runs inference; this is the **feasibility-proven** baseline.
4. **Required testing:** loss decreases over the run; assert peak VRAM ≤ 32 GB **and** peak RAM ≤ 96 GB; assert *all* parameters received gradients/updates (not a LoRA subset); checkpoint reload + generation smoke test; measured offloaded-state size reconciles with the memory math (±tolerance).

### Sprint 3 — Second Offload Data Point: FSDP CPU-offload  *(needs S1; parallel-dev with S2)*

1. **Goal:** A second, framework-independent offload measurement (PyTorch-native FSDP) to cross-check ZeRO and compare offload implementations.
2. **Requirements:**
   - Configure **FSDP with CPU offload** (`offload_params` / optimizer-state offload) via `accelerate`, full-shard, bf16, gradient checkpointing.
   - Run the same fixed protocol; record trace + summary row.
   - Note any setup/stability differences vs. ZeRO-Offload (sharding wrap policy, Blackwell/sm_120 caveats per §B).
3. **Definition of done:** FSDP CPU-offload full-FT run completes within both memory ceilings; trace + row persisted; a short note on ZeRO-vs-FSDP setup differences captured for the findings doc.
4. **Required testing:** same asserts as Sprint 2 (memory ceilings, all-params-updated, loss decreases); FSDP and ZeRO peak-VRAM/RAM numbers are within an explainable range of each other (flag if wildly divergent).

### Sprint 4 — VRAM-Direct Methods: GaLore / Q-GaLore / LOMO / AdaLOMO  *(needs S1; the *fast* route; tracks parallelizable)*

1. **Goal:** Measure the on-GPU (no-offload-tax) techniques expected to be **faster** than offload, and capture the speed gap.
2. **Requirements:**
   - **GaLore (§2.9)** — low-rank gradient projection; full-param FT in far less VRAM. Tune projection rank + SVD update interval.
   - **Q-GaLore (§2.11)** — GaLore + quantization; the best single bet to fit fully on-GPU.
   - **LOMO (§2.12)** — fuse gradient compute with the update (SGD-like footprint; no full grad/optimizer materialization).
   - **AdaLOMO (§2.12)** — LOMO + adaptive state; compare quality/speed against plain LOMO.
   - Each run uses the fixed protocol; record VRAM+RAM trace and summary row (RAM should stay ~neutral here — confirm it).
   - *Internal tracks (GaLore family vs. LOMO family) developed in parallel against the shared `benchmark()` interface.*
3. **Definition of done:** each technique that *fits in 32 GB VRAM* trains end-to-end full-param without OOM; its trace + row persisted; any technique that does **not** fit is recorded as `fits=no` with the OOM point documented (a valid negative result).
4. **Required testing:** loss decreases (or, for noisy methods, trends down over enough steps); assert all params are in the trainable set; assert peak VRAM ≤ 32 GB for the "fits" claim; wall-clock/step recorded and compared to the Sprint 2 offload baseline (expect VRAM-direct faster — flag if not); GaLore SVD-interval sensitivity spot-checked.

### Sprint 5 — Remaining VRAM-Direct: BAdam + MeZO (+ NVMe fallback if needed)  *(needs S1)*

1. **Goal:** Round out the table with the time-for-memory (BAdam) and last-resort (MeZO) techniques, and run the NVMe fallback *only if* RAM unexpectedly pinched in S2/S3.
2. **Requirements:**
   - **BAdam (§2.13)** — block-coordinate: one transformer block holds grads/optimizer state at a time, cycling through all blocks (full-param over the run). Expect higher wall-clock — measure it.
   - **MeZO (§2.14)** — zeroth-order, forward-only (inference-level memory). Expect severe slowdown / noise — measure steps-to-signal, not just per-step time.
   - **ZeRO-Infinity NVMe offload (§2.6)** — run **only as a fallback** if the 96 GB RAM was insufficient in the offload sprints; otherwise mark `not needed` in the table with a one-line justification.
   - Each: fixed protocol, trace + summary row.
3. **Definition of done:** BAdam and MeZO rows captured (fits / VRAM / RAM / wall-clock / quality-spot-check); NVMe row either captured (if triggered) or explicitly marked not-needed with justification; BAdam confirmed to cover all blocks over the run (full-param).
4. **Required testing:** BAdam — verify every block is visited (full-param coverage), loss decreases across cycles, wall-clock penalty quantified vs. baseline; MeZO — convergence-trend check over many forward passes, memory confirmed at inference level; NVMe (if run) — memory ceilings + SSD-throughput note.

### Sprint 6 — Ablation Study: isolate each optimization's contribution  *(needs S1 + a feasible reference config from S2/S4; objective B)*

1. **Goal:** Run a formal **ablation study** measuring the **isolated marginal contribution** of each stackable optimization — 8-bit/paged Adam, gradient checkpointing, activation offload, drop-fp32-master-copy, and optimizer-offload-itself — to peak VRAM, peak RAM, and step time. This is the answer to "how much does each lever buy us," independent of whether it was *needed* to fit, so we can budget VRAM headroom for the hypernetwork we'll add on top. The raw study data lands in `results/phase05/lever_ablation.{csv,parquet}`; the writeup (`docs/phase-0.5-ablation-study.md`) is assembled in Sprint 7.
2. **Requirements:**
   - **Pick two reference frames** (anchors), both full-param Mistral-7B on the fixed protocol:
     - an **offload anchor** (the Sprint 2 ZeRO config) for the offload-side levers (8-bit Adam, optimizer-offload, activation offload), and
     - an **on-GPU anchor** (a Sprint 4 config such as GaLore) for the VRAM-direct-side levers (gradient checkpointing, drop-fp32).
     A lever is benchmarked against whichever anchor it meaningfully applies to (some, like gradient checkpointing, apply to both — measure on both).
   - **Two complementary sweeps per anchor**, so we capture both standalone effect and interaction:
     - **Add-one-in** — from a minimal config (levers off, raising batch/seq only as far as still-feasible), toggle each lever **on alone** and measure the delta.
     - **Leave-one-out** — from the full stack (all levers on), toggle each lever **off alone** and measure the delta. (When a lever is load-bearing — removing it OOMs — that *is* the result: record `fits=no` and the OOM point as its contribution.)
   - For each cell, change **exactly one flag** vs. its anchor; run the fixed protocol through the same `benchmark()` harness; record the VRAM+RAM-vs-iteration trace + summary row, tagged with `{anchor, lever, direction}`.
   - Emit `results/phase05/lever_ablation.{csv,parquet}` with per-lever **ΔVRAM_gb, ΔRAM_gb, Δwallclock_per_step_s, Δtokens_per_s**, and a **freed-VRAM headroom** figure per lever.
   - *Note the cost:* this is the sprint that most expands the overnight run (≈ 2 anchors × ~4 levers × 2 directions ≈ 12–16 extra short runs) — comfortably within the overnight window, so all cells run (no time-box; see "Run scope & failure fallback").
3. **Definition of done:** the lever-ablation table is populated — every stackable lever has a measured isolated ΔVRAM / ΔRAM / Δspeed (or an explicit "load-bearing: removing it OOMs" / "n/a for this anchor" marking); a **recommended headroom config** is identified (the lever stack that leaves the most free VRAM at acceptable speed, for the hypernetwork to live in); results feed the Sprint 7 table/note.
4. **Required testing:** assert each ablation cell differs from its anchor by **exactly one flag** (config diff check); sign/direction sanity (gradient checkpointing ↓VRAM but ↑step-time; 8-bit Adam ↓optimizer-state RAM/VRAM; drop-fp32 ↓VRAM slightly); deltas reconcile with the memory math (±tolerance); add-one-in and leave-one-out agree in sign for each lever (flag large disagreement as an interaction effect worth a note).

### Sprint 7 — Trade-off Table, Plots & Feasibility Note  *(needs S2–S6; the headline deliverable)*

1. **Goal:** Assemble the populated **technique × trade-off table**, the **lever-ablation table**, the comparison plots, and the recommendation — the artifact the whole spike exists to produce.
2. **Requirements:**
   - Aggregate every technique's summary row into `results/phase05/feasibility_table.{csv,parquet}` and render the Markdown table with columns: `technique, config/flags, fits (≤32GB VRAM / ≤96GB RAM), peak_vram_gb, peak_ram_gb, free_vram_headroom_gb (32 − peak), wallclock_per_step_s, tokens_per_s, eval_quality, notes` — i.e. fill in the skeleton already in [`llm_optimizations.md`](./llm_optimizations.md). The **headroom column directly answers "how much room is left for the hypernetwork"** under each route.
   - **Write the ablation study** (`docs/phase-0.5-ablation-study.md`) from Sprint 6's `lever_ablation.{csv,parquet}`: the ablation table (per-lever isolated **ΔVRAM / ΔRAM / Δspeed** and freed headroom), the lever-contribution chart, the add-one-in vs. leave-one-out comparison (interaction effects), and a one-paragraph conclusion on which levers are worth their cost and the recommended headroom stack for the hypernetwork.
   - **Add a relative-effect view** the user asked for: each technique's VRAM and **speed** expressed relative to the offload baseline (e.g. ×slower, % VRAM), so the speed-vs-memory trade-off is legible at a glance.
   - **Render comparison plots:** overlay per-technique **VRAM-vs-iteration** and **RAM-vs-iteration** from `results/phase05/mem_trace/`; a **speed-vs-peak-memory scatter** (wall-clock/step on one axis, peak VRAM on the other) so the Pareto front is visible; and a **lever-contribution bar chart** (freed VRAM per lever) for objective B.
   - **Quality eval (full):** run each technique's checkpoint over the **full held-out set of all fixed-protocol SNI tasks** (same SNI metric as Phase 0 — exact-match / ROUGE-L), so the "train longer for a better result" trade-off is grounded in a real per-task quality number, not just speed. Eval is inference-only and cheap relative to the training runs (~+1.5–4 h total across all techniques on one GPU), so it's affordable within the overnight benchmark window. **Caveat:** for noisy/slow methods (esp. MeZO, and BAdam mid-cycle) the short benchmark run may not have converged — report their quality as "where it reached at N steps," not as a fair quality verdict, and flag this in the table notes.
   - Write `docs/phase-0.5-findings.md`: memory math vs. measured reality, both tables, the plots, and an explicit **recommendation** — *fastest viable route*, *the speed↔quality trade-off*, and a **recommended headroom config for the hypernetwork** (which route + lever stack leaves enough free VRAM for Phase 2's hypernetwork while staying fast enough).
   - Push a W&B report (best-effort, per §B — never a gate).
3. **Definition of done:** the trade-off table is fully populated (every benchmarked technique has a row incl. free-VRAM headroom; non-fitting/not-run ones are marked, not blank); the **ablation study (`docs/phase-0.5-ablation-study.md` + `lever_ablation.{csv,parquet}`) is committed** — every stackable lever has an isolated contribution or an explicit load-bearing/n-a marking; each fitting technique has a **per-task held-out quality number** (with the not-converged caveat flagged where it applies); the memory-vs-iteration plots, speed-vs-memory scatter, and lever-contribution chart render from saved traces; the findings note with a clear recommendation (incl. the hypernetwork-headroom config) is committed; the table in `llm_optimizations.md` is updated (or linked) with the real numbers.
4. **Required testing:** table schema validation (no silently-empty cells — every technique is either measured or explicitly `fits=no`/`not-run`); plots render from saved traces with correct axes/units (GB vs. step, s/step vs. GB); numbers in the table reconcile with the per-run traces; the recommended config re-runs from its committed YAML and reproduces its row within tolerance.

### Sprint 8 — Follow-ups: deferred techniques + quality depth  *(needs S7; closes the remaining gaps)*

1. **Goal:** Close the gaps left after the first overnight pass — the two deferred techniques (**MeZO**, **FSDP CPU-offload**), the **LOMO/AdaLOMO gradient-clipping** fix, a **harder second eval task**, and a **method-combination** study — so the trade-off picture is complete and quality-grounded, not just memory/speed.
2. **Requirements:**
   - **MeZO** (zeroth-order, forward-only): implement the two-forward seed-synchronized perturbation/update loop (no backward, no grads, no optimizer state); confirm the inference-level memory floor; measure quality (expected low in a 50-step budget — note it).
   - **FSDP CPU-offload**: single-process FSDP with `CPUOffload(offload_params=True)` + transformer-layer auto-wrap; cross-check the DeepSpeed offload result; try fp32 and 8-bit optimizer.
   - **LOMO/AdaLOMO + gradient clipping**: implement LOMO's two-pass clipped update (`grad_norm` retains the graph → `fused_backward` reuses it); sweep LR to find where the SGD-like optimizer learns (the shared 1e-5 is an Adam LR).
   - **Second eval task** (`task1344` RTE entailment): re-run all working techniques on a harder task for cross-task quality signal.
   - **Method-combination matrix**: stack memory tricks (e.g. BAdam + 8-bit), spend headroom (batch/no-ckpt), GaLore rank sweep.
   - Fold every result into the findings tables and `llm_optimizations.md`; update the recommendation.
3. **Definition of done:** MeZO and FSDP have rows (fits + VRAM + RAM + speed + eval, or `fits=no` with the memory-math reason); clipped-LOMO table present; dual-task (task843 + task1344) trade-off tables with an eval-quality column; combinations table; recommendation updated to reflect quality (not just memory/speed); all committed + pushed.
4. **Required testing:** MeZO confirmed forward-only (no grads; VRAM ≈ inference floor); FSDP within both memory ceilings *or* `fits=no` reconciled with the offload memory-math; clipped LOMO does not diverge (loss bounded) and clears chance on eval; per-technique eval scores reconcile across the two tasks (ranking stable); table schema validation (no silently-empty cells).

> **Status: ✅ complete.** MeZO (13.8 GB forward-only floor; eval ~0 at 50 steps — needs far more), clipped LOMO (0.84 EM @ lr 5e-4 — the headroom winner), dual-task eval (task843 + task1344), and the combinations matrix (BAdam+8bit → 15.7 GB @ 0.80 EM) are all in the findings. FSDP CPU-offload: see findings (offload family). Results: `results/phase05/{feasibility_table,feasibility_table_task1344,combinations,clipped_lomo}.{csv,parquet,md}`.

---

## Parallelism map

```
        ┌─────────────────────────────────────────────┐
        │ Sprint 1 — Memory math + measurement harness │  (BLOCKER)
        │            + host-RAM probe                  │
        └───────────────────────┬─────────────────────┘
                                │
        ┌──────────────┬────────┴───────┬───────────────┐
        ▼              ▼                ▼               ▼
 ┌────────────┐ ┌────────────┐ ┌──────────────┐ ┌────────────┐
 │ S2 ZeRO-   │ │ S3 FSDP    │ │ S4 GaLore/   │ │ S5 BAdam / │   ← parallel DEV
 │ Offload    │ │ CPU-offload│ │ Q-GaLore /   │ │ MeZO       │
 │ +8bit Adam │ │            │ │ LOMO/AdaLOMO │ │ (+NVMe fb) │
 └─────┬──────┘ └─────┬──────┘ └──────┬───────┘ └─────┬──────┘
       └──────────────┴───────┬───────┴───────────────┘
                             ▼
            ┌──────────────────────────────────────────────┐
            │ S6 — Ablation study (isolate each lever's Δ)   │  ← needs a feasible anchor from S2/S4
            └───────────────────────┬──────────────────────┘
                                    ▼
            ┌────────────────────────────────────────────┐
            │ S7 — Trade-off table + ablation study +      │
            │      plots + feasibility note                │
            └────────────────────────────────────────────┘
```

- **Sprint 1** is a hard blocker — no run is benchmarked until the measurement protocol + RAM probe are green (otherwise rows aren't comparable).
- **Sprints 2–5** are independent *development* tracks against the shared `benchmark()` interface.
- **Sprint 6** (lever ablation) needs a feasible anchor config to exist first (from S2 for offload-side levers, S4 for on-GPU-side levers) — but reuses the same harness, so it's mostly extra *runs*, not extra code.
- **Single-GPU caveat:** "parallel" = parallel *development*. Actual benchmark *runs* serialize on the one 32 GB GPU — schedule the long offload runs (S2/S3) and the ablation sweep (S6) so they don't block the cheaper VRAM-direct runs.
- **Sprint 7** needs all technique rows + the ablation table.

---

## Run scope & failure fallback (no fixed time-box)

**There is no time-box.** The full matrix is benchmarked unattended overnight, and the runs are short (fixed N steps, not training to convergence) — all 8 techniques + the ~12–16 ablation cells + the held-out evals total an estimated **~6–12 GPU-hours**, which fits comfortably in one overnight window. So we **run everything**; nothing is dropped for time.

What this section *does* govern is **resilience** — what happens if a single technique misbehaves, so one bad run never sinks the batch:

- **Per-run isolation (already in the Sprint 1 pre-flight, item 5):** each technique/ablation run is wrapped so an OOM, hang, or crash logs `fits=no` + the failure and **moves on** to the next run. A hang gets a per-run wall-clock watchdog (e.g. kill a run exceeding a generous multiple of its expected step time) so MeZO's many forward passes or a stuck offload run can't consume the whole night.
- **Order so the cheap, high-value runs finish first** (in case the night *is* cut short by something external — power, a driver crash): Sprint 2 offload baseline → VRAM-direct (Sprint 4) → ablation study (Sprint 6) → the long-tail (BAdam, MeZO, NVMe fallback). That way even a truncated night still yields the feasibility proof + the fastest-route data + the ablation study.
- **Anything that didn't complete** is marked in the table as `not-run` (with the reason), never left silently blank — so a partial overnight result is still honest and re-runnable from its committed config.

**Minimum that must be present for the phase to count as done** (everything else is bonus the overnight run should also produce): Sprint 1 (the protocol), Sprint 2 (the one working feasibility proof), Sprint 6 (the ablation study — objective B; the hypernetwork's headroom budget depends on it), and Sprint 7 (the tables + recommendation).

---

## Phase 0.5 exit gate

Carried from [`../notes.md`](../notes.md) §C2 Phase 0.5:

> A short **feasibility note** with the memory math + **measured peak VRAM *and* peak system RAM + wall-clock per technique**, and a **working config that full-finetunes Mistral-7B end-to-end here** (expected via offload at minimum), **plus a recommendation on the fastest viable route**.

Concretely, the gate clears when: (1) at least one committed config provably full-finetunes Mistral-7B on this box within both memory ceilings; (2) the technique × trade-off table (incl. free-VRAM headroom) is populated for every benchmarked technique; (3) the **ablation study** (`docs/phase-0.5-ablation-study.md`) reports each stackable optimization's isolated contribution (objective B); and (4) the findings note states the fastest viable route, the speed↔quality trade-off, and a **recommended headroom config that leaves room for the Phase 2 hypernetwork**. Clearing this gate decides whether — and how cheaply, *and with how much headroom for the hypernetwork* — full FT of a 7B stays an option later in the project rather than being small-model-only.
