# Phase 0 — Sprint Plan: Foundations & Three-Way Fine-Tuning Comparison

*Sprint-planning material for the project's **first engineering phase**. Pairs with [`../notes.md`](../notes.md) §C2 (timeline) and §B (32 GB practical tips).*

---

## What we are trying to achieve in Phase 0

Two goals, delivered together:

1. **De-risk the hardware/toolchain.** Stand up a working Blackwell/sm_120 training stack (CUDA 12.8+, current PyTorch, `bitsandbytes`, PEFT, `transformers`, `trl`) and prove we can train end-to-end on the 5090 with comfortable memory headroom. This is the single most likely thing to eat time, so it goes first.
2. **Produce a trustworthy three-way fine-tuning comparison.** Run **full fine-tuning vs. regular LoRA vs. QLoRA** on a *common small base model*, across **3–5 Super-Natural-Instructions (SNI) tasks**, fully logged to **Weights & Biases**, and emit a **comparison table + machine-readable results dataset**. The point is to *see and quantify the differences* — quality, peak VRAM, wall-clock, trainable-parameter count — before we scale up to the hypernetwork work in later phases.

**Outcome:** a reproducible harness + a results artifact (`results/comparison.csv`/`.parquet` + a rendered Markdown table + **a GPU-memory-vs-training-iteration plot overlaying the three methods** + a W&B report) that empirically characterizes how the three regimes trade off on this exact hardware.

> The companion question — *"can we full-finetune Mistral-7B on this box at all?"* — is **Phase 0.5**, a separate time-boxed feasibility spike (see `notes.md` §C2). This document covers Phase 0 only.

---

## Hard constraints (these drive every design choice)

- **Single GPU: 32 GB VRAM. System RAM: 96 GB.**
- **Full fine-tuning a 7B model does not fit on-GPU.** bf16 + standard Adam on a 7B ≈ 14 GB weights + 14 GB grads + ~56 GB optimizer states ≈ **84 GB**, over the 32 GB VRAM budget. (With 96 GB RAM, CPU offload *can* now spill the optimizer/grad state off-GPU — that's the Phase 0.5 spike — but offload adds a PCIe-bandwidth tax.) So the three-way comparison still runs on a **common small base** chosen so *full FT fits natively in VRAM*, keeping it apples-to-apples and offload-tax-free.
- **Full FT of even a 2–3B model is near the ceiling** — Gemma-2-2B full FT needs 8-bit Adam + gradient checkpointing + small batch to stay under 32 GB.
- **Single GPU ⇒ training runs serialize.** "Parallel sprints" below means parallel *development*; actual training jobs queue on the one GPU.

### Model ladder (apples-to-apples, smallest → up)

> **Ungated-first decision (2026-06-17).** Gemma-2-2B-Instruct and the Llama-3.2 models are **gated** on HuggingFace (need an accepted license + an `HF_TOKEN`). To keep Phase 0 fully autonomous with no credentials, the ladder below uses **ungated** models end-to-end. Swapping the gated Gemma/Llama bases back in is split out as **Sprint 7** (a follow-on test once a token is available). The harness is model-agnostic — only the config's `base_model` changes.

| Rung | Model | Gated? | Role |
|---|---|---|---|
| 0 | **SmolLM2-135M** | no | end-to-end plumbing; get all three methods green fast |
| 1 | **Qwen2.5-0.5B-Instruct** | no | small instruct model; fast iteration |
| 2 | **Qwen2.5-1.5B-Instruct** | no | top of ungated Phase 0; full FT here is the VRAM stress test |
| 3 | Gemma-2-2B-Instruct *(Sprint 7)* | **yes** | gated stress rung — run once `HF_TOKEN` is set |
| — | Llama-3.2-1B-Instruct *(Sprint 7, optional)* | **yes** | gated cross-family check |

All three methods (full FT, LoRA, QLoRA) run on the **same** base at each rung, so the comparison stays controlled. **Execution order is smallest-first:** confirm the full pipeline on SmolLM2-135M, then ladder up to Qwen2.5-0.5B and Qwen2.5-1.5B.

### Proposed repo layout (created during the sprints)

```
src/lora_lab/      data/ · methods/ · train/ · eval/ · utils/ (vram, logging)
configs/           one YAML per run (method × model × task × hparams)
scripts/           entrypoints (train, eval, build_table)
results/           comparison.csv / .parquet + table · mem_trace/ (per-run mem-vs-step) · plots/ (gpu-mem-vs-iteration)
environment.yml    dedicated conda env spec
```

---

## Sprints

Each sprint lists: **(1) Goal/objective · (2) What needs to be accomplished · (3) Definition of done · (4) Required testing.**

### Sprint 1 — Toolchain & Hardware De-risk  *(BLOCKER — must finish first)*

1. **Goal:** A reproducible Blackwell/sm_120 environment, in a **dedicated conda env for this repo**, where 4-bit quantization, LoRA, and bf16 training all run on the GPU.
2. **Accomplish:**
   - Create a dedicated conda env: `conda create -n lora_lab python=3.11` (or compatible).
   - Install + pin: CUDA 12.8+, PyTorch (sm_120 build), `transformers`, `peft`, `bitsandbytes`, `trl`, `accelerate`, `datasets`, `wandb`.
   - Commit `environment.yml` (+ `requirements.txt`/`pyproject.toml`) so the env is one-command reproducible.
   - Write `scripts/smoke_test.py`: load SmolLM2-135M in 4-bit NF4, attach a LoRA, run a forward + backward step, print `torch.cuda.max_memory_allocated()`.
   - Add a small VRAM-logging helper in `src/lora_lab/utils/`.
3. **Definition of done:** smoke test passes on the 5090 inside the `lora_lab` conda env; versions pinned & committed; `conda env create -f environment.yml` reproduces the env; VRAM helper returns sane per-phase numbers.
4. **Required testing:** smoke test (load/forward/backward in both 4-bit and bf16); `python -m bitsandbytes` self-diagnostic passes; assert the CUDA device is sm_120; confirm a LoRA attaches and one optimizer step runs without error.

### Sprint 2 — Data Pipeline (SNI tasks)  *(parallel with Sprint 3)*

1. **Goal:** Deterministic, versioned data loaders for **3–5 SNI tasks** with locked train/val/test splits.
2. **Accomplish:**
   - Select 3–5 SNI tasks (sources: `allenai/natural-instructions`, or per-task data from `Lots-of-LoRAs`).
   - Prompt formatting matching each base model's instruct/chat template; tokenization; fixed seeds.
   - A small held-out eval set per task; a `configs/tasks.yaml` manifest pinning task ids + split hashes.
   - `get_dataset(task, tokenizer)` returning tokenized train/val/test.
3. **Definition of done:** `get_dataset` works for every chosen task; splits are reproducible (hash-checked); decoded sample batches render correctly under each base model's chat template.
4. **Required testing:** unit test on split determinism (same seed → same hashes); token-length distribution sanity check; decoded-batch eyeball test; handling of empty / oversized examples.

### Sprint 3 — Experiment Harness + W&B Integration  *(parallel with Sprint 2 — W&B is best-effort, NOT a gate)*

1. **Goal:** A config-driven run harness parameterizing `{method × model × task × hparams}` that logs metrics + per-phase VRAM. **W&B is a thin, best-effort logging layer** — if auth/setup is unavailable, runs still execute and log locally.
2. **Accomplish:**
   - Dataclass/YAML config system; configs round-trip (load → run → reproduce).
   - Standardized logging: train/val loss, throughput (tok/s), step time, **GPU memory per step (a time series, so it plots as memory-vs-iteration)**, trainable-param count & %, full config snapshot.
   - W&B project init + run naming `{method}-{model}-{task}`; clean `WANDB_MODE=offline`/disabled fallback.
   - A `--dry-run` that logs a fake run end-to-end.
3. **Definition of done:** local logging (loss + VRAM + config) works and configs round-trip; W&B online logging works *when creds are present* but offline/disabled never blocks a run.
4. **Required testing:** config parse/validation tests; offline run produces expected local artifacts; VRAM logger returns sane numbers on the smoke model; dry-run completes without a GPU.

> **Note:** W&B is intentionally low-priority polish. The user will debug/test/clean up the whole phase later — do not get stuck perfecting W&B; a working offline logger is sufficient to proceed.

### Sprint 4 — Three Training Methods  *(needs S1; integrates S2+S3; 3 internal tracks parallelizable)*

1. **Goal:** One shared trainer interface with three interchangeable backends — **full FT**, **regular LoRA** (bf16 base), **QLoRA** (4-bit NF4 base) — selected by config.
2. **Accomplish:**
   - Common `train(config)` entrypoint.
   - LoRA / QLoRA backends via PEFT; a plain full-FT path.
   - For **Gemma-2-2B full FT**: enable 8-bit Adam + gradient checkpointing + small batch to stay under 32 GB.
   - **Record GPU memory *as a function of training iteration* for every run.** At each step (or a fixed step interval), sample `torch.cuda.memory_allocated()` / `torch.cuda.memory_reserved()` (GB) via the Sprint 1 VRAM helper and log `gpu_mem_gb` vs. `step` to the harness (W&B logs this as a live curve; also persist the raw trace to `results/mem_trace/{method}-{model}-{task}.csv` so it can be re-plotted offline). The per-run **peak** is just the max of this trace and still feeds the Sprint 5 `peak_vram_gb` column.
   - Checkpoint saving: adapter weights for LoRA/QLoRA, full weights for FT.
   - Each method trains end-to-end on the smallest model + one task.
   - *Internal tracks (full-FT / LoRA / QLoRA) can be built in parallel against the agreed interface.*
3. **Definition of done:** all three methods train end-to-end on the smallest model + one SNI task, log to the harness, and save a reloadable checkpoint without OOM; **a GPU-memory-vs-iteration trace is recorded and persisted for each method run**; Gemma-2-2B full FT verified to fit (documented peak from the trace).
4. **Required testing:** each backend runs a few steps with decreasing loss; trainable-param counts match expectation per method (full ≫ LoRA ≈ QLoRA); **the GPU-memory-vs-iteration trace is captured with one sample per logged step, non-zero, and its peak follows the expected ordering (QLoRA < LoRA < full FT)**; checkpoint reload + inference smoke test; OOM-guard test at the Gemma-2-2B full-FT rung.

### Sprint 5 — Evaluation & Comparison Table/Dataset  *(needs S2 + S4)*

1. **Goal:** One eval harness plus a generated **comparison table** and machine-readable **results dataset**.
2. **Accomplish:**
   - Task metric on the held-out set (exact-match / ROUGE-L per SNI convention).
   - Collect per-run rows: `method, base_model, task, trainable_params, pct_params, peak_vram_gb, wallclock_per_epoch, final_train_loss, eval_metric, checkpoint_size_mb`.
   - Write `results/comparison.csv` (+ `.parquet`) and render a Markdown table.
   - **Produce the GPU-memory-vs-iteration plot:** read the per-run memory traces from `results/mem_trace/` (Sprint 4) and render `results/plots/gpu_mem_vs_iter_{model}-{task}.png` overlaying the three methods (full FT / LoRA / QLoRA) on shared axes (x = training iteration/step, y = GPU memory GB), so the memory profiles are directly comparable. Save a combined matplotlib figure; also surface it in the W&B report.
   - Push a W&B summary/report (best-effort).
3. **Definition of done:** running eval over the smallest-model runs produces a populated table (one row per method × task), a results dataset file, **and the overlaid GPU-memory-vs-iteration plot(s)**; numbers reconcile with the logged metrics.
4. **Required testing:** metric correctness on a tiny known fixture; table schema validation; **the memory plot renders from saved traces with one curve per method and correct axis labels/units (GB vs. step)**; reproducibility — re-run eval on a saved checkpoint → same metric within tolerance.

### Sprint 6 — Scale-up & Full Comparison Matrix  *(needs all; GPU-serial)*

1. **Goal:** Execute the full matrix and deliver the Phase 0 result artifact.
2. **Accomplish:**
   - Run `{full FT, LoRA, QLoRA} × {3–5 SNI tasks} × model ladder up to Gemma-2-2B-Instruct`.
   - Populate the final `comparison.csv`/`.parquet`.
   - Build a W&B report; write a short findings summary (what differs across the three regimes in quality/memory/speed/params).
3. **Definition of done:** complete comparison table + dataset committed; W&B dashboard/report shared; findings note written; every run reproducible from its config.
4. **Required testing:** full-pipeline re-run from config on at least one cell to confirm reproducibility; sanity bounds on metrics (LoRA/QLoRA within an expected gap of full FT); assert VRAM never exceeds 32 GB across the matrix.

### Sprint 7 — Gated-model follow-on (Gemma-2-2B / Llama-3.2)  *(NEW — needs all + an HF token)*

1. **Goal:** Re-run the validated Phase 0 pipeline on the **gated** bases (Gemma-2-2B-Instruct, optionally Llama-3.2-1B-Instruct) to confirm the comparison generalizes beyond the ungated ladder.
2. **Accomplish:**
   - Set `HF_TOKEN` (`huggingface-cli login`) and accept each model's license on its HF page.
   - Reuse the exact same configs with `base_model` switched to the gated id (the harness is model-agnostic; only LoRA `target_modules` may differ by family — Gemma uses the same `q/k/v/o_proj` names).
   - Run `{full FT, LoRA, QLoRA} × {tasks} × {Gemma-2-2B(, Llama-3.2-1B)}`, append rows to `results/comparison.csv`/`.parquet`, and add the Gemma memory-vs-iteration plot.
   - Gemma-2-2B full FT is the real VRAM stress test (8-bit Adam + gradient checkpointing + small batch); document the peak from its trace.
3. **Definition of done:** gated rows present in the comparison table + dataset; Gemma full-FT peak documented and under 32 GB; plots rendered.
4. **Required testing:** token/licence preflight check; same metric-bound and VRAM-ceiling asserts as Sprint 6.

> **Status: ✅ complete.** Executed once an `HF_TOKEN` was provided — Gemma-2-2B-it
> and Llama-3.2-1B-Instruct ran the full `{full FT, LoRA, QLoRA} × 5 tasks` matrix.
> All 75 cells present in `results/comparison.*`; Gemma-2-2B full-FT peaks 25.9 GB
> (< 32 GB) with 8-bit Adam + gradient checkpointing + batch 1 / grad-accum 8. See
> `docs/phase-0-findings.md` (gated rungs section). Run via
> `scripts/run_matrix.py --tier all` (or `--config configs/matrix/run-matrix.yaml`).

---

## Parallelism map

```
        ┌──────────────────────────────┐
        │ Sprint 1 — Toolchain (BLOCKER)│
        └───────────────┬──────────────┘
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
 ┌──────────────────┐      ┌─────────────────────────┐
 │ Sprint 2 — Data  │      │ Sprint 3 — Harness + W&B │   ← run in PARALLEL
 └────────┬─────────┘      └────────────┬────────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
        ┌────────────────────────────────────────┐
        │ Sprint 4 — 3 Methods (FT / LoRA / QLoRA) │  ← 3 backends built in parallel
        └────────────────────┬───────────────────┘
                             ▼
              ┌────────────────────────────┐
              │ Sprint 5 — Eval + Table     │
              └──────────────┬─────────────┘
                             ▼
              ┌────────────────────────────┐
              │ Sprint 6 — Full Matrix (GPU-serial) │
              └────────────────────────────┘
```

- **Sprint 1** is a hard blocker — nothing GPU-touching starts until the toolchain is green.
- **Sprints 2 and 3** run in parallel (disjoint code, no GPU contention).
- **Sprint 4**'s three method backends are developed in parallel against the shared interface.
- **Sprint 5** needs S2 + S4. **Sprint 6** needs everything.
- **Single-GPU caveat:** "parallel" = parallel *development*. Actual *training runs* always serialize on the one 32 GB GPU — schedule the long full-matrix runs (S6) accordingly.

---

## Phase 0 exit gate

Carried from `notes.md` §C2 Phase 0: **you can train *and* merge a LoRA on the 5090 with comfortable headroom**, AND **the three-way comparison table is produced and reproducible** across the model ladder. Clearing this gate is the prerequisite for Phase 1 (build the LoRA library).
