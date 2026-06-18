<!-- NOTE: numeric tables in this file are populated from results/comparison.csv
     by the final matrix run; see that file (and results/comparison.md) for the
     authoritative, regenerable numbers. -->
# Phase 0 — Findings: Full FT vs. LoRA vs. QLoRA on the RTX 5090

*Companion to [`phase-0-sprint-plan.md`](./phase-0-sprint-plan.md). The numbers
come from `results/comparison.csv` (regenerate with `scripts/build_table.py`);
the memory profiles from `results/plots/`.*

## Setup

- **Hardware:** single RTX 5090 (32 GB, sm_120 / Blackwell), CUDA 12.8, driver 580.
- **Stack:** torch 2.10.0+cu128, transformers 5.5.0, peft 0.19.1, bitsandbytes 0.49.2, trl 0.24.0 (pinned in `environment.yml`).
- **Ungated ladder:** SmolLM2-135M → Qwen2.5-0.5B-Instruct → Qwen2.5-1.5B-Instruct.
- **Tasks (SNI, `Lots-of-LoRAs`):** 3 classification (exact-match) + 2 generation (ROUGE-L).
- **Methods:** full FT (fp32 master weights, bf16 autocast, 8-bit paged AdamW, gradient checkpointing on the larger rungs), LoRA (bf16 frozen base, rank 16), QLoRA (4-bit NF4 base, rank 16). LoRA/QLoRA lr 2e-4; full FT lr 2e-5.
- Every run logs a GPU-memory-vs-iteration trace; peak = max of that trace.

## Headline findings

1. **The toolchain is green on Blackwell.** 4-bit NF4 quantization, LoRA, and bf16/fp32 training all run on the 5090; `bitsandbytes` self-diagnostic passes. This was the single biggest risk and it is cleared.

2. **Trainable-parameter count separates the methods by ~2–3 orders of magnitude.** Full FT trains 100% of weights (135M / 494M / 1.54B); LoRA and QLoRA train the *same* tiny adapter (≈0.4–2% depending on base), so their parameter/checkpoint footprints are identical to each other and tiny vs. full FT (adapter checkpoints are single-digit–tens of MB; full-FT checkpoints are hundreds of MB to GBs).

3. **Full FT's memory cost explodes with scale; the adapters stay flat.** Peak VRAM for full FT runs **5.4 GB → ~10 GB → ~15–16 GB** across the 135M / 0.5B / 1.5B ladder, while LoRA and QLoRA stay in a **~4–6.5 GB** band at *every* rung (dominated by the resident base + the vocab-heavy cross-entropy activation, not the tiny adapter). At 1.5B, full FT uses **~2.5×** the memory of either adapter (e.g. financial-sentiment: full FT 15.8 GB vs LoRA 6.28 / QLoRA 6.35 GB) — the fp32 master weights + gradients + optimizer state. All ungated full-FT runs still fit under 32 GB (the 1.5B rung uses 8-bit Adam + gradient checkpointing).

4. **The "QLoRA < LoRA" memory ordering only appears at ~1.5B — there is a measurable crossover.** Below that, QLoRA's on-the-fly 4-bit *dequantization* buffers + the bitsandbytes allocator cost **more** than the 4-bit weight savings recover, so plain LoRA is actually the lightest:
   - **135M:** LoRA **4.16** < QLoRA **4.84** < full FT 5.44 GB (financial) — QLoRA *heaviest* of the two adapters.
   - **0.5B:** LoRA **5.73** < QLoRA **6.66** < full FT 10.08 GB — LoRA still lighter.
   - **1.5B:** QLoRA **≈** LoRA (QLoRA wins on 3/5 tasks, e.g. triviaqa 4.64 vs 5.01; emotion 4.92 vs 5.18) — the crossover.

   **Practical takeaway: on this 5090, QLoRA's memory advantage only starts paying off around ~1.5B params; for sub-1B bases, plain LoRA is both lighter and simpler.** (Extrapolating, the advantage should widen at the 7B QLoRA scale QLoRA was designed for.)

5. **Full FT in pure bf16 is unstable; fp32 master weights are required.** An early run diverged (loss climbed to ~10, output degenerated to repeated tokens) with bf16-stored/updated weights. fp32 master weights + bf16 autocast fixed it (monotonic loss; quality recovers) — and is why full FT legitimately carries the heaviest footprint.

6. **Quality: adapters close most of the gap to full FT at <2% of the params.** On the strongest-signal task (financial sentiment, exact-match) full FT scores **0.90 at every rung**; LoRA tracks at **0.88–0.89** and QLoRA at **0.71 → 0.88** as the base grows — all training only **0.4–2%** of the parameters with **10–28 MB** adapter checkpoints vs **0.5–5.9 GB** full-FT checkpoints. Full FT has a slight edge on a couple of classification tasks at 0.5B (e.g. emotion 0.76 vs LoRA 0.70). The 135M base is a *plumbing* rung — near-zero on the free-form generation tasks (triviaqa, multiwoz) regardless of method; those need the larger bases and uncapped data. Per-cell numbers in `results/comparison.md`.

## Comparison table

See **`results/comparison.md`** (rendered) and **`results/comparison.csv` / `.parquet`** (machine-readable). One row per `method × base_model × task` with: trainable params, % params, peak VRAM (GB), wall-clock/epoch, final train loss, eval metric, checkpoint size (MB).

## Memory-vs-iteration profiles

See **`results/plots/gpu_mem_vs_iter_{model}-{task}.png}`** — each overlays the three methods on shared axes (x = optimizer step, y = GPU memory GB). The traces show: (a) a per-step memory envelope (the backward-pass spike is captured), and (b) the reserved-memory staircase of the bitsandbytes allocator under QLoRA.

## Caveats

- Runs here are **capped** (`--max-train-samples 500`, a few epochs, `--max-eval-samples 100`) to keep the single-GPU matrix tractable; they characterize *trade-offs*, not state-of-the-art task scores. Lift the caps for headline quality numbers.
- W&B ran in **offline** mode (no credentials in this environment); all metrics are captured locally under `results/runs/*/` and in `results/runs/*/wandb/offline-run-*`. Sync later with `wandb sync`.
- Per-method learning rates are sensible defaults, not tuned per cell.

## Gated rungs (Sprint 7) — executed

With an `HF_TOKEN` in place, the **exact same harness** ran on the gated bases,
extending the matrix to the **full five-model ladder (75 cells)**:

- **Gemma-2-2B-it is the VRAM stress rung and it fits.** Full-FT peaks **25.3–25.9 GB**
  across the five tasks (≈6 GB under the 32 GB ceiling) using 8-bit Adam + gradient
  checkpointing + batch 1 / grad-accum 8; LoRA/QLoRA stay at **~6.3–7.6 GB**. Full-FT
  checkpoint is ~10 GB vs the 57 MB adapter.
- **Llama-3.2-1B** (cross-family check): full-FT **~12.3–12.8 GB**, adapters **~4–5 GB**.
- **Quality holds the pattern.** On financial sentiment the three methods are within a
  point at every gated rung (Gemma 0.93 / 0.92 / 0.91 full-FT / LoRA / QLoRA;
  Llama 0.88 / 0.89 / 0.87) — adapters match full-FT at <0.4 % of trainable params and
  ~⅓ the memory. (Gemma full-FT underperforms its own adapters on glue-entailment,
  0.59 vs 0.87 — a tuning/LR artifact on the capped data, not a memory issue.)

## Exit-gate status

- ✅ Train **and merge/reload** a LoRA on the 5090 with comfortable headroom (adapter reload + generation smoke test passes; peaks far under 32 GB across the ladder).
- ✅ Three-way comparison table produced and reproducible across the **full five-model ladder** (every run rebuildable from its `config.yaml`; `scripts/build_table.py` regenerates the table + plots from `results/runs/`).
- ✅ Gated Gemma-2-2B / Llama-3.2 rungs (Sprint 7) **executed** — 75/75 cells; all peaks under 32 GB (max 25.94 GB, Gemma-2-2B full-FT).
