# LoRA_Lab

Research workspace for **training a hypernetwork that emits LoRA adapter weights on a single RTX 5090 (32 GB), studied through a mechanistic-interpretability lens.**

The core bet: replace per-task gradient descent with a learned function that outputs adapter weights directly — and then ask the question almost no one has asked yet, *are generated adapters mechanistically interpretable, and can interpretability tools both explain and improve weight generation?*

## Goals

1. **Build a working hypernetwork → LoRA pipeline** on one 5090 — base model frozen and 4-bit quantized, hypernetwork small and trainable. Start from the most reproducible recipe (text-conditioned LLM adaptation, distillation from a pre-trained LoRA library, à la Text-to-LoRA / Drag-and-Drop).
2. **Make it fit in 32 GB** by composing orthogonal memory levers: QLoRA/NF4 weights, gradient checkpointing, 8-bit/GaLore optimizer state, CPU offload as a pressure valve.
3. **Interpret the generated weights** — train SAEs on activation deltas, compare monosemanticity/feature-geometry of generated vs. hand-trained LoRAs, test whether the shared hypernetwork induces a structured weight basis, and whether generation can be steered in interpretable directions.

### The whitespace this targets
> *"Are hypernetwork-generated LoRA adapters mechanistically interpretable, and can interpretability tools both explain and improve weight generation?"*

Current work treats the hypernetwork as a black box judged only on downstream accuracy. Pointing SAEs / task-arithmetic / diff-interpretation at its outputs — on a controlled single-GPU testbed where we generate the weight distribution ourselves — sits in a gap none of the surveyed papers fully occupies.

## Repository structure

| Path | Contents |
|---|---|
| `summaries.md` | Per-paper summaries of the literature (~35 papers across hypernetworks, PEFT, VLAs, interpretability). |
| `themes.md` | Cross-cutting synthesis — the 5 themes and current research thrusts the project sits at the intersection of. |
| `notes.md` | Practical playbook — scoping questions, the 32 GB memory budget, Blackwell/sm_120 toolchain gotchas, engineering hygiene, phased timeline. |
| `docs/` | Sprint-planning material. `docs/phase-0-sprint-plan.md` breaks the first engineering phase into sprints. |
| `pdfs/` | Source PDFs, numbered by section (gitignored — large binaries). |

## Current status & next steps

This is currently a **literature-review and scoping phase** — no training code yet. Open decisions to settle before writing code (see `notes.md §A`):

- **Conditioning signal** for v0 (recommended: text task-description, most reproducible).
- **Output parameterization** — full A/B matrices vs. VeRA-style scalings vs. per-layer factors (output dimensionality dominates trainability and memory; start small).
- **Supervision** — task set and size of the LoRA library to distill from; budget the library-generation compute explicitly.
- **Base model** — quantized 7–8B (Llama/Qwen/Gemma class) is the sweet spot for 32 GB.
- **Generalization eval** — the held-out split design *is* the experiment.

First engineering phase (**Phase 0**, see [`docs/phase-0-sprint-plan.md`](./docs/phase-0-sprint-plan.md)): verify the Blackwell toolchain (CUDA 12.8+, current PyTorch, sm_120 `bitsandbytes`) end-to-end on a tiny base, then run a controlled **three-way fine-tuning comparison — full FT vs. LoRA vs. QLoRA** — on a common small model (laddering up to Gemma-2-2B-Instruct) across 3–5 SNI tasks, logged to **Weights & Biases**, producing a comparison table + results dataset. A companion spike (**Phase 0.5**) tests whether any memory trick can full-finetune Mistral-7B on this 32 GB box.

## Notes

- `pdfs/` and `.claude/` are gitignored; only the markdown notes are tracked.
