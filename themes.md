# Themes & Current Thrusts

Cross-cutting synthesis of the [summaries](./summaries.md), oriented toward your project: *training a hypernetwork to output LoRA weights on a single RTX 5090, with a mechanistic-interpretability angle.*

---

## Theme 1 — Amortizing fine-tuning into a single forward pass

The central idea uniting Text-to-LoRA, Doc-to-LoRA, Drag-and-Drop LLMs, HypeLoRA, and HyperVLA: **replace per-task gradient descent with a learned function that emits adapter weights directly.** You pay a large one-time meta-training cost to train the hypernetwork, then adaptation becomes O(1) inference instead of O(thousands of steps). The conditioning signal is what differentiates the methods:

| Method | Conditioned on | Output |
|---|---|---|
| Text-to-LoRA | task *description* (text) | LoRA for an LLM |
| Doc-to-LoRA | a *document* | LoRA that internalizes context |
| Drag-and-Drop LLMs | a batch of *prompts* | LoRA weights |
| HypeLoRA | a *layer embedding* | per-layer LoRA factors |
| HyperVLA | *task* (robotics) | a compact policy |
| HyperLoRA (portrait) | *identity images* | LoRA for a diffusion model |

**The open design question for your project:** what is the right conditioning signal *and* the right output parameterization? The lineage runs HyperNetworks (2016) → LoRA (2021) → T2L/DnD (2025) — you're working at the current frontier.

## Theme 2 — The output space is the hard part (and where interp meets engineering)

Generating millions of LoRA parameters directly is hard to train and memory-hungry. Three responses recur:
- **Distillation of a LoRA library** (T2L, DnD): pre-train many ordinary LoRAs, then teach the hypernetwork to reconstruct/compress them. This gives clean supervision and is the most reproducible recipe to copy first.
- **Shrink the target** (VeRA, Transformer²): emit only per-layer *scaling vectors* over a shared/random basis, or only *singular-value* scalings, instead of full A/B matrices. Far fewer numbers to predict.
- **Structured low-rank edits as a first-class object** (ROME's rank-one edit, Task Arithmetic's task vectors): the broader ML community already treats weight deltas as structured, composable directions — which is both an inductive prior for your hypernetwork's output head and a reason to expect the outputs to be interpretable.

## Theme 3 — Fitting it on 32 GB: a stack of orthogonal memory levers

No single trick fits big training on a 5090; you compose them, and they attack *different* memory pools:
- **Weights** → 4-bit quantization (QLoRA / NF4).
- **Activations** → gradient checkpointing (√n memory for one extra forward pass).
- **Optimizer state** → 8-bit optimizers, GaLore's low-rank projection, or CPU offload (ZeRO-Offload).
- **Everything, streamed per-layer** → FSDP-style materialize-then-reshard ("load layers as you backprop").

Because you only train a *small* hypernetwork while the base model stays frozen and quantized, your situation is closer to QLoRA than to full pretraining — the base model is a frozen 4-bit feature extractor, and your trainable footprint is the hypernetwork plus the LoRA it emits. This is favorable: the 5090 is genuinely viable here.

**A second lever class — *making full fine-tuning itself fit*.** The levers above keep the base frozen; a separate family instead trains *all* parameters but shrinks the gradient/optimizer-state cost so full FT fits a small GPU:
- **Low-rank gradients** → GaLore / Q-GaLore project gradients (and optionally quantize the projections) into a low-rank subspace, keeping full-rank weight updates.
- **Fused update** → LOMO / AdaLOMO compute-and-apply the gradient in one step so full gradients/optimizer state are never stored (SGD-like footprint, Adam-like LR for AdaLOMO).
- **Block-coordinate** → BAdam keeps only one transformer block's gradients + optimizer state live at a time, cycling through all blocks.
- **Zeroth-order** → MeZO drops backprop entirely (forward-only), so training memory ≈ inference memory — slow but extreme.

This class is what the **Phase 0.5** spike benchmarks (`notes.md` §C2). On **32 GB VRAM + 96 GB system RAM** there are now *two* realistic routes to 7B full FT: CPU-offload paths (ZeRO-Offload/FSDP), newly unblocked because 96 GB RAM comfortably holds the offloaded optimizer/gradient state (the simple "just works" path, but PCIe-bound); and the *VRAM-direct* techniques above, which keep the optimizer on-GPU and should be faster. The spike's job is to measure the speed gap, not to prove feasibility. For Phase 0's apples-to-apples comparison we still use a small common base where ordinary full FT fits *natively* (no offload tax); these techniques are how the 7B ceiling gets lifted later.

## Theme 4 — VLAs as a second, harder testbed

The VLA papers (RT-1 → RT-2 → OpenVLA → π0) trace control becoming a token-prediction / flow problem on top of a VLM, and OpenVLA + the OFT study show LoRA already works for *single-arm, low-frequency* robot adaptation but **breaks down for high-frequency bimanual control.** HyperVLA is the direct analog of your project in robotics (hypernetwork → policy). The interesting tension: a text-conditioned hypernetwork is natural for LLM tasks, but for VLAs the conditioning may need to be a demonstration, a goal image, or an embodiment descriptor — a richer, multimodal signal.

## Theme 5 — Mechanistic interpretability as the "what did it actually learn?" lens

This is your differentiator, and the literature now has a real bridge from interp to weight-space:
- **Superposition / SAEs** (Toy Models, Towards/Scaling Monosemanticity, Cunningham) give you tools to decompose activations into interpretable features.
- **Circuits** (Math Framework, IOI) let you attribute behavior to specific heads/weights.
- **Weight-space results** (ROME, Task Arithmetic) show fine-tuning deltas are localized and composable.
- **Direct adapter-interpretation** (Learning to Interpret Weight Differences; Feature Geometry of LoRA Adapters) point SAEs and self-description *at LoRA deltas* — and the early finding that LoRA features are near-orthogonal to base features is provocative.

The synthesis: a hypernetwork that emits LoRA weights is a *generative model over the weight-delta space that interpretability is just learning to read.* That's an unusually clean place to do interp — you control the data-generating process.

---

## Current thrusts you're positioned at the intersection of

1. **Generated adapters vs. trained adapters — the quality gap.** Hypernetwork-emitted LoRAs still trail individually-trained ones on hard tasks. Closing this gap (better output parameterization, DoRA-style decomposition, better distillation targets) is open.
2. **Conditioning signal richness.** Text → document → prompts → demonstrations. What's the most *sample-efficient* and *generalizable* conditioning for new tasks? Multimodal conditioning (for VLAs) is wide open.
3. **Interpretability of generated weights.** Almost nobody has asked: are *hypernetwork-generated* LoRAs more or less interpretable than hand-trained ones? Does the shared hypernetwork induce a more structured, monosemantic weight basis? This is a genuinely novel, fundable question and your strongest potential contribution.
4. **Controllability / editing via the weight manifold.** If task vectors compose and ROME edits are rank-one, can you *steer* a hypernetwork's output in interpretable directions — generate-then-edit, or condition on an SAE feature you want to amplify?
5. **Memory-efficient meta-training as an enabler.** The systems stack (QLoRA + checkpointing + 8-bit/GaLore + offload) is what makes this PhD-on-one-GPU plausible rather than an H100 project. Demonstrating a credible 5090 recipe is itself a contribution to accessibility.

## The whitespace (where your project could plant a flag)
> **"Are hypernetwork-generated LoRA adapters mechanistically interpretable, and can interpretability tools both *explain* and *improve* weight generation?"**

Almost all current work treats the hypernetwork as a black box judged only by downstream accuracy. Pointing SAEs / task-arithmetic / DIT at its outputs — on a controlled, single-GPU testbed where you generate the weight distribution yourself — sits in a gap none of these 35 papers fully occupies.
