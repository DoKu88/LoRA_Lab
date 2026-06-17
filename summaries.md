# Paper Summaries — Hypernetworks, LoRA, VLAs & Mechanistic Interpretability

Literature review compiled 2026-06-16. PDFs are in [`pdfs/`](./pdfs/). Each entry is one paragraph. Three Anthropic interpretability articles are web-only (no arXiv PDF) and are linked instead.

**Reading order if you're new to this:** start with §1.1 (Text-to-LoRA) and §1.3 (HyperNetworks) for the core idea, then §2.1 (LoRA) and §2.2 (QLoRA) for the adapter + memory mechanics, then skim §4 for the interpretability lens.

---

## 1. Hypernetworks that Generate LoRA / Model Weights (the core idea)

### 1.1 Text-to-LoRA: Instant Transformer Adaption
*Charakorn, Cetin, Tang, Lange (Sakana AI), 2025 — ICML 2025 — [arXiv:2506.06105](https://arxiv.org/abs/2506.06105) — [`pdfs/1.1_Text-to-LoRA.pdf`](./pdfs/1.1_Text-to-LoRA.pdf)*
T2L (the paper that inspired this project) is a hypernetwork that generates a task-specific LoRA adapter for an LLM in a **single forward pass**, conditioned only on a natural-language description of the target task. It is meta-trained by distilling a library of pre-trained, task-specific LoRA adapters (GSM8K, ARC, BoolQ, …) into one shared network, compressing hundreds of adapters into a single set of weights. At inference it produces adapters for *unseen* tasks zero-shot from a text prompt, matching or approaching the performance of individually fine-tuned LoRAs while eliminating per-task training. This is the canonical reference for your exact target.

### 1.2 Doc-to-LoRA: Learning to Instantly Internalize Contexts
*Charakorn, Cetin, Uesaka, Lange (Sakana AI), 2026 — [arXiv:2602.15902](https://arxiv.org/abs/2602.15902) — [`pdfs/1.2_Doc-to-LoRA.pdf`](./pdfs/1.2_Doc-to-LoRA.pdf)*
The document-conditioned sibling of T2L (confirmed real — your term was correct). D2L is a hypernetwork that, given a document, generates a LoRA adapter that **internalizes that document's content into the base model's weights** in one forward pass — effectively meta-learned, approximate context distillation. The architecture pairs a Perceiver-style cross-attention encoder over per-layer token activations with output heads mapping latent queries to LoRA matrices. On needle-in-a-haystack tasks it reaches near-perfect accuracy on inputs ~5× longer than the base context window, while storing the "knowledge" in <50 MB of adapter weights versus a multi-GB KV-cache for in-context learning.

### 1.3 HyperNetworks (foundational)
*Ha, Dai, Le (Google Brain), 2016 — ICLR 2017 — [arXiv:1609.09106](https://arxiv.org/abs/1609.09106) — [`pdfs/1.3_HyperNetworks-Ha2016.pdf`](./pdfs/1.3_HyperNetworks-Ha2016.pdf)*
The origin of the whole paradigm: a small network ("hypernetwork") generates the weights of a larger target network, trained end-to-end by backprop. It frames this as a genotype→phenotype relationship and a relaxed form of weight-sharing, with *static* hypernets for CNNs and *dynamic* hypernets for RNNs/LSTMs, achieving near-SOTA on language modeling and sequence tasks with far fewer learnable parameters. Every "hypernetwork → LoRA" method is a descendant of this.

### 1.4 Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights
*Liang, Tang, Zhou, Zhao, Shi, et al. (NUS, UT Austin, Oxford, St. Gallen), 2025 — [arXiv:2506.16406](https://arxiv.org/abs/2506.16406) — [`pdfs/1.4_Drag-and-Drop-LLMs.pdf`](./pdfs/1.4_Drag-and-Drop-LLMs.pdf)*
The closest contemporaneous competitor to T2L. DnD maps a handful of *unlabeled task prompts* directly to LoRA weight updates with no per-task training: a lightweight text encoder distills each prompt batch into condition embeddings, and a cascaded hyper-convolutional decoder turns them into the full set of LoRA matrices. Trained on prompt–checkpoint pairs, it produces adapters in seconds with up to ~12,000× lower overhead than fine-tuning and strong cross-domain zero-shot generalization on reasoning, math, coding, and multimodal benchmarks.

### 1.5 Transformer² (Transformer-Squared): Self-Adaptive LLMs
*Sun, Cetin, Tang (Sakana AI / Institute of Science Tokyo), 2025 — ICLR 2025 — [arXiv:2501.06252](https://arxiv.org/abs/2501.06252) — [`pdfs/1.5_Transformer-Squared.pdf`](./pdfs/1.5_Transformer-Squared.pdf)*
Sakana's adjacent line on real-time adaptation. Instead of generating LoRA weights, Transformer² adapts at inference by **selectively scaling only the singular components** of the base weight matrices ("singular value fine-tuning"). A two-pass mechanism first dispatches/identifies task properties, then mixes RL-trained task-specific "expert" vectors to specialize behavior for the incoming prompt. Useful as a conceptual contrast: dynamic SVD expert-vector adaptation vs. hypernetwork weight generation.

### 1.6 HypeLoRA: Hyper-Network-Generated LoRA Adapters for Calibrated LM Fine-Tuning
*Trojan, Gębala, 2026 — [arXiv:2603.19278](https://arxiv.org/abs/2603.19278) — [`pdfs/1.6_HypeLoRA.pdf`](./pdfs/1.6_HypeLoRA.pdf)*
Replaces independently-trained per-layer LoRA A/B matrices with a single **shared hypernetwork that generates the LoRA factors conditioned on a layer embedding**, inducing structural coupling across layers, and studies this with an eye toward better-calibrated LLM fine-tuning. A direct, small-scale instance of the exact mechanism you want to build — worth reading closely for the layer-conditioning design.

### 1.7 HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis
*ByteDance et al., 2025 — [arXiv:2503.16944](https://arxiv.org/abs/2503.16944) — [`pdfs/1.7_HyperLoRA-Portrait.pdf`](./pdfs/1.7_HyperLoRA-Portrait.pdf)*
A vision-domain instance of the paradigm: an adaptive plug-in network encodes identity images and, instead of feeding tokens into attention (IP-Adapter style), **directly generates LoRA weights for a diffusion backbone**, combining LoRA's fidelity with the zero-shot capability of encoder/adapter methods for tuning-free personalized portraits. Shows the hypernetwork-→-LoRA idea generalizes beyond LLMs.

### 1.8 A Brief Review of Hypernetworks in Deep Learning (survey)
*Chauhan, Zhou, Lu, Molaei, Clifton (Oxford), 2023/24 — [arXiv:2306.06955](https://arxiv.org/abs/2306.06955) — [`pdfs/1.8_Hypernetworks-Review-Chauhan2023.pdf`](./pdfs/1.8_Hypernetworks-Review-Chauhan2023.pdf)*
The reference survey for situating LoRA-generating hypernets in the broader design space. Defines hypernets, proposes a taxonomy along five axes (inputs, outputs, input/output variability, architecture), and reviews applications across continual learning, transfer/zero-shot learning, pruning, uncertainty quantification, causal inference, NLP, and RL, plus open challenges. Good for vocabulary and for finding adjacent ideas.

### 1.9 HyperLoader: Integrating Hypernetwork-Based LoRA and Adapter Layers into Multi-Task Transformers
*Ortiz-Barajas, Gómez-Adorno, Solorio, 2024 — [arXiv:2407.01411](https://arxiv.org/abs/2407.01411) — [`pdfs/1.9_HyperLoader.pdf`](./pdfs/1.9_HyperLoader.pdf)*
Combines LoRA *and* adapter layers in a multi-task sequence-labelling setting, with a **hypernetwork that generates the PEFT-module weights conditioned on the task, the transformer layer, and the position within that layer.** This captures shared cross-task structure while encapsulating task-specific knowledge in the generated weights to reduce task interference, and it achieves the best average performance across tasks in both high- and low-resource regimes. A direct, concrete instance of task-and-layer-conditioned LoRA generation — useful as a design reference for your conditioning scheme.

### 1.10 From Instance Training to Instruction Learning: Task Adapters Generation from Instructions (TAGI)
*Liao, He, Xu, Zhang, Hao, Liu, Liu, Zhao, 2024 — NeurIPS 2024 — [arXiv:2406.12382](https://arxiv.org/abs/2406.12382) — [`pdfs/1.10_TaskAdaptersGeneration.pdf`](./pdfs/1.10_TaskAdaptersGeneration.pdf)*
TAGI produces a task-specific model **directly from a task's natural-language instruction**, with no per-task retraining: a hypernetwork generates the task adapter (LoRA-style) parameters, then knowledge distillation aligns this instruction-based model with an instance-trained model across predictions, logits/confidence, *and* model parameters. Training is two-phase — hypernetwork "preparation" to learn adapter generation, then distillation-based refinement — and on Super-NaturalInstructions and P3 it matches or beats meta-learning baselines at substantially lower compute. Conceptually the closest predecessor to Text-to-LoRA's instruction→weights idea, and its parameter-level distillation is a recipe worth copying.

---

## 2. LoRA / PEFT Methods & Memory-Efficient Training (the engineering)

### 2.1 LoRA: Low-Rank Adaptation of Large Language Models
*Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen, 2021 — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) — [`pdfs/2.1_LoRA.pdf`](./pdfs/2.1_LoRA.pdf)*
Freezes pretrained weights and injects trainable rank-decomposition matrices (a low-rank product BA) into each Transformer layer, so only the small matrices are updated. Cuts trainable parameters by up to 10,000× and optimizer-state memory ~3× vs. full fine-tuning, with **no added inference latency** since BA merges into the base weights. Matches or beats full fine-tuning on RoBERTa/DeBERTa/GPT-2/GPT-3. The foundation the entire review rests on.

### 2.2 QLoRA: Efficient Finetuning of Quantized LLMs
*Dettmers, Pagnoni, Holtzman, Zettlemoyer, 2023 — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — [`pdfs/2.2_QLoRA.pdf`](./pdfs/2.2_QLoRA.pdf)*
**The canonical single-GPU recipe.** Backpropagates gradients through a frozen 4-bit quantized model into LoRA adapters, fine-tuning a 65B model on one 48GB GPU at full 16-bit quality. Three ideas: 4-bit NormalFloat (NF4) for normally-distributed weights, Double Quantization (quantizing the quantization constants), and Paged Optimizers (NVIDIA unified memory to absorb gradient-checkpointing spikes). Directly relevant to fitting your hypernetwork+base-model training in 32 GB.

### 2.3 DoRA: Weight-Decomposed Low-Rank Adaptation
*Liu, Wang, Yin, Molchanov, Wang, Cheng, Chen, 2024 — ICML 2024 Oral — [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) — [`pdfs/2.3_DoRA.pdf`](./pdfs/2.3_DoRA.pdf)*
Decomposes pretrained weights into **magnitude and direction**, applies LoRA only to the directional update, and trains the magnitude separately. This makes LoRA's learning dynamics resemble full fine-tuning, closing the accuracy gap at no extra inference cost, and consistently beats LoRA on commonsense reasoning and vision-language tasks (LLaMA, LLaVA, VL-BART). A drop-in upgrade if your generated adapters underperform plain LoRA.

### 2.4 VeRA: Vector-based Random Matrix Adaptation
*Kopiczko, Blankevoort, Asano, 2023 — [arXiv:2310.11454](https://arxiv.org/abs/2310.11454) — [`pdfs/2.4_VeRA.pdf`](./pdfs/2.4_VeRA.pdf)*
Shares a **single pair of frozen, random low-rank matrices across all layers** and trains only small per-layer scaling vectors, cutting trainable parameters ~10× below LoRA at comparable accuracy. Especially relevant to a hypernetwork: it suggests the network may only need to emit small per-layer scaling vectors over a shared random basis, drastically shrinking the prediction target.

### 2.5 The Power of Scale for Parameter-Efficient Prompt Tuning
*Lester, Al-Rfou, Constant, 2021 — EMNLP 2021 — [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) — [`pdfs/2.5_Prompt-Tuning.pdf`](./pdfs/2.5_Prompt-Tuning.pdf)*
Learns a few continuous "soft prompt" embeddings prepended to the input while the model stays frozen, and shows that as scale grows toward billions of parameters this becomes competitive with full fine-tuning — tuning only thousands of parameters per task. A distinct, ultra-light PEFT family and a useful baseline/contrast to weight-space adaptation.

### 2.6 ZeRO-Offload: Democratizing Billion-Scale Model Training
*Ren, Rajbhandari, Aminabadi, Ruwase, Yang, Zhang, Li, He, 2021 — USENIX ATC 2021 — [arXiv:2101.06840](https://arxiv.org/abs/2101.06840) — [`pdfs/2.6_ZeRO-Offload.pdf`](./pdfs/2.6_ZeRO-Offload.pdf)*
Offloads optimizer states, gradients, and Adam computation from GPU to CPU while keeping forward/backward on-GPU, enabling 13B-parameter training on a single GPU. Carefully minimizes data movement and CPU compute on the critical path and ships a CPU-optimized Adam. This is the systems counterpart to "load state as you backprop," and underpins the offload paths you'll likely need on a 5090.

### 2.7 Training Deep Nets with Sublinear Memory Cost (Gradient Checkpointing)
*Chen, Xu, Zhang, Guestrin, 2016 — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174) — [`pdfs/2.7_Gradient-Checkpointing.pdf`](./pdfs/2.7_Gradient-Checkpointing.pdf)*
Introduces gradient (activation) checkpointing: store only a √n-sized subset of activations on the forward pass and **recompute the rest during backprop**, cutting activation memory to O(√n) for one extra forward pass. The foundational paper for the technique you named — the cheapest big lever for fitting activations in 32 GB.

### 2.8 8-bit Optimizers via Block-wise Quantization
*Dettmers, Lewis, Shleifer, Zettlemoyer, 2021 — ICLR 2022 — [arXiv:2110.02861](https://arxiv.org/abs/2110.02861) — [`pdfs/2.8_8bit-Optimizers.pdf`](./pdfs/2.8_8bit-Optimizers.pdf)*
Stores optimizer state (Adam moments) in 8 bits while matching 32-bit performance, via block-wise quantization (robust to outliers), dynamic non-linear quantization, and a stable embedding layer. Released as a two-line drop-in in `bitsandbytes`, it removes one of training's largest memory consumers and underpins most consumer-GPU fine-tuning tooling.

### 2.9 GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
*Zhao, Zhang, Chen, Wang, Anandkumar, Tian, 2024 — ICML 2024 Oral — [arXiv:2403.03507](https://arxiv.org/abs/2403.03507) — [`pdfs/2.9_GaLore.pdf`](./pdfs/2.9_GaLore.pdf)*
Projects gradients onto a low-rank subspace (exploiting the inherent low-rank structure of LLM gradients) and keeps optimizer states in that compressed subspace, allowing **full-parameter** learning with far less memory — reducing optimizer-state memory up to 65.5% and enabling 7B LLaMA pretraining on a single 24 GB consumer GPU. Unlike LoRA it doesn't restrict the search to a low-rank weight subspace, so it can match full-rank quality. Relevant if you ever want to train the *base* model rather than just adapters.

### 2.10 PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
*Zhao, Gu, Varma, Luo, Huang, Xu, Wright, et al. (Meta), 2023 — VLDB 2023 — [arXiv:2304.11277](https://arxiv.org/abs/2304.11277) — [`pdfs/2.10_PyTorch-FSDP.pdf`](./pdfs/2.10_PyTorch-FSDP.pdf)*
Production design of FSDP: shards parameters, gradients, and optimizer states, and **materializes full parameters only per-layer as needed** during forward/backward, then re-shards — the literal "load layers as you go" pattern you described. Co-designed with PyTorch internals, with CPU offload, compute/communication overlap, and mixed precision, achieving near-DDP throughput on much larger models. (Single-GPU you'd use it mainly for its offload/per-layer-materialization machinery.)

### Full-parameter fine-tuning under tight memory (Phase 0.5 candidates — no local PDFs yet)

*Added for the **Phase 0.5** spike ("can we full-finetune Mistral-7B on 32 GB VRAM + 32 GB RAM?", see `notes.md` §C2). These make **full-parameter** training fit on a small GPU by attacking the gradient/optimizer-state pools rather than freezing the base. PDFs are not yet in `pdfs/` — links are to arXiv; treat as a reading queue.*

### 2.11 Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients
*Zhang, Liu, Hu, Lee, Wang, et al., 2024 — [arXiv:2407.08296](https://arxiv.org/abs/2407.08296)*
Extends GaLore (§2.9) by **quantizing both the weights (INT4/INT8) and the low-rank projection matrices**, and skipping projection updates in layers whose gradient subspace is stable ("layer-adaptive"). Pushes full-parameter pretraining/fine-tuning into a smaller memory envelope than GaLore (e.g. 7B regimes on ~16 GB-class budgets), at some quality/throughput cost. The most aggressive "full-parameter on a consumer GPU" option to benchmark first in Phase 0.5.

### 2.12 LOMO / AdaLOMO: Full-Parameter Fine-Tuning with Limited Resources
*Lv, Yang, Liu, Gao, Guo, Qiu, 2023 — LOMO [arXiv:2306.09782](https://arxiv.org/abs/2306.09782); AdaLOMO [arXiv:2310.10195](https://arxiv.org/abs/2310.10195)*
**LOMO** ("LOw-Memory Optimization") **fuses the gradient computation and the parameter update into one step**, so full gradients and optimizer state never need to be materialized — collapsing the footprint toward SGD-like (weights + activations only) and enabling full-parameter tuning of large models on far less VRAM. **AdaLOMO** adds an Adam-style adaptive per-parameter learning rate while keeping the fused, low-memory update, recovering most of the optimization quality LOMO's vanilla SGD-like rule gives up. Directly targets the Phase 0.5 question; the main cost is throughput and tuning sensitivity.

### 2.13 BAdam: A Memory-Efficient Full-Parameter Optimization Method
*Luo, Hu, Zhang, Yuan, Sun, Yin, Wei, Zhang, 2024 — NeurIPS 2024 — [arXiv:2404.02827](https://arxiv.org/abs/2404.02827)*
Applies **block-coordinate descent over the transformer's blocks**: at any moment only *one* block is "active" and carries gradients + Adam optimizer state, while the rest are frozen; the active block cycles across the network so all parameters are eventually updated (full-parameter over the run). This slashes the gradient/optimizer-state memory roughly by the block count, letting a 7B model full-finetune on a single consumer GPU. A clean middle ground between LoRA and naive full FT for the Phase 0.5 benchmark.

### 2.14 MeZO: Fine-Tuning Language Models with Just Forward Passes
*Malladi, Gao, Nichani, Damian, Lee, Chen, Arora, 2023 — NeurIPS 2023 — [arXiv:2305.17333](https://arxiv.org/abs/2305.17333)*
A **memory-efficient zeroth-order optimizer**: estimates gradients from *forward passes only* (perturb weights, compare losses) so training memory equals **inference memory** — no backprop, no stored activations or optimizer state. Can full-finetune very large models on tight hardware, but converges slowly and noisily and typically needs prompts/many steps. The "last resort" option in Phase 0.5 — relevant if every backprop-based trick still OOMs.

### 2.6′ ZeRO-Infinity (NVMe offload, extends §2.6)
*Rajbhandari, Ruwase, Yang, He, 2021 — SC '21 — [arXiv:2104.07857](https://arxiv.org/abs/2104.07857)*
The successor to ZeRO-Offload (§2.6) that adds **NVMe (SSD) offload** of parameters, gradients, and optimizer states, plus bandwidth-centric partitioning — enabling models far larger than GPU+CPU memory by streaming state off disk. Listed here because it's the **only offload path that survives this machine's 32 GB RAM ceiling** for 7B full FT; the trade-off is heavy dependence on a fast SSD and large throughput penalties. A Phase 0.5 fallback to measure, not a default.

---

## 3. Vision-Language-Action Models & Efficient Adaptation

### 3.1 OpenVLA: An Open-Source Vision-Language-Action Model
*Kim, Pertsch, Karamcheti, Xiao, Balakrishna, … Finn, 2024 — [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) — [`pdfs/3.1_OpenVLA.pdf`](./pdfs/3.1_OpenVLA.pdf)*
A 7B open-source VLA on a Llama-2 backbone with a fused DINOv2 + SigLIP visual encoder, trained on 970k Open X-Embodiment demos; it emits discretized robot actions as language tokens and controls multiple embodiments, outperforming the closed 55B RT-2-X by 16.5% absolute despite being 7× smaller. Crucially for you, OpenVLA explicitly demonstrates **LoRA (and quantized) fine-tuning** for cheap adaptation to new robots — the canonical reference for PEFT on VLAs.

### 3.2 RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
*Brohan, Brown, Carbajal, Chebotar, … (Google DeepMind), 2023 — [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) — [`pdfs/3.2_RT-2.pdf`](./pdfs/3.2_RT-2.pdf)*
Co-fine-tunes large internet-pretrained VLMs on both web vision-language tasks and robot trajectories, representing actions as text tokens so one model jointly learns perception, language, and control. This transfer yields strong generalization to novel objects and emergent semantic reasoning, and establishes the VLA paradigm of **control as token prediction on top of a VLM**.

### 3.3 π0: A Vision-Language-Action Flow Model for General Robot Control
*Black, Brown, Driess, … Finn, et al. (Physical Intelligence), 2024 — [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) — [`pdfs/3.3_pi0.pdf`](./pdfs/3.3_pi0.pdf)*
A generalist policy pairing a pretrained VLM backbone with a separate "action expert" that generates **continuous action chunks via flow matching** (rather than discretized tokens), trained across single-arm, dual-arm, and mobile manipulators for dexterous high-frequency tasks like laundry folding. Represents the SOTA continuous-action VLA design and a key contrast to the token-based VLAs LoRA is usually applied to.

### 3.4 RT-1: Robotics Transformer for Real-World Control at Scale
*Brohan, Brown, Carbajal, Chebotar, Dabis, … Finn (Google), 2022 — [arXiv:2212.06817](https://arxiv.org/abs/2212.06817) — [`pdfs/3.4_RT-1.pdf`](./pdfs/3.4_RT-1.pdf)*
A scalable Transformer policy trained on 130k real-world episodes across 700+ tasks from 13 robots over 17 months; it tokenizes images and instructions and outputs discretized actions, showing high-capacity task-agnostic training yields strong zero-shot generalization. The foundational "robotics transformer" that motivated the VLA line and the Open X-Embodiment data underlying later adapted models.

### 3.5 HyperVLA: Efficient Inference in VLAs via Hypernetworks
*Xiong, Li, Wang, Jackson, Foerster, Whiteson (Oxford), 2025 — [arXiv:2510.04898](https://arxiv.org/abs/2510.04898) — [`pdfs/3.5_HyperVLA.pdf`](./pdfs/3.5_HyperVLA.pdf)*
**The most on-target VLA paper for you.** A hypernetwork generates a small, task-specific policy at test time, so only a compact policy is active at inference while full multi-task capacity is retained during training — cutting activated parameters ~90× and accelerating inference ~120× vs. monolithic VLAs, while matching or exceeding zero-/few-shot success. Key tricks: vision-foundation-model priors, hypernetwork normalization, and a tailored action-generation strategy. This is the hypernetwork-→-policy analog of T2L for robotics.

### 3.6 Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OFT)
*Kim, Finn, Liang, 2025 — [arXiv:2502.19645](https://arxiv.org/abs/2502.19645) — [`pdfs/3.6_FineTuningVLA-OFT.pdf`](./pdfs/3.6_FineTuningVLA-OFT.pdf)*
A systematic study of how to fine-tune VLAs (on OpenVLA), proposing Optimized Fine-Tuning (parallel decoding, action chunking, continuous action representation, L1 regression). It analyzes **when LoRA suffices vs. when full fine-tuning is needed** — finding LoRA effective for single-arm low-frequency control but limited for high-frequency bimanual robots — directly informing where PEFT is viable for robot policies.

### 3.7 A Survey on Efficient Vision-Language-Action Models
*Yu, Wang, Zeng, Zhang, Zhang, Wang, Gao, Song, Sebe, Shen, 2025 — [arXiv:2510.24795](https://arxiv.org/abs/2510.24795) — [`pdfs/3.7_EfficientVLA-Survey.pdf`](./pdfs/3.7_EfficientVLA-Survey.pdf)*
A comprehensive review of efficiency across the VLA pipeline — efficient model design (architectures + compression), efficient training, efficient data collection — taxonomizing methods that make multi-billion-parameter VLAs cheaper to train, adapt, and deploy. Your map of where LoRA/PEFT and hypernetwork adaptation sit in the broader efficient-VLA landscape.

---

## 4. Mechanistic Interpretability (the dedicated thread)

### Core mech-interp

### 4.1 Toy Models of Superposition
*Elhage, Hume, Olsson, Schiefer, Henighan, et al. (Anthropic), 2022 — [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) — [`pdfs/4.1_ToyModelsOfSuperposition.pdf`](./pdfs/4.1_ToyModelsOfSuperposition.pdf)*
Uses small ReLU nets on synthetic sparse-feature data to study **superposition** — representing more features than dimensions by storing them as non-orthogonal directions, at the cost of interference that nonlinearities clean up. Demonstrates a phase change between dedicated-neuron and superposition regimes and links to adversarial vulnerability. Establishes superposition/polysemanticity as the central obstacle to neuron-level interpretation — and the reason a *low-rank* weight edit can have broadly distributed effects.

### 4.2 A Mathematical Framework for Transformer Circuits
*Elhage, Nanda, Olsson, Henighan, Joseph, et al. (Anthropic), 2021 — web-only: [transformer-circuits.pub/2021/framework](https://transformer-circuits.pub/2021/framework/index.html)*
A linear-algebraic framework for reverse-engineering attention-only transformers: each head decomposes into an independent QK (attention-pattern) circuit and an OV (token-effect) circuit on the residual stream, so the model becomes a sum of interpretable token→logit paths. Introduces **induction heads** as a key in-context-learning mechanism. Defines the residual-stream/circuits vocabulary every weight-space analysis uses. *(No PDF — read on the web.)*

### 4.3 Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
*Bricken, Templeton, Batson, et al. (Anthropic), 2023 — web-only: [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features/index.html)*
Applies a **sparse autoencoder** to a one-layer transformer's activations, decomposing a 512-neuron MLP into 4000+ sparse, largely monosemantic features (DNA, legal text, HTTP, Hebrew, …). Argues features, not neurons, are the right unit of analysis and that they're largely universal across models. The dictionary-learning lens here is exactly what later work points at LoRA deltas. *(No PDF — read on the web.)*

### 4.4 Sparse Autoencoders Find Highly Interpretable Features in Language Models
*Cunningham, Ewart, Riggs, Huben, Sharkey, 2023 — ICLR 2024 — [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) — [`pdfs/4.4_SAE_HighlyInterpretableFeatures_Cunningham.pdf`](./pdfs/4.4_SAE_HighlyInterpretableFeatures_Cunningham.pdf)*
The peer-reviewed, citable anchor for SAE interpretability (concurrent with Anthropic's work): SAEs trained to reconstruct LM activations recover sparse feature directions more interpretable and monosemantic than PCA or raw neurons, with features supporting causal interventions. The methodological basis for SAE analyses of fine-tuned/LoRA weight spaces.

### 4.5 Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
*Templeton, Conerly, Marcus, et al. (Anthropic), 2024 — web-only: [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)*
Scales SAEs to a production model (Claude 3 Sonnet) with up to 34M features on the middle-layer residual stream, recovering multilingual/multimodal/abstract features (including safety-relevant ones: deception, power-seeking, sycophancy) that **causally steer behavior** consistently with their interpretations. Validates dictionary learning at frontier scale, not just toy transformers. *(No PDF — read on the web.)*

### 4.6 Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small
*Wang, Variengien, Conmy, Shlegeris, Steinhardt, 2022 — ICLR 2023 — [arXiv:2211.00593](https://arxiv.org/abs/2211.00593) — [`pdfs/4.6_IOI_Circuit_Wang.pdf`](./pdfs/4.6_IOI_Circuit_Wang.pdf)*
Reverse-engineers how GPT-2 small does indirect-object identification, identifying a circuit of 26 attention heads in 7 functional classes (name-mover, S-inhibition, duplicate-token, …) via causal interventions like path patching, and introduces faithfulness/completeness/minimality criteria for circuit claims. The circuit-discovery methodology that lets you ask *which* weights/heads a generated adapter actually modifies.

### Bridge: interpretability ↔ weights / fine-tuning / LoRA

### 4.7 Locating and Editing Factual Associations in GPT (ROME)
*Meng, Bau, Andonian, Belinkov, 2022 — NeurIPS 2022 — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) — [`pdfs/4.7_ROME_LocatingEditingFactualAssociations_Meng.pdf`](./pdfs/4.7_ROME_LocatingEditingFactualAssociations_Meng.pdf)*
Uses **causal tracing** to localize where GPT stores facts (middle-layer MLP feed-forward modules over the subject token), then introduces Rank-One Model Editing — treating MLP layers as linear key-value memories and applying a **rank-one weight update** to insert/change a fact. The canonical bridge from interpretability to weight editing, and conceptually adjacent to how a low-rank adapter modifies behavior.

### 4.8 Editing Models with Task Arithmetic (Task Vectors)
*Ilharco, Ribeiro, Wortsman, Gururangan, Schmidt, Hajishirzi, Farhadi, 2022 — ICLR 2023 — [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) — [`pdfs/4.8_TaskArithmetic_Ilharco.pdf`](./pdfs/4.8_TaskArithmetic_Ilharco.pdf)*
Defines a **task vector** as the weight difference between a fine-tuned model and its pretrained init — a direction in weight space — and shows these vectors can be negated (to unlearn), added (to multi-task), and combined by analogy, steering behavior predictably across vision and NLP. The foundational result that fine-tuning deltas (and thus LoRA adapters and *generated* weights) are structured, composable objects.

### 4.9 Learning to Interpret Weight Differences in Language Models
*Goel, Kim, Shavit, Wang, 2025 — [arXiv:2510.05092](https://arxiv.org/abs/2510.05092) — [`pdfs/4.9_LearningToInterpretWeightDifferences_Goel.pdf`](./pdfs/4.9_LearningToInterpretWeightDifferences_Goel.pdf)*
Tackles the opacity of fine-tuning weight diffs (including LoRA adapters) with **Diff Interpretation Tuning**: train a reusable "DIT-adapter" on synthetic labeled weight diffs so that, applied to a compatible fine-tuned model, it makes the model **describe its own modifications in natural language** — revealing hidden behaviors and newly acquired knowledge. A direct interpretability-of-adapters paper.

### 4.10 Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis
*Prasanth K K, 2026 — [arXiv:2605.28896](https://arxiv.org/abs/2605.28896) — [`pdfs/4.10_FeatureGeometryOfLoRAAdapters_SAE.pdf`](./pdfs/4.10_FeatureGeometryOfLoRAAdapters_SAE.pdf)*
Trains adapter-specific SAEs on Gemma-2-9B (ranks 4/8/16/32) and compares them to pretrained feature dictionaries via cosine similarity, principal angles, and CKA, finding LoRA-induced feature dictionaries are **near-orthogonal to base-model features** (principal angles ~74°, cosine ~0.07) — i.e., LoRA creates representational structure existing base-model interpretability tools don't capture. The most direct SAE-on-LoRA bridge, but note: single-author, very recent preprint — treat as an emerging-work data point, not settled.

---

## 5. Hypernetworks & LoRA Generation for Diffusion Models (parked — revisit after the LLM study)

*The same "generate the weights instead of training them" idea, in the image domain. This cluster is deferred: the project is an LLM study first (see `notes.md` §C1). Note **§1.7 (HyperLoRA for portraits) also belongs here** — it's listed under §1 for the hypernetwork-→-LoRA lineage but is a diffusion method; cross-referenced below as §5.4.*

### 5.1 HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models (canonical)
*Ruiz, Li, Jampani, Wei, Hou, Pritch, Wadhwa, Rubinstein, Aberman (Google), 2023 — [arXiv:2307.06949](https://arxiv.org/abs/2307.06949) — [`pdfs/5.1_HyperDreamBooth.pdf`](./pdfs/5.1_HyperDreamBooth.pdf)*
The defining diffusion analog of Text-to-LoRA: a hypernetwork that, from a **single face image**, predicts personalized low-rank weight residuals for a Stable Diffusion model. Introduces "Lightweight DreamBooth" (LiDB), a low-rank-within-low-rank decomposition shrinking the personalized delta to ~0.1% of full DreamBooth (~100 KB), and follows the hypernetwork's initial prediction with fast rank-relaxed fine-tuning to recover subject fidelity — ~25× faster than DreamBooth while preserving model knowledge and editability. The clearest "hypernetwork → low-rank weights" precedent outside LLMs.

### 5.2 Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models (E4T)
*Gal, Arar, Atzmon, Bermano, Chechik, Cohen-Or, 2023 — [arXiv:2302.12228](https://arxiv.org/abs/2302.12228) — [`pdfs/5.2_E4T-EncoderTuning.pdf`](./pdfs/5.2_E4T-EncoderTuning.pdf)*
An encoder that, from one image of a concept, predicts a word-embedding **plus a set of weight offsets** to the diffusion model in a single forward pass, giving a domain-aware initialization; a regularizer keeps the predicted updates small and close to pretrained weights (preserving editability), after which a brief (~5–15 s) fine-tune locks in the instance. Trained once per domain (faces, cats, art). An early "encoder predicts weights" approach and a conceptual predecessor to LoRA-prediction hypernetworks.

### 5.3 Neural Network Diffusion (p-diff) — *contrast paper*
*Wang, Tang, Zeng, Yin, Xu, Zhou, Zang, Darrell, Liu, You, 2024 — [arXiv:2402.13144](https://arxiv.org/abs/2402.13144) — [`pdfs/5.3_NeuralNetworkDiffusion.pdf`](./pdfs/5.3_NeuralNetworkDiffusion.pdf)*
Uses a standard latent diffusion model to **generate neural-network parameters** (not images): an autoencoder compresses trained weights into a latent, and a diffusion model synthesizes new latents that decode into full, functional parameter sets matching or exceeding the SGD-trained originals (and provably distinct from them, not memorization). Included as a deliberate **contrast** to hypernetworks — it generates weights via an *iterative diffusion process over weights* rather than a single feed-forward conditioned prediction, framing the broader "generate weights, don't train them" paradigm your project sits inside.

### 5.4 HyperLoRA: Parameter-Efficient Adaptive Generation for Portrait Synthesis *(cross-ref → see §1.7)*
*ByteDance et al., 2025 — [arXiv:2503.16944](https://arxiv.org/abs/2503.16944) — [`pdfs/1.7_HyperLoRA-Portrait.pdf`](./pdfs/1.7_HyperLoRA-Portrait.pdf) — full summary at §1.7.*
A plug-in network encodes identity images and **directly generates LoRA weights for a diffusion backbone** (rather than feeding tokens into attention), giving tuning-free personalized portraits. Filed under §1 for lineage, but squarely a diffusion-cluster method.

### 5.5 DiffLoRA: Generating Personalized Low-Rank Adaptation Weights with Diffusion Models
*Wu, Shi, Wei, Sun, Yang, Shen, 2024 — [arXiv:2408.06740](https://arxiv.org/abs/2408.06740) — [`pdfs/5.5_DiffLoRA.pdf`](./pdfs/5.5_DiffLoRA.pdf)*
Bridges the two paradigms above: uses a **diffusion model *as* the hypernetwork** to predict personalized LoRA weights for a text-to-image model directly from reference portraits, no per-subject fine-tuning. Combines a LoRA-weight autoencoder (compressing LoRA params into a latent) with a Mixture-of-Image-Features conditioning module that injects identity into a diffusion transformer operating over LoRA latents; the generated LoRA merges into the base model for identity-consistent images. The most on-the-nose "generate a LoRA in one pass" diffusion paper.

### 5.6 Domain-Agnostic Tuning-Encoder for Fast Personalization of Text-to-Image Models
*Arar, Gal, Atzmon, Chechik, Cohen-Or, Shamir, Bermano, 2023 — [arXiv:2307.06925](https://arxiv.org/abs/2307.06925) — [`pdfs/5.6_DomainAgnosticEncoder.pdf`](./pdfs/5.6_DomainAgnosticEncoder.pdf)*
Generalizes E4T into a **single domain-agnostic encoder** that personalizes across arbitrary concepts (not one encoder per domain): from a single image it predicts a textual embedding plus **low-rank attention-weight offsets**, with a regularization scheme that removes the need for domain-specific training or per-concept optimization. Extends the encoder-predicts-LoRA-deltas line to open-domain subjects.

---

## Coverage notes
- **42 sources with PDFs/links** (39 downloaded PDFs + 3 web-only Anthropic articles §4.2/§4.3/§4.5), **plus 6 reading-queue additions** for the Phase 0.5 full-FT spike (§2.11 Q-GaLore, §2.12 LOMO/AdaLOMO, §2.13 BAdam, §2.14 MeZO, §2.6′ ZeRO-Infinity) — these have no local PDFs yet; download before citing.
- The user's named techniques are all covered: quantization (§2.2, §2.8), gradient checkpointing (§2.7), and layer-wise/offloaded backprop (§2.6 ZeRO-Offload, §2.6′ ZeRO-Infinity, §2.10 FSDP).
- **Full-parameter-on-a-budget cluster (§2.9, §2.11–§2.14):** GaLore, Q-GaLore, LOMO/AdaLOMO, BAdam, MeZO — the candidate techniques for getting Mistral-7B to full-finetune on 32 GB (see `notes.md` §C2 Phase 0.5).
- Provenance flags: §4.10 is a single-author 2026 preprint; §1.6 and §1.2 are very recent. Weight their claims accordingly.

## Parked scope (LLM-first)
The project starts as an **LLM study** (base-model choice and timeline in `notes.md` §C1–C2). Two clusters here are intentionally deferred until that study ships:
- **§3 — VLAs / robot policies.** The second testbed, harder (multimodal conditioning, high-frequency control). Revisit once the LLM interp result is in hand.
- **§5 — Diffusion models.** The vision-domain version of the same hypernetwork-→-LoRA idea; useful as precedent and a possible third domain, but not on the critical path now.
