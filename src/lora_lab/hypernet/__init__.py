"""Phase 2 — the text-conditioned hypernetwork (T2L recipe).

Generates a task-specific LoRA adapter for a frozen base model from a natural-
language task description, in a single forward pass. Meta-trained by distilling
the Phase-1 library (reconstruction warmup) then SFT (backprop task loss through
the frozen base into the hypernetwork).

Modules:
  apply   inject generated A/B factors as a live LoRA on a frozen base (grads flow to A/B)
  model   the (stub, S1) hypernetwork: (task_emb, layer, module) -> LoRA A/B per target
  recon   reconstruction objective (relative Frobenius error on ΔW vs a target library LoRA; no base forward)
"""
