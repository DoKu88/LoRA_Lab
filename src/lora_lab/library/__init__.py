"""Phase 1 — LoRA library assembly.

Turns the public ``Lots-of-LoRAs`` release (per-task rank-16 LoRA adapters +
per-task SNI datasets, all for ``Mistral-7B-Instruct-v0.2``) into a versioned,
quality-gated library that does double duty as (a) the Phase-2 hypernetwork's
training data and (b) the Phase-3 hand-trained interp baseline.

Modules:
  manifest      build/load/validate the versioned task manifest + coverage report
  descriptions  extract each task's natural-language definition (the conditioning input)
  gate          Sprint-4 quality gate — adapter-vs-base eval on a single resident 7B
"""
