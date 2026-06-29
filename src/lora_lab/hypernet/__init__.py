"""Text-conditioned hypernetwork (Text-to-LoRA).

Generates a task-specific LoRA adapter for a frozen base model from a natural-
language task description, in a single forward pass. Trained two ways:
reconstruction of existing library LoRAs, and SFT (backprop a task loss through
the frozen base into the hypernetwork).

Modules:
  config     the run config (HyperConfig)
  model      text encoder + output heads + the LoRA generator
  apply      inject generated A/B factors as a live LoRA on a frozen base
  data       samplers (reconstruction targets / SFT example batches)
  train      the training loop, the two losses, model build + validation
  retrieval  nearest-neighbor baseline (for eval)
"""
