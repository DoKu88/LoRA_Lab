"""Trainable-parameter accounting (one of the comparison columns)."""

from __future__ import annotations


def count_parameters(model) -> dict[str, float]:
    """Return trainable / total parameter counts and the trainable %.

    For 4-bit (QLoRA) bases the frozen weights are stored packed, so ``total``
    reflects the packed element count — ``trainable`` (the LoRA adapter) is the
    number we actually compare across methods, and it is unaffected by packing.
    """
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = (100.0 * trainable / total) if total else 0.0
    return {
        "trainable_params": int(trainable),
        "total_params": int(total),
        "pct_params": round(pct, 6),
    }
