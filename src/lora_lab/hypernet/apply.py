"""Inject hypernetwork-generated LoRA factors as a live adapter on a frozen base.

The key Phase-2 plumbing requirement (Sprint 1): apply a generated ΔW = scaling·(B·A)
to the base's targeted Linear layers such that (a) the base forward uses ΔW and
(b) gradients flow back into A/B — which are *outputs of the hypernetwork*, not
registered base parameters. We do this by wrapping each target ``nn.Linear`` in a
module that reads the current (A, B) from a shared registry the hypernetwork
writes each step. The base weights stay frozen; only A/B (hence the hypernetwork)
receive gradient.

Unlike PEFT (which registers A/B as the model's own parameters), here A/B are
*external* graph tensors, so a fresh adapter can be generated per task on every
step with no parameter surgery.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRARegistry(dict):
    """Maps target module key -> (A, B). A: (r, in), B: (out, r). Set per step."""

    def set_adapter(self, adapter: dict) -> None:
        self.clear()
        self.update(adapter)


class _LoRAWrapper(nn.Module):
    """Wrap a frozen Linear; add scaling·(x·Aᵀ·Bᵀ) when the registry has this key."""

    def __init__(self, base: nn.Linear, key: str, registry: LoRARegistry, scaling: float):
        super().__init__()
        self.base = base
        self.key = key
        self.registry = registry
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        ab = self.registry.get(self.key)
        if ab is not None:
            a, b = ab  # a:(r,in)  b:(out,r)
            # x:(...,in) -> (...,r) via Aᵀ -> (...,out) via Bᵀ
            out = out + self.scaling * F.linear(F.linear(x, a), b)
        return out


def _last_attr(model: nn.Module, dotted: str):
    parent = model.get_submodule(dotted.rsplit(".", 1)[0]) if "." in dotted else model
    return parent, dotted.rsplit(".", 1)[-1]


def iter_target_linears(model: nn.Module, target_modules):
    """Yield (qualified_name, Linear) for every Linear whose leaf name matches."""
    targets = set(target_modules)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            yield name, mod


def target_specs(model: nn.Module, target_modules) -> dict[str, tuple[int, int]]:
    """{key: (in_features, out_features)} for every targeted Linear — the shape
    contract the hypernetwork's output heads must satisfy."""
    return {n: (m.in_features, m.out_features) for n, m in iter_target_linears(model, target_modules)}


def inject(model: nn.Module, target_modules, registry: LoRARegistry, *, scaling: float):
    """Replace each target Linear with a wrapper reading from ``registry``.

    Returns a list of (key, original_linear) so the wrapping can be undone.
    """
    handles = []
    for name, mod in list(iter_target_linears(model, target_modules)):
        parent, attr = _last_attr(model, name)
        setattr(parent, attr, _LoRAWrapper(mod, name, registry, scaling))
        handles.append((name, mod))
    return handles


def remove(model: nn.Module, handles) -> None:
    """Restore the original Linears (undo :func:`inject`)."""
    for name, original in handles:
        parent, attr = _last_attr(model, name)
        setattr(parent, attr, original)


@contextmanager
def lora_injected(model: nn.Module, target_modules, registry: LoRARegistry, *, scaling: float):
    """Context manager: inject for the duration, restore on exit."""
    handles = inject(model, target_modules, registry, scaling=scaling)
    try:
        yield handles
    finally:
        remove(model, handles)
