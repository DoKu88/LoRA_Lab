#!/usr/bin/env python
"""Phase 2 Sprint 1 — generate->apply->backprop smoke on a real tiny base (CPU).

Validates the plumbing against an actual transformer (default SmolLM2-135M):
generate a LoRA from a task embedding, inject it onto the frozen base, run a
masked-LM forward, and backprop the task loss into the hypernetwork. CPU-only by
design so it never contends with a GPU job.

    conda run -n lora_lab python scripts/phase2_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.hypernet.apply import LoRARegistry, lora_injected, target_specs  # noqa: E402
from lora_lab.hypernet.model import HyperLoRA  # noqa: E402
from lora_lab.hypernet.recon import reconstruction_loss  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-135M"
TARGETS = ["q_proj", "v_proj"]
TASK_DIM = 16


def main() -> int:
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[smoke] loading {BASE} on CPU")
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to("cpu").eval()
    for p in model.parameters():
        p.requires_grad_(False)  # base frozen

    specs = target_specs(model, TARGETS)
    print(f"[smoke] {len(specs)} target modules (e.g. {next(iter(specs))})")
    hyper = HyperLoRA(specs, task_dim=TASK_DIM, rank=8, alpha=16)
    print(f"[smoke] hypernetwork params: {hyper.num_params():,}")

    task_emb = torch.randn(TASK_DIM)

    # --- objective 1: reconstruction (no base forward) ---------------------
    # Unit-scale toy target so the loss/grad are clearly non-trivial (generated
    # ΔW is 0 at init, so this is a real residual to regress).
    target = {k: (torch.randn(8, specs[k][0]), torch.randn(specs[k][1], 8)) for k in specs}
    rloss = reconstruction_loss(hyper(task_emb), target, scaling=hyper.scaling)
    rloss.backward()
    g_recon = sum(p.grad.abs().sum().item() for p in hyper.parameters() if p.grad is not None)
    print(f"[smoke] reconstruction: loss={rloss.item():.5f}  grad_mass={g_recon:.4f}")
    assert g_recon > 0, "reconstruction produced no hypernetwork gradient"
    hyper.zero_grad(set_to_none=True)

    # --- objective 2: one-step SFT (task loss through the frozen base) ------
    batch = tok(["The capital of France is Paris.",
                 "Two plus two equals four."],
                return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    # no-op check: injected logits == base logits at init (base_b=0)
    with torch.no_grad():
        base_logits = model(**batch).logits
    reg = LoRARegistry()
    reg.set_adapter(hyper(task_emb))
    with lora_injected(model, TARGETS, reg, scaling=hyper.scaling):
        with torch.no_grad():
            noop_logits = model(**batch).logits
        assert torch.allclose(base_logits, noop_logits, atol=1e-4), "init adapter not a no-op"
        # now the real differentiable forward
        reg.set_adapter(hyper(task_emb))
        out = model(**batch, labels=labels)
    out.loss.backward()
    g_sft = sum(p.grad.abs().sum().item() for p in hyper.parameters() if p.grad is not None)
    base_grads = [p for p in model.parameters() if p.grad is not None]
    print(f"[smoke] SFT: loss={out.loss.item():.5f}  grad_mass={g_sft:.4f}  "
          f"base_params_with_grad={len(base_grads)}")
    assert g_sft > 0, "SFT produced no hypernetwork gradient"
    assert not base_grads, "base model received gradient (should be frozen)"

    print("[smoke] OK — generate->apply->backprop green on a real base (both objectives)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
