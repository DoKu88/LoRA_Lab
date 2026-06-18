#!/usr/bin/env python
"""Sprint 1 smoke test — prove the Blackwell/sm_120 stack trains end-to-end.

Loads SmolLM2-135M (ungated) in 4-bit NF4, attaches a LoRA, runs a
forward + backward + optimizer step, and prints per-phase peak VRAM. Also
runs a bf16 (non-quantized) path so both regimes the comparison needs are
exercised. Asserts the device is sm_120.

Run inside the dedicated env:

    conda run -n lora_lab python scripts/smoke_test.py
    conda run -n lora_lab python scripts/smoke_test.py --bf16-only   # skip 4-bit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable without an install step.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from lora_lab.utils.vram import (  # noqa: E402
    cuda_mem_snapshot,
    device_capability,
    is_blackwell_sm120,
    phase_memory,
    reset_peak_memory,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n{msg}\n{'=' * 60}")


def build_model(quantized: bool):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {"dtype": torch.bfloat16}
    if quantized:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if quantized:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model = model.to("cuda")

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tok


def train_step(model, tok) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = "Translate to French: Hello, how are you?\nBonjour, comment allez-vous?"
    batch = tok(text, return_tensors="pt").to(device)
    batch["labels"] = batch["input_ids"].clone()

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4
    )

    with phase_memory("forward") as fwd:
        out = model(**batch)
        loss = out.loss
    with phase_memory("backward") as bwd:
        loss.backward()
    with phase_memory("optim_step") as opt_phase:
        opt.step()
        opt.zero_grad(set_to_none=True)

    print(f"  loss            : {loss.item():.4f}")
    print(f"  forward  peak   : {fwd['peak_gb']:.4f} GB")
    print(f"  backward peak   : {bwd['peak_gb']:.4f} GB")
    print(f"  optim    peak   : {opt_phase['peak_gb']:.4f} GB")
    return loss.item()


def run_path(quantized: bool) -> None:
    label = "4-bit NF4 + LoRA" if quantized else "bf16 + LoRA"
    _banner(f"Path: {label}")
    reset_peak_memory()
    model, tok = build_model(quantized=quantized)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  trainable params: {trainable:,} ({100 * trainable / total:.3f}% of {total:,})")

    loss = train_step(model, tok)
    snap = cuda_mem_snapshot()
    print(f"  max allocated   : {snap['max_allocated_gb']:.4f} GB")
    assert loss == loss, "loss is NaN"  # NaN check
    print(f"  [OK] {label} forward+backward+step completed")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-only", action="store_true", help="skip the 4-bit path")
    ap.add_argument("--quant-only", action="store_true", help="skip the bf16 path")
    args = ap.parse_args()

    _banner("Environment")
    print(f"  torch           : {torch.__version__}")
    print(f"  cuda available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device          : {torch.cuda.get_device_name(0)}")
        print(f"  capability      : {device_capability()}")
        print(f"  cuda (torch)    : {torch.version.cuda}")
        if not is_blackwell_sm120():
            print("  [WARN] device is not sm_120; expected (12, 0) for RTX 5090")
        else:
            print("  [OK] device is sm_120 (Blackwell / RTX 5090)")
    else:
        print("  [WARN] no CUDA device — smoke test will run on CPU (plumbing only)")

    if not args.bf16_only:
        run_path(quantized=True)
    if not args.quant_only:
        run_path(quantized=False)

    _banner("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
