#!/usr/bin/env python
"""Checkpoint reload + inference smoke test (Sprint 4 required testing).

Reloads a saved checkpoint — LoRA/QLoRA adapter (re-attached to its base) or a
full-FT model — and generates on one held-out eval prompt to prove the
checkpoint is loadable and produces text.

    conda run -n lora_lab python scripts/reload_check.py \
        --checkpoint results/runs/lora-SmolLM2-135M-task843_.../checkpoint \
        --base-model HuggingFaceTB/SmolLM2-135M \
        --task task843_financial_phrasebank_classification
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from lora_lab.data.sni import get_dataset  # noqa: E402


def is_adapter(ckpt: Path) -> bool:
    return (ckpt / "adapter_config.json").exists()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ckpt))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if is_adapter(ckpt):
        from peft import PeftModel

        print(f"[reload] adapter checkpoint -> base {args.base_model}")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(ckpt))
    else:
        print("[reload] full-weights checkpoint")
        model = AutoModelForCausalLM.from_pretrained(str(ckpt), dtype=torch.bfloat16)
    model = model.to(device).eval()

    bundle = get_dataset(args.task, tok, max_eval_samples=1)
    ex = bundle.test_eval[0]
    input_ids = torch.tensor([ex["input_ids"]], device=device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(json.dumps({
        "checkpoint": str(ckpt),
        "is_adapter": is_adapter(ckpt),
        "generated": gen.strip(),
        "references": ex["references"],
    }, indent=2))
    assert isinstance(gen, str)
    print("[OK] checkpoint reloaded and generated text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
