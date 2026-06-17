"""Held-out generation + metric scoring for a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from ..data.sni import get_dataset
from .metrics import score_predictions


def _clean_generation(text: str) -> str:
    """Take the first non-empty line of a generation.

    SNI gold outputs are a single label (classification) or a single line
    (generation), but models often emit the answer then keep going
    ("positive\\n\\nNegative Example ..."). Scoring the first non-empty line
    matches the task convention without rewarding the run-on.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def _load_model(checkpoint: Path, base_model: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched generation

    if (checkpoint / "adapter_config.json").exists():
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base, str(checkpoint))
    else:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), dtype=torch.bfloat16)
    return model.to(device).eval(), tok


def evaluate_checkpoint(
    checkpoint: str | Path,
    base_model: str,
    task: str,
    metric: str,
    *,
    max_eval_samples: int = 200,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    max_seq_len: int = 512,
) -> dict:
    """Generate on the held-out test split and score. Returns a result dict."""
    checkpoint = Path(checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = _load_model(checkpoint, base_model, device)

    bundle = get_dataset(
        task, tok, max_seq_len=max_seq_len, max_eval_samples=max_eval_samples
    )
    examples = bundle.test_eval
    preds: list[str] = []
    refs: list[list[str]] = [ex["references"] for ex in examples]

    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        maxlen = max(len(e["input_ids"]) for e in chunk)
        input_ids, attn = [], []
        for e in chunk:  # left-pad
            pad = maxlen - len(e["input_ids"])
            input_ids.append([tok.pad_token_id] * pad + e["input_ids"])
            attn.append([0] * pad + e["attention_mask"])
        input_ids = torch.tensor(input_ids, device=device)
        attn = torch.tensor(attn, device=device)
        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j in range(len(chunk)):
            gen = out[j][input_ids.shape[1]:]
            preds.append(_clean_generation(tok.decode(gen, skip_special_tokens=True)))

    scored = score_predictions(preds, refs, metric)
    scored.update(
        task=task,
        base_model=base_model,
        checkpoint=str(checkpoint),
        sample_predictions=preds[:5],
        sample_references=refs[:5],
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scored
