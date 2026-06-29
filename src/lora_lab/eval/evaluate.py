"""Held-out generation + metric scoring for a saved checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.task_dataset import get_dataset
from .metrics import score_predictions


def _clean_generation(text: str) -> str:
    """Take the first non-empty line of a generation.

    Gold outputs are a single label (classification) or a single line
    (generation), but models often emit the answer then keep going
    ("positive\\n\\nNegative Example ..."). Scoring the first non-empty line
    matches the task convention without rewarding the run-on.
    """
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text.strip()


def _generate_and_score(model, tok, examples, metric, *, max_new_tokens, batch_size):
    """Greedy-generate on prompt-only examples and score against references.

    Shared by checkpoint eval and inline (post-train, in-memory) eval. Left-pads
    for batched generation; restores the tokenizer's padding side afterward.
    """
    device = next(model.parameters()).device
    prev_side = tok.padding_side
    tok.padding_side = "left"
    preds: list[str] = []
    refs: list[list[str]] = [ex["references"] for ex in examples]
    try:
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
                    input_ids=input_ids, attention_mask=attn,
                    max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
            for j in range(len(chunk)):
                gen = out[j][input_ids.shape[1]:]
                preds.append(_clean_generation(tok.decode(gen, skip_special_tokens=True)))
    finally:
        tok.padding_side = prev_side
    scored = score_predictions(preds, refs, metric)
    scored["sample_predictions"] = preds[:5]
    scored["sample_references"] = refs[:5]
    return scored


def evaluate_inline(model, tok, bundle, metric, *, max_new_tokens=64, batch_size=8):
    """Eval an in-memory model (just-trained) on its held-out split.

    Flips the model to inference config (cache on, grad-checkpointing off, eval
    mode), scores, then restores train config so the caller can continue/save.
    Avoids persisting a 14 GB checkpoint just to eval it.
    """
    was_training = model.training
    prev_cache = getattr(model.config, "use_cache", None)
    gc_enabled = getattr(model, "is_gradient_checkpointing", False)
    model.eval()
    model.config.use_cache = True
    if gc_enabled:
        model.gradient_checkpointing_disable()
    try:
        scored = _generate_and_score(model, tok, bundle.test_eval, metric,
                                     max_new_tokens=max_new_tokens, batch_size=batch_size)
    finally:
        if gc_enabled:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if prev_cache is not None:
            model.config.use_cache = prev_cache
        if was_training:
            model.train()
    return scored


def _load_model(checkpoint: Path, base_model: str, device: str):
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left-pad for batched generation

    if (checkpoint / "adapter_config.json").exists():
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
