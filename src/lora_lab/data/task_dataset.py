"""Deterministic, versioned task-dataset loaders.

Pulls per-task instruction datasets from the public ``Lots-of-LoRAs`` repos,
formats each example with the base model's chat template when it has one
(instruct models) and a plain fallback otherwise (base models like
SmolLM2-135M), tokenizes with prompt-masking for supervised fine-tuning, and
exposes a held-out eval view with raw references for metric scoring.

Determinism: splits come pre-defined from the source repos; any subsampling
is seeded; ``split_hash`` over the (ordered) example ids lets us assert that
the exact split used is reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets as hfds
import torch
import yaml

# -100 is torch's CrossEntropy ignore_index — masks prompt tokens from loss.
IGNORE_INDEX = -100

_DEFAULT_MANIFEST = "configs/tasks.yaml"


@dataclass
class TaskSpec:
    name: str
    hf_repo: str
    kind: str  # classification | generation
    metric: str  # exact_match | rougeL
    description: str = ""


def load_tasks_manifest(path: str | Path = _DEFAULT_MANIFEST) -> dict[str, TaskSpec]:
    """Parse ``configs/tasks.yaml`` into ``{task_name: TaskSpec}``."""
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    out: dict[str, TaskSpec] = {}
    for name, spec in (raw.get("tasks") or {}).items():
        out[name] = TaskSpec(
            name=name,
            hf_repo=spec["hf_repo"],
            kind=spec.get("kind", "generation"),
            metric=spec.get("metric", "rougeL"),
            description=(spec.get("description") or "").strip(),
        )
    return out


def split_hash(ids: list[str]) -> str:
    """Stable hash of an ordered list of example ids (12 hex chars)."""
    h = hashlib.sha256()
    for i in ids:
        h.update(str(i).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Prompt / supervised-example construction
# ---------------------------------------------------------------------------
def _first(output: Any) -> str:
    """``output`` is a list of acceptable answers; training uses the 1st."""
    if isinstance(output, (list, tuple)):
        return str(output[0]) if output else ""
    return str(output)


def _references(output: Any) -> list[str]:
    if isinstance(output, (list, tuple)):
        return [str(o) for o in output]
    return [str(output)]


def _has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def _flatten_ids(x: Any) -> list[int]:
    """Normalize tokenizer output to a flat list[int].

    ``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding`` (a dict
    subclass) in transformers 5.x and a plain list in older versions; either
    may also be nested ([[ids]]). Handle all shapes.
    """
    # BatchEncoding subclasses UserDict (NOT dict), so test by access, not type.
    if not isinstance(x, (list, tuple)):
        x = x["input_ids"]
    if x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x)


def _prompt_ids(tokenizer, input_text: str) -> list[int]:
    """Tokenize the prompt with a generation prompt appended.

    Instruct models: use the chat template. Base models: plain text with a
    trailing newline so the completion starts on its own line.
    """
    if _has_chat_template(tokenizer):
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return _flatten_ids(out)
    enc = tokenizer(input_text + "\n", add_special_tokens=True)
    return _flatten_ids(enc)


def build_supervised(
    tokenizer,
    input_text: str,
    output_text: str,
    max_seq_len: int,
) -> dict[str, list[int]] | None:
    """Build one prompt-masked supervised example.

    Returns ``{input_ids, attention_mask, labels}`` with prompt tokens set to
    ``IGNORE_INDEX`` in ``labels``. Returns ``None`` for degenerate examples
    (empty completion, or a prompt that already fills ``max_seq_len`` leaving
    no room for the answer) so callers can drop them.
    """
    output_text = (output_text or "").strip()
    if not output_text:
        return None

    prompt_ids = _prompt_ids(tokenizer, input_text)
    eos = tokenizer.eos_token_id
    completion_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
    if eos is not None:
        completion_ids = completion_ids + [eos]

    # Reserve at least 1 completion token; truncate the *prompt* from the left
    # (keep the instance, which sits at the end of the input) if needed.
    max_prompt = max_seq_len - len(completion_ids)
    if max_prompt <= 0:
        # Completion alone overflows; keep a minimal tail of the completion.
        completion_ids = completion_ids[: max_seq_len - 1] + (
            [eos] if eos is not None else []
        )
        prompt_ids = prompt_ids[:1]
    elif len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids[:]
    attention_mask = [1] * len(input_ids)
    if not any(t != IGNORE_INDEX for t in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_prompt(
    tokenizer,
    input_text: str,
    output: Any,
    max_seq_len: int,
) -> dict[str, Any]:
    """Build a generation prompt + references for held-out eval."""
    prompt_ids = _prompt_ids(tokenizer, input_text)
    if len(prompt_ids) > max_seq_len:
        prompt_ids = prompt_ids[-max_seq_len:]
    return {
        "input_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
        "references": _references(output),
        "input_text": input_text,
    }


# ---------------------------------------------------------------------------
# Bundle + collator
# ---------------------------------------------------------------------------
@dataclass
class DatasetBundle:
    task: TaskSpec
    train: list[dict]
    val: list[dict]
    test_eval: list[dict]
    hashes: dict[str, str] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task.name,
            "metric": self.task.metric,
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test_eval),
            "hashes": self.hashes,
        }


class DataCollatorForSupervised:
    """Pad ``input_ids``/``attention_mask``/``labels`` to the batch max."""

    def __init__(self, tokenizer):
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: list[dict]):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [IGNORE_INDEX] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------
def _subsample(ds, n: int, seed: int):
    """Deterministically take the first ``n`` rows of a seeded shuffle."""
    if n is None or n < 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def get_dataset(
    task: str | TaskSpec,
    tokenizer,
    *,
    max_seq_len: int = 512,
    max_train_samples: int = -1,
    max_val_samples: int = 64,
    max_eval_samples: int = 200,
    seed: int = 42,
    manifest: str | Path = _DEFAULT_MANIFEST,
) -> DatasetBundle:
    """Return tokenized train/val and an eval view for one task.

    Splits are taken from the source repo (train/valid/test); subsampling is
    seeded and the resulting ordered ids are hashed into ``bundle.hashes`` for
    the reproducibility assertion.
    """
    hfds.disable_progress_bars()

    spec = task if isinstance(task, TaskSpec) else load_tasks_manifest(manifest)[task]
    raw = hfds.load_dataset(spec.hf_repo)

    # Source repos use 'valid' for the dev split; fall back gracefully.
    val_key = "valid" if "valid" in raw else ("validation" if "validation" in raw else None)
    test_key = "test" if "test" in raw else val_key

    train_raw = _subsample(raw["train"], max_train_samples, seed)
    val_raw = _subsample(raw[val_key], max_val_samples, seed) if val_key else []
    test_raw = _subsample(raw[test_key], max_eval_samples, seed) if test_key else []

    hashes = {
        "train": split_hash(list(train_raw["id"])) if len(train_raw) else "",
        "val": split_hash(list(val_raw["id"])) if val_key and len(val_raw) else "",
        "test": split_hash(list(test_raw["id"])) if test_key and len(test_raw) else "",
    }

    train: list[dict] = []
    for ex in train_raw:
        built = build_supervised(tokenizer, ex["input"], _first(ex["output"]), max_seq_len)
        if built is not None:
            train.append(built)

    val: list[dict] = []
    for ex in val_raw:
        built = build_supervised(tokenizer, ex["input"], _first(ex["output"]), max_seq_len)
        if built is not None:
            val.append(built)

    test_eval = [
        build_prompt(tokenizer, ex["input"], ex["output"], max_seq_len)
        for ex in test_raw
    ]

    return DatasetBundle(task=spec, train=train, val=val, test_eval=test_eval, hashes=hashes)
