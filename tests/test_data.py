"""Data-pipeline unit tests.

The masking/truncation/collation logic is exercised with a tiny fake
tokenizer so these run on CPU with no network. A separate networked sanity
check (scripts/check_data.py) covers real tokenizers + split-hash determinism.
"""

import pytest

from lora_lab.data.sni import (
    IGNORE_INDEX,
    DataCollatorForSupervised,
    build_prompt,
    build_supervised,
    split_hash,
)


class FakeTokenizer:
    """Whitespace tokenizer with integer ids; optional chat template."""

    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.eos_token_id = 99
        self.pad_token_id = 0

    def _ids(self, text):
        return [hash(w) % 90 + 1 for w in text.split()]

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self._ids(text)}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = " ".join(m["content"] for m in messages)
        ids = self._ids(text)
        if add_generation_prompt:
            ids = ids + [self._ids("ASSISTANT")[0]]
        return ids


def test_split_hash_deterministic_and_order_sensitive():
    assert split_hash(["a", "b", "c"]) == split_hash(["a", "b", "c"])
    assert split_hash(["a", "b"]) != split_hash(["b", "a"])
    assert len(split_hash(["x"])) == 12


def test_supervised_masks_prompt():
    tok = FakeTokenizer()
    ex = build_supervised(tok, "translate this sentence", "le resultat", max_seq_len=64)
    assert ex is not None
    # exactly the completion tokens are supervised; prompt is IGNORE_INDEX
    n_supervised = sum(1 for l in ex["labels"] if l != IGNORE_INDEX)
    assert n_supervised >= 1
    # last label is the eos token
    assert ex["input_ids"][-1] == tok.eos_token_id
    assert ex["labels"][-1] == tok.eos_token_id
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_supervised_chat_template_path():
    tok = FakeTokenizer(chat_template="dummy")
    ex = build_supervised(tok, "hello there", "general kenobi", max_seq_len=64)
    assert ex is not None
    assert any(l != IGNORE_INDEX for l in ex["labels"])


def test_empty_output_dropped():
    tok = FakeTokenizer()
    assert build_supervised(tok, "some input", "   ", max_seq_len=64) is None


def test_oversized_truncates_to_max_len():
    tok = FakeTokenizer()
    long_input = "word " * 500
    ex = build_supervised(tok, long_input, "answer", max_seq_len=32)
    assert ex is not None
    assert len(ex["input_ids"]) <= 32
    # completion still supervised after truncation
    assert any(l != IGNORE_INDEX for l in ex["labels"])


def test_build_prompt_keeps_references():
    tok = FakeTokenizer()
    p = build_prompt(tok, "what is 2+2", ["four", "4"], max_seq_len=64)
    assert p["references"] == ["four", "4"]
    assert len(p["input_ids"]) == len(p["attention_mask"])


def test_collator_pads_batch():
    tok = FakeTokenizer()
    feats = [
        build_supervised(tok, "a b c", "x", max_seq_len=64),
        build_supervised(tok, "a b c d e f g", "y z", max_seq_len=64),
    ]
    collate = DataCollatorForSupervised(tok)
    batch = collate(feats)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    # padded positions are ignored in labels and masked in attention
    assert (batch["labels"] == IGNORE_INDEX).any()
    assert batch["input_ids"].shape[0] == 2
