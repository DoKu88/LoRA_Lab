#!/usr/bin/env python
"""Networked data sanity check (Sprint 2 required testing).

For each manifest task and a couple of base tokenizers:
  * verify split-hash determinism (load twice -> identical hashes) and that
    the full-split hashes match what's pinned in configs/tasks.yaml;
  * report the token-length distribution of supervised examples;
  * eyeball one decoded batch (prompt vs. supervised completion).

    conda run -n lora_lab python scripts/check_data.py
    conda run -n lora_lab python scripts/check_data.py --task task1564_triviaqa_answer_generation
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from lora_lab.data.sni import IGNORE_INDEX, get_dataset, load_tasks_manifest  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "tasks.yaml"
# A base model (no chat template) and an instruct model (chat template).
TOKENIZERS = ["HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B-Instruct"]


def _pinned_full_hashes() -> dict:
    with MANIFEST.open() as f:
        doc = yaml.safe_load(f)
    return {k: v.get("split_hashes", {}) for k, v in doc["tasks"].items()}


def check_task(task: str, tok_id: str, pinned: dict) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tok_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"\n{'#' * 70}\n# {task}  |  tokenizer={tok_id}\n{'#' * 70}")

    # full splits (no subsampling) -> compare to pinned manifest hashes
    full = get_dataset(
        task, tok, max_train_samples=-1, max_val_samples=-1,
        max_eval_samples=-1, manifest=MANIFEST,
    )
    full2 = get_dataset(
        task, tok, max_train_samples=-1, max_val_samples=-1,
        max_eval_samples=-1, manifest=MANIFEST,
    )
    assert full.hashes == full2.hashes, "non-deterministic split hashes!"
    pin = pinned.get(task, {})
    # manifest uses 'valid' key; bundle uses 'val'
    ok_train = full.hashes["train"] == pin.get("train")
    ok_test = full.hashes["test"] == pin.get("test")
    print(f"  split-hash determinism : OK (train={full.hashes['train']})")
    print(f"  matches pinned manifest: train={ok_train} test={ok_test}")

    lens = [len(ex["input_ids"]) for ex in full.train]
    sup = [sum(1 for l in ex["labels"] if l != IGNORE_INDEX) for ex in full.train]
    if lens:
        print(
            f"  token len  n={len(lens)}  min={min(lens)} "
            f"median={statistics.median(lens):.0f} max={max(lens)} "
            f"p95={sorted(lens)[int(0.95 * len(lens)) - 1]}"
        )
        print(f"  supervised(completion) tokens: median={statistics.median(sup):.0f}")

    # eyeball one example
    if full.train:
        ex = full.train[0]
        prompt_ids = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l == IGNORE_INDEX]
        comp_ids = [l for l in ex["labels"] if l != IGNORE_INDEX]
        print("  --- decoded prompt (masked) ---")
        print("   ", tok.decode(prompt_ids)[:300].replace("\n", "\\n"))
        print("  --- decoded completion (supervised) ---")
        print("   ", repr(tok.decode(comp_ids)))
    if full.test_eval:
        ev = full.test_eval[0]
        print("  --- eval references ---")
        print("   ", ev["references"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None, help="single task (default: all)")
    ap.add_argument("--tokenizer", default=None, help="single tokenizer id")
    args = ap.parse_args()

    import datasets as hfds

    hfds.disable_progress_bars()

    tasks = [args.task] if args.task else list(load_tasks_manifest(MANIFEST))
    toks = [args.tokenizer] if args.tokenizer else TOKENIZERS
    pinned = _pinned_full_hashes()

    for task in tasks:
        for tok_id in toks:
            check_task(task, tok_id, pinned)
    print("\n[OK] data sanity check complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
