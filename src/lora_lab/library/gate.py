"""Sprint 4 — the quality gate: every library LoRA must beat the base.

For each task we eval the adapter-on-base vs. the bare base on the *same*
held-out test split and record ``margin = adapter - base``. Adapters that don't
clear a committed threshold τ are quarantined (a valid negative result).

Efficiency: the 7B base is loaded **once** and stays resident. Adapters attach
via ``PeftModel.load_adapter`` / ``set_adapter``; the base score comes from the
*same* model under ``disable_adapter()`` — so we never reload the 7B per task,
and base/adapter are guaranteed identical except for the LoRA. Adapters are
deleted after each task to keep memory flat across the full sweep.

Metric is inferred per task (the Lots-of-LoRAs repos don't label kind): a small
set of short distinct gold labels ⇒ classification/exact_match, else
generation/rougeL.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from ..data.sni import TaskSpec, get_dataset
from ..eval.evaluate import _generate_and_score
from .manifest import BASE_MODEL, EXPECTED_RANK, EXPECTED_TARGETS, LibraryEntry, _sha

# "Clearly beats base" threshold (recorded in the findings). EM/ROUGE-L are on
# [0,1]; τ=0.05 means a ≥5-point absolute lift on the task's own metric.
DEFAULT_TAU = 0.05


def infer_kind_metric(test_eval: list[dict]) -> tuple[str, str]:
    """Heuristic: short, low-cardinality gold answers ⇒ classification."""
    firsts = [str(ex["references"][0]).strip() for ex in test_eval if ex["references"]]
    if not firsts:
        return "generation", "rougeL"
    distinct = set(firsts)
    short = all(len(f.split()) <= 4 for f in firsts)
    if short and len(distinct) <= max(15, len(firsts) // 8):
        return "classification", "exact_match"
    return "generation", "rougeL"


def assert_compatible(adapter_config: dict, task_num: str) -> int:
    """S2 invariant: the adapter must apply to our exact base.

    Hard invariants are **base model** and **target modules** — those decide
    whether the adapter even loads correctly. Rank is *not* a compatibility
    constraint (any LoRA rank applies to the same base); this release ships a
    mix of r=16 and r=43 adapters (the rank-adaptive Compress-then-Serve
    variants), so we *record* the rank rather than assert a single value.
    Returns the adapter's actual rank.
    """
    base = adapter_config.get("base_model_name_or_path", "")
    assert base == BASE_MODEL, f"{task_num}: adapter base {base!r} != {BASE_MODEL!r}"
    targets = set(adapter_config.get("target_modules", []))
    assert targets == EXPECTED_TARGETS, \
        f"{task_num}: targets {targets} != {EXPECTED_TARGETS}"
    return int(adapter_config.get("r", EXPECTED_RANK))


class GateRunner:
    """Holds the resident base + PEFT wrapper across the whole sweep."""

    def __init__(self, base_model: str = BASE_MODEL, max_eval_samples: int = 150):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_eval_samples = max_eval_samples
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
        self.base = base.to(self.device).eval()
        self.peft = None  # lazily created on first adapter

    def _bundle(self, entry: LibraryEntry):
        spec = TaskSpec(name=entry.task_name, hf_repo=entry.dataset_repo,
                        kind="generation", metric="rougeL", description=entry.description)
        return get_dataset(spec, self.tok, max_seq_len=512,
                           max_eval_samples=self.max_eval_samples)

    def run_task(self, entry: LibraryEntry, tau: float = DEFAULT_TAU) -> dict:
        """Eval base vs adapter for one task; return a result row."""
        from huggingface_hub import hf_hub_download
        from peft import PeftModel

        t0 = time.time()
        bundle = self._bundle(entry)
        kind, metric = infer_kind_metric(bundle.test_eval)

        # --- attach adapter onto the resident base -----------------------
        cfg = json.load(open(hf_hub_download(entry.adapter_repo, "adapter_config.json")))
        rank = assert_compatible(cfg, entry.task_num)
        name = entry.task_num
        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.base, entry.adapter_repo,
                                                  adapter_name=name).eval()
        else:
            self.peft.load_adapter(entry.adapter_repo, adapter_name=name)
        self.peft.set_adapter(name)

        # adapter score (LoRA active) then base score (same model, LoRA off)
        adapter = _generate_and_score(self.peft, self.tok, bundle.test_eval, metric,
                                      max_new_tokens=16 if kind == "classification" else 64,
                                      batch_size=16)
        with self.peft.disable_adapter():
            base = _generate_and_score(self.peft, self.tok, bundle.test_eval, metric,
                                       max_new_tokens=16 if kind == "classification" else 64,
                                       batch_size=16)
        self.peft.delete_adapter(name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        margin = round(adapter["score"] - base["score"], 4)
        passed = margin >= tau
        return {
            "task_num": entry.task_num,
            "task_name": entry.task_name,
            "rank": rank,
            "kind": kind,
            "metric": metric,
            "n_eval": adapter["n"],
            "split_hash": bundle.hashes.get("test", ""),
            "adapter_score": round(adapter["score"], 4),
            "base_score": round(base["score"], 4),
            "margin": margin,
            "tau": tau,
            "gate": "pass" if passed else "fail",
            "wallclock_s": round(time.time() - t0, 1),
            "adapter_hash": _sha(entry.adapter_repo),
            "sample_pred": adapter.get("sample_predictions", [])[:2],
            "sample_ref": adapter.get("sample_references", [])[:2],
        }
