"""Samplers — feed the training loop a batch of (description, ground-truth).

LibraryReconSampler   reconstruction: (descriptions, [target LoRA adapters])
SFTSampler            SFT: (description, tokenized+prompt-masked example batch)
SyntheticReconSampler random targets for a CPU smoke (no downloads)

Each exposes ``batch(n)`` returning ``(descriptions, targets)`` for one step.
Samplers read a single split of the locked split file (``train`` or ``val``);
adapter loading is cached. ``parse_lora_state_dict`` maps PEFT keys to base keys.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..data.task_dataset import DataCollatorForSupervised, TaskSpec, get_dataset

# PEFT saves LoRA factors as
#   base_model.model.<module path>.lora_A.weight   shape (r, in)
#   base_model.model.<module path>.lora_B.weight   shape (out, r)
# which already match our (A:(r,in), B:(out,r)) convention.
_PEFT_PREFIX = "base_model.model."


def _base_key(peft_key: str, suffix: str) -> str:
    stripped = peft_key[len(_PEFT_PREFIX):] if peft_key.startswith(_PEFT_PREFIX) else peft_key
    return stripped[: -len("." + suffix)]


def parse_lora_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{peft_key: tensor} -> {base_module_key: (A:(rank,in), B:(out,rank))}.

    Keys come out matching the base model's ``named_modules`` paths, so they
    align 1:1 with the keys the generator produces from ``target_specs``.
    """
    factor_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith("lora_A.weight"):
            factor_pairs.setdefault(_base_key(key, "lora_A.weight"), {})["A"] = tensor
        elif key.endswith("lora_B.weight"):
            factor_pairs.setdefault(_base_key(key, "lora_B.weight"), {})["B"] = tensor
    return {base: (factors["A"], factors["B"])
            for base, factors in factor_pairs.items() if "A" in factors and "B" in factors}


def _split_tasks(split_path: str | Path, which: str) -> list[str]:
    """Task names in one split (``train`` / ``val`` / ``held_out``)."""
    raw = yaml.safe_load(Path(split_path).read_text()) or {}
    return list((raw.get(which) or {}).keys())


def _load_library(library_path: str | Path) -> dict[str, dict]:
    raw = yaml.safe_load(Path(library_path).read_text()) or {}
    return raw.get("tasks") or {}


class LibraryReconSampler:
    """Reconstruction targets: each task's library LoRA A/B tensors."""

    def __init__(self, split_path: str | Path, library_path: str | Path,
                 *, split: str = "train", seed: int = 0):
        tasks = _split_tasks(split_path, split)
        library = _load_library(library_path)
        # keep only tasks with an adapter + description recorded
        self.tasks = [task for task in tasks if task in library
                      and library[task].get("adapter_repo") and library[task].get("description")]
        self.library = library
        self.rng = random.Random(seed)
        self._cache: dict[str, dict] = {}

    def _load_adapter(self, task: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if task not in self._cache:
            path = hf_hub_download(self.library[task]["adapter_repo"], "adapter_model.safetensors")
            self._cache[task] = parse_lora_state_dict(load_file(path))
        return self._cache[task]

    def batch(self, n: int) -> tuple[list[str], list[dict]]:
        tasks = [self.rng.choice(self.tasks) for _ in range(n)]
        descriptions = [self.library[task]["description"] for task in tasks]
        targets = [self._load_adapter(task) for task in tasks]
        return descriptions, targets

    def __len__(self) -> int:
        return len(self.tasks)


class SFTSampler:
    """SFT batches: a task's examples, chat-templated + prompt-masked."""

    def __init__(self, split_path: str | Path, library_path: str | Path, tokenizer,
                 *, split: str = "train", max_seq_len: int = 512, seed: int = 0):
        tasks = _split_tasks(split_path, split)
        library = _load_library(library_path)
        self.tokenizer = tokenizer
        self.collate = DataCollatorForSupervised(tokenizer)
        self.rng = random.Random(seed)
        self.tasks, self._bundles, self.library = [], {}, library
        for task in tasks:
            spec = library.get(task)
            if not (spec and spec.get("dataset_repo") and spec.get("description")):
                continue
            task_spec = TaskSpec(name=task, hf_repo=spec["dataset_repo"],
                                 kind=spec.get("kind", "generation"),
                                 metric=spec.get("metric", "rougeL"), description=spec["description"])
            self._bundles[task] = get_dataset(task_spec, tokenizer, max_seq_len=max_seq_len)
            self.tasks.append(task)

    def batch(self, n: int) -> tuple[list[str], dict]:
        task = self.rng.choice(self.tasks)
        examples = self._bundles[task].train
        picks = [self.rng.choice(examples) for _ in range(min(n, len(examples)))]
        return [self.library[task]["description"]], self.collate(picks)

    def __len__(self) -> int:
        return len(self.tasks)


class SyntheticReconSampler:
    """Random target adapters + dummy descriptions — a CPU smoke (no downloads)."""

    def __init__(self, target_specs: dict[str, tuple[int, int]], rank: int, seed: int = 0):
        self.specs = target_specs
        self.rank = rank
        self.torch_generator = torch.Generator().manual_seed(seed)
        self._descriptions = ["classify sentiment", "translate text", "answer the question",
                              "summarize the passage", "detect entailment"]
        self._index = 0

    def batch(self, n: int) -> tuple[list[str], list[dict]]:
        descriptions, targets = [], []
        for _ in range(n):
            target = {key: (torch.randn(self.rank, in_features, generator=self.torch_generator),
                            torch.randn(out_features, self.rank, generator=self.torch_generator))
                      for key, (in_features, out_features) in self.specs.items()}
            descriptions.append(self._descriptions[self._index % len(self._descriptions)])
            self._index += 1
            targets.append(target)
        return descriptions, targets
