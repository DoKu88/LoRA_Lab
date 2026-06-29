"""Meta-training samplers — feed the loop from the library data.

LibraryReconSampler: for each TRAIN-split task, load its Lots-of-LoRAs adapter's
A/B tensors and yield (description, target_adapter) for the reconstruction
objective. SNISFTSampler: tokenize a train task's SNI batch with prompt-masking
and yield (description, batch) for the SFT objective.

Both read only the **train** split — never the held-out tasks (the leakage
contract). Adapter loading is cached. The PEFT->base key mapping is pure logic
(``parse_lora_state_dict``).
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ..data.sni import DataCollatorForSupervised, TaskSpec, get_dataset

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

    Keys come out matching the base model's ``named_modules`` paths (e.g.
    ``model.layers.0.self_attn.q_proj``), so they align 1:1 with the keys the
    HyperLoRAGenerator produces from ``target_specs``.
    """
    factor_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith("lora_A.weight"):
            factor_pairs.setdefault(_base_key(key, "lora_A.weight"), {})["A"] = tensor
        elif key.endswith("lora_B.weight"):
            factor_pairs.setdefault(_base_key(key, "lora_B.weight"), {})["B"] = tensor
    return {base: (factors["A"], factors["B"])
            for base, factors in factor_pairs.items() if "A" in factors and "B" in factors}


def _load_split_train(split_path: str | Path) -> list[str]:
    raw = yaml.safe_load(Path(split_path).read_text()) or {}
    return list((raw.get("train") or {}).keys())


def _load_library(library_path: str | Path) -> dict[str, dict]:
    raw = yaml.safe_load(Path(library_path).read_text()) or {}
    return raw.get("tasks") or {}


class LibraryReconSampler:
    """Yield (description, target_adapter) over the TRAIN split for reconstruction."""

    def __init__(self, split_path: str | Path, library_path: str | Path,
                 *, seed: int = 0):
        train = _load_split_train(split_path)
        library = _load_library(library_path)
        # keep only train tasks that have an adapter + description recorded
        self.tasks = [task for task in train if task in library and library[task].get("adapter_repo")
                      and library[task].get("description")]
        self.library = library
        self.rng = random.Random(seed)
        self._cache: dict[str, dict] = {}

    def _load_adapter(self, task: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if task not in self._cache:
            path = hf_hub_download(self.library[task]["adapter_repo"], "adapter_model.safetensors")
            self._cache[task] = parse_lora_state_dict(load_file(path))
        return self._cache[task]

    def sample(self) -> tuple[str, dict]:
        task = self.rng.choice(self.tasks)
        return self.library[task]["description"], self._load_adapter(task)

    def __len__(self) -> int:
        return len(self.tasks)


class SNISFTSampler:
    """Yield (description, prompt-masked batch) over the TRAIN split for SFT.

    Reuses the ``data/sni`` pipeline (chat-template + prompt masking) so SFT
    trains on exactly the task data the library adapters were trained for.
    """

    def __init__(self, split_path: str | Path, library_path: str | Path, tokenizer,
                 *, batch_size: int = 4, max_seq_len: int = 512, seed: int = 0):
        train = _load_split_train(split_path)
        library = _load_library(library_path)
        self.tokenizer = tokenizer
        self.collate = DataCollatorForSupervised(tokenizer)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.tasks, self._bundles, self.library = [], {}, library
        for task in train:
            spec = library.get(task)
            if not (spec and spec.get("dataset_repo") and spec.get("description")):
                continue
            task_spec = TaskSpec(name=task, hf_repo=spec["dataset_repo"],
                                 kind=spec.get("kind", "generation"),
                                 metric=spec.get("metric", "rougeL"), description=spec["description"])
            self._bundles[task] = get_dataset(task_spec, tokenizer, max_seq_len=max_seq_len)
            self.tasks.append(task)

    def sample(self) -> tuple[str, dict]:
        task = self.rng.choice(self.tasks)
        train = self._bundles[task].train
        picks = [self.rng.choice(train) for _ in range(min(self.batch_size, len(train)))]
        return self.library[task]["description"], self.collate(picks)

    def __len__(self) -> int:
        return len(self.tasks)
