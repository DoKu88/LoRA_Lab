"""Real meta-training samplers (Sprint 3/4) — feed the loop from Phase-1 data.

LibraryReconSampler (S3): for each TRAIN-split task, load its Lots-of-LoRAs
adapter's A/B tensors and yield (description, target_adapter) for the
reconstruction objective. SNISFTSampler (S4): tokenize a train task's SNI batch
with prompt-masking and yield (description, batch) for the SFT objective.

Both read only the **train** split of the Phase-1 locked split — never the
held-out tasks (the leakage contract). Adapter loading is lazy + cached. The
PEFT->base key mapping is pure logic (``parse_lora_state_dict``), unit-tested
without any download.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import yaml

# PEFT saves LoRA factors as
#   base_model.model.<module path>.lora_A.weight   shape (r, in)
#   base_model.model.<module path>.lora_B.weight   shape (out, r)
# which already match our (A:(r,in), B:(out,r)) convention.
_PEFT_PREFIX = "base_model.model."


def _base_key(peft_key: str, suffix: str) -> str:
    k = peft_key[len(_PEFT_PREFIX):] if peft_key.startswith(_PEFT_PREFIX) else peft_key
    return k[: -len("." + suffix)]


def parse_lora_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """{peft_key: tensor} -> {base_module_key: (A:(r,in), B:(out,r))}.

    Keys come out matching the base model's ``named_modules`` paths (e.g.
    ``model.layers.0.self_attn.q_proj``), so they align 1:1 with the keys the
    HyperLoRAGenerator produces from ``target_specs``.
    """
    halves: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd.items():
        if k.endswith("lora_A.weight"):
            halves.setdefault(_base_key(k, "lora_A.weight"), {})["A"] = v
        elif k.endswith("lora_B.weight"):
            halves.setdefault(_base_key(k, "lora_B.weight"), {})["B"] = v
    return {base: (ab["A"], ab["B"]) for base, ab in halves.items() if "A" in ab and "B" in ab}


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
        lib = _load_library(library_path)
        # keep only train tasks that have an adapter + description recorded
        self.tasks = [t for t in train if t in lib and lib[t].get("adapter_repo")
                      and lib[t].get("description")]
        self.lib = lib
        self.rng = random.Random(seed)
        self._cache: dict[str, dict] = {}

    def _load_adapter(self, task: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        if task not in self._cache:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            path = hf_hub_download(self.lib[task]["adapter_repo"], "adapter_model.safetensors")
            self._cache[task] = parse_lora_state_dict(load_file(path))
        return self._cache[task]

    def sample(self) -> tuple[str, dict]:
        task = self.rng.choice(self.tasks)
        return self.lib[task]["description"], self._load_adapter(task)

    def __len__(self) -> int:
        return len(self.tasks)


class SNISFTSampler:
    """Yield (description, prompt-masked batch) over the TRAIN split for SFT.

    Reuses the Phase-0 ``data/sni`` pipeline (chat-template + prompt masking) so
    SFT trains on exactly the task data the library adapters were trained for.
    """

    def __init__(self, split_path: str | Path, library_path: str | Path, tokenizer,
                 *, batch_size: int = 4, max_seq_len: int = 512, seed: int = 0):
        from ..data.sni import DataCollatorForSupervised, TaskSpec, get_dataset

        train = _load_split_train(split_path)
        lib = _load_library(library_path)
        self.tok = tokenizer
        self.collate = DataCollatorForSupervised(tokenizer)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.tasks, self._bundles, self.lib = [], {}, lib
        for t in train:
            spec = lib.get(t)
            if not (spec and spec.get("dataset_repo") and spec.get("description")):
                continue
            ts = TaskSpec(name=t, hf_repo=spec["dataset_repo"],
                          kind=spec.get("kind", "generation"),
                          metric=spec.get("metric", "rougeL"), description=spec["description"])
            self._bundles[t] = get_dataset(ts, tokenizer, max_seq_len=max_seq_len)
            self.tasks.append(t)

    def sample(self) -> tuple[str, dict]:
        task = self.rng.choice(self.tasks)
        train = self._bundles[task].train
        picks = [self.rng.choice(train) for _ in range(min(self.batch_size, len(train)))]
        return self.lib[task]["description"], self.collate(picks)

    def __len__(self) -> int:
        return len(self.tasks)
