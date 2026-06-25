"""Sprint 1 — build & validate the versioned library manifest.

The manifest (`configs/phase1/library.yaml`) is the single source of truth for
*which tasks are in the library* and *where every artifact for each comes from*.
We build it by listing the ``Lots-of-LoRAs`` org on the HuggingFace Hub:

  - adapters are repos named ``Mistral-7B-Instruct-v0.2-4b-r16-task<NUM>`` (the
    ``4b`` records that the base was loaded in 4-bit *during training*; the saved
    adapter is plain rank-16 LoRA that applies to the bf16 or 4-bit base);
  - datasets are repos named ``task<NUM>_<slug>`` (the full SNI task name).

So the task *number* is the join key (the name-normalization S3 calls out): an
adapter knows only ``task280``; its dataset/description carry the descriptive
slug. We record both, plus the per-task coverage, and never drop a task silently.

A deterministic **pilot subset** is flagged for the first end-to-end S4/S5/S6
pass; the full 1172-task sweep reuses the same manifest and is resumable.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = "configs/phase1/library.yaml"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EXPECTED_RANK = 16
EXPECTED_TARGETS = {"q_proj", "k_proj", "v_proj"}
ADAPTER_RE = re.compile(r"-task(\d+)$")
DATASET_RE = re.compile(r"^(task\d+)_")


@dataclass
class LibraryEntry:
    task_num: str                       # "task280" — the join key
    task_name: str                      # "task280_stereoset_classification_..." (dataset slug)
    adapter_repo: str
    dataset_repo: str
    rank: int = EXPECTED_RANK
    target_modules: list[str] = field(default_factory=lambda: sorted(EXPECTED_TARGETS))
    description: str = ""                # filled by descriptions.py (S3)
    description_source: str = ""         # "sni_definition" | "sakana" | ...
    kind: str = ""                       # classification | generation (inferred at gate, S4)
    metric: str = ""                     # exact_match | rougeL
    split_role: str = "unassigned"       # train | val | held-out (S5)
    pilot: bool = False                  # in the first end-to-end pass?
    adapter_hash: str = ""               # filled on fetch (S2)
    description_hash: str = ""           # filled by descriptions.py (S3)
    status: str = "ok"                   # ok | quarantined
    reason: str = ""                     # why quarantined / partial

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Hub discovery
# ---------------------------------------------------------------------------
def discover_from_hub() -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(adapter_by_num, dataset_by_num)`` from the Hub.

    ``adapter_by_num``  : {"task280": "Lots-of-LoRAs/Mistral-...-task280"}
    ``dataset_by_num``  : {"task280": "Lots-of-LoRAs/task280_stereoset_..."}
    """
    from huggingface_hub import HfApi

    api = HfApi()
    adapter_by_num: dict[str, str] = {}
    for m in api.list_models(author="Lots-of-LoRAs"):
        mm = ADAPTER_RE.search(m.id)
        if mm:
            adapter_by_num[f"task{int(mm.group(1))}"] = m.id

    dataset_by_num: dict[str, str] = {}
    for d in api.list_datasets(author="Lots-of-LoRAs"):
        slug = d.id.split("/", 1)[1]
        dm = DATASET_RE.match(slug)
        if dm:
            dataset_by_num[dm.group(1)] = d.id
    return adapter_by_num, dataset_by_num


# Deterministic pilot: a fixed spread of task *numbers* spanning the id range
# (so it samples diverse SNI task families), plus the five already-cached tasks
# from configs/tasks.yaml so the first run reuses local data where possible.
PILOT_TASK_NUMS = [
    "task020", "task022", "task039", "task190", "task280", "task290",
    "task379", "task391", "task442", "task620", "task512", "task843",
    "task1344", "task1564", "task639", "task1342", "task1391", "task033",
]


def build_manifest(pilot_nums: list[str] | None = None) -> list[LibraryEntry]:
    """Build the full candidate manifest; flag the pilot subset."""
    adapters, datasets = discover_from_hub()
    pilot = set(pilot_nums or PILOT_TASK_NUMS)
    entries: list[LibraryEntry] = []
    for num in sorted(set(adapters) | set(datasets), key=lambda t: int(t[4:])):
        a = adapters.get(num)
        d = datasets.get(num)
        if a and d:
            entries.append(LibraryEntry(
                task_num=num, task_name=d.split("/", 1)[1],
                adapter_repo=a, dataset_repo=d, pilot=(num in pilot),
            ))
        else:  # partial coverage — record, never drop silently
            entries.append(LibraryEntry(
                task_num=num, task_name=(d.split("/", 1)[1] if d else num),
                adapter_repo=a or "", dataset_repo=d or "",
                status="quarantined",
                reason="missing_adapter" if not a else "missing_dataset",
            ))
    return entries


# ---------------------------------------------------------------------------
# (de)serialization + validation
# ---------------------------------------------------------------------------
def save_manifest(entries: list[LibraryEntry], path: str | Path = DEFAULT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_model": BASE_MODEL,
        "expected_rank": EXPECTED_RANK,
        "expected_target_modules": sorted(EXPECTED_TARGETS),
        "version_hash": manifest_hash(entries),
        "tasks": {e.task_num: e.to_dict() for e in entries},
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def load_manifest(path: str | Path = DEFAULT_PATH) -> list[LibraryEntry]:
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    out = []
    for num, spec in (raw.get("tasks") or {}).items():
        spec = dict(spec)
        spec.pop("task_num", None)
        out.append(LibraryEntry(task_num=num, **spec))
    return out


def manifest_hash(entries: list[LibraryEntry]) -> str:
    """Content hash over the identity + provenance fields (not mutable status)."""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x.task_num):
        h.update(
            f"{e.task_num}|{e.adapter_repo}|{e.dataset_repo}|"
            f"{e.adapter_hash}|{e.description_hash}|{e.split_role}\x00".encode()
        )
    return h.hexdigest()[:16]


def coverage_report(entries: list[LibraryEntry]) -> dict[str, Any]:
    full = [e for e in entries if e.status == "ok"]
    quarantined = [e for e in entries if e.status != "ok"]
    reasons: dict[str, int] = {}
    for e in quarantined:
        reasons[e.reason] = reasons.get(e.reason, 0) + 1
    return {
        "candidate": len(entries),
        "full_coverage": len(full),
        "quarantined": len(quarantined),
        "quarantine_reasons": reasons,
        "pilot": sum(1 for e in entries if e.pilot),
        "with_description": sum(1 for e in full if e.description),
        "with_split_role": sum(1 for e in full if e.split_role != "unassigned"),
    }


def validate(entries: list[LibraryEntry]) -> None:
    """Assert the manifest invariants (S1 required testing)."""
    nums = [e.task_num for e in entries]
    assert len(nums) == len(set(nums)), "duplicate task numbers in manifest"
    for e in entries:
        assert e.split_role in ("train", "val", "held-out", "unassigned"), \
            f"{e.task_num}: bad split_role {e.split_role!r}"
        assert e.status in ("ok", "quarantined"), f"{e.task_num}: bad status"
        if e.status == "ok":
            assert e.adapter_repo and e.dataset_repo, \
                f"{e.task_num}: ok entry missing a repo"
