"""SNI data pipeline (Sprint 2)."""

from .sni import (
    DataCollatorForSupervised,
    DatasetBundle,
    TaskSpec,
    build_prompt,
    build_supervised,
    get_dataset,
    load_tasks_manifest,
    split_hash,
)

__all__ = [
    "DataCollatorForSupervised",
    "DatasetBundle",
    "TaskSpec",
    "build_prompt",
    "build_supervised",
    "get_dataset",
    "load_tasks_manifest",
    "split_hash",
]
