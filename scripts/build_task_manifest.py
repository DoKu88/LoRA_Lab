#!/usr/bin/env python
"""Pin reproducible split hashes into configs/tasks.yaml.

Computes the SHA-256 hash of each task's *full* train/valid/test split (over
the ordered example ids) and writes them back under ``split_hashes``. These
are stable properties of the source data, so a later load that reproduces
them proves the split is the one we versioned (S2 definition of done).

    conda run -n lora_lab python scripts/build_task_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from lora_lab.data.sni import load_tasks_manifest, split_hash  # noqa: E402

MANIFEST = Path(__file__).resolve().parents[1] / "configs" / "tasks.yaml"


def main() -> int:
    import datasets as hfds

    hfds.disable_progress_bars()

    specs = load_tasks_manifest(MANIFEST)
    with MANIFEST.open() as f:
        doc = yaml.safe_load(f)

    for name, spec in specs.items():
        raw = hfds.load_dataset(spec.hf_repo)
        hashes = {}
        sizes = {}
        for split in raw:
            hashes[split] = split_hash(list(raw[split]["id"]))
            sizes[split] = len(raw[split])
        doc["tasks"][name]["split_hashes"] = hashes
        doc["tasks"][name]["split_sizes"] = sizes
        print(f"{name}: sizes={sizes} hashes={hashes}")

    with MANIFEST.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False)
    print(f"\nWrote split hashes to {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
