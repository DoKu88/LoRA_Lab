"""Sprint 5 — lock the frozen train / val / held-out split.

The held-out split is *sacred*: tasks here are never seen by the Phase-2
hypernetwork during training, so leakage invalidates every downstream gate.
We partition the gate-passing library, content-hash the held-out task-id set
(the lock), and assert the leakage guard (no held-out id in train).

T2L follows Brüel-Gabrielsson et al.: 479 train / 11 val / 10 contamination-
removed out of a 500-task English subset. We mirror the *proportions* over
whatever gate-passing tasks we actually cover, and record any deviation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = "configs/phase1/heldout_split.yaml"

# Known SNI contamination removals (T2L drops these from training). Recorded so
# they are absent from all three buckets even if they cleared the quality gate.
CONTAMINATION = [
    # task ids flagged by Brüel-Gabrielsson et al. as overlapping eval sets;
    # extend as the full list is reconciled (pilot carries none of these).
]


@dataclass
class Split:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    held_out: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)  # contamination
    seed: int = 42

    def lock_hash(self) -> str:
        """Content hash over the held-out id set (order-independent) — the lock."""
        h = hashlib.sha256()
        for t in sorted(self.held_out):
            h.update(f"{t}\x00".encode())
        return h.hexdigest()[:16]

    def assert_valid(self) -> None:
        s_tr, s_va, s_ho = set(self.train), set(self.val), set(self.held_out)
        assert s_tr.isdisjoint(s_va), "train/val overlap"
        assert s_tr.isdisjoint(s_ho), "LEAKAGE: held-out id in train"
        assert s_va.isdisjoint(s_ho), "val/held-out overlap"
        removed = set(self.removed)
        assert removed.isdisjoint(s_tr | s_va | s_ho), \
            "contamination-removed task present in a live bucket"


def make_split(passing_task_nums: list[str], *, n_val: int = 1, n_heldout: int = 3,
               seed: int = 42) -> Split:
    """Partition gate-passing tasks. Held-out + val are sampled deterministically;
    the rest are train. (Pilot uses small n_val/n_heldout; the full run scales to
    T2L's 11 / 10.)"""
    pool = [t for t in sorted(set(passing_task_nums)) if t not in set(CONTAMINATION)]
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    held_out = sorted(shuffled[:n_heldout])
    val = sorted(shuffled[n_heldout:n_heldout + n_val])
    train = sorted(set(pool) - set(held_out) - set(val))
    sp = Split(train=train, val=val, held_out=held_out,
               removed=list(CONTAMINATION), seed=seed)
    sp.assert_valid()
    return sp


def save_split(sp: Split, names: dict[str, str], path: str | Path = DEFAULT_PATH) -> Path:
    """Persist the locked split. ``names`` maps task_num -> full task_name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lock_hash": sp.lock_hash(),
        "seed": sp.seed,
        "counts": {"train": len(sp.train), "val": len(sp.val),
                   "held_out": len(sp.held_out), "removed": len(sp.removed)},
        "held_out": {t: names.get(t, t) for t in sp.held_out},
        "val": {t: names.get(t, t) for t in sp.val},
        "removed": sp.removed,
        "train": {t: names.get(t, t) for t in sp.train},
    }
    with path.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
    return path


def load_split(path: str | Path = DEFAULT_PATH) -> Split:
    with Path(path).open() as f:
        raw = yaml.safe_load(f) or {}
    return Split(
        train=list((raw.get("train") or {}).keys()),
        val=list((raw.get("val") or {}).keys()),
        held_out=list((raw.get("held_out") or {}).keys()),
        removed=list(raw.get("removed") or []),
        seed=raw.get("seed", 42),
    )
