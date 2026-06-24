"""Sprint 3 — align each task with its natural-language description.

The description is the Phase-2 hypernetwork's *conditioning input*, so coverage
must be complete: a task with no description can't be a training example.

Source: every ``Lots-of-LoRAs`` SNI example embeds the canonical task definition
at the head of its ``input`` field as ``"Definition: <text> ..."`` followed by
the concrete instance (and sometimes "Positive Example"/"Output:" blocks). We
extract the definition span — this *is* the SNI ``Definition`` that T2L's task
descriptions are derived from — giving a clean, programmatic, 100%-coverage
source keyed by task number. (SakanaAI/text-to-lora's ``tasks/`` folder is an
alternative source, recorded as ``description_source`` when used.)
"""

from __future__ import annotations

import re

# The instance/example markers that follow the Definition in an SNI input.
_STOP_MARKERS = [
    "\nPositive Example", "\nNegative Example", "\nExample ",
    "\nInput:", "\nNow, ", "\nGiven ", "\nQ:", "\nQuestion:",
]


def extract_definition(input_text: str) -> str:
    """Pull the ``Definition:`` span out of one SNI input string.

    Returns the definition text (without the literal "Definition:" label), or
    "" if the input doesn't carry one in the expected form.
    """
    if not input_text:
        return ""
    m = re.search(r"Definition\s*:\s*", input_text)
    if not m:
        return ""
    rest = input_text[m.end():]
    # Cut at the first instance/example marker so we keep only the definition.
    cut = len(rest)
    for marker in _STOP_MARKERS:
        idx = rest.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return " ".join(rest[:cut].split()).strip()


def description_for_task(dataset_repo: str, *, n_probe: int = 8) -> str:
    """Load a few examples of a task and return its (majority) definition.

    SNI keeps the definition constant across a task's examples; we probe a
    handful and take the most common non-empty extraction to be robust to the
    odd malformed row.
    """
    import datasets as hfds

    hfds.disable_progress_bars()
    ds = hfds.load_dataset(dataset_repo)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = ds[split].select(range(min(n_probe, len(ds[split]))))
    counts: dict[str, int] = {}
    for ex in rows:
        d = extract_definition(ex.get("input", ""))
        if d:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return ""
    return max(counts, key=counts.get)
