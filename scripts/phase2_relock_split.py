#!/usr/bin/env python
"""Phase 2 — re-lock the train/val/held-out split (curated, supersedes the pilot).

Builds the curated 30-task held-out set across three generalization axes where the
retrieval baseline is plausible-but-wrong (sizing-options doc / sprint-plan §2):

  format  (15): hold out the *generation* form of a dataset whose *classification*
                form is trained (retrieval grabs the same topic, wrong format).
  language (8): hold out a translation direction whose language is otherwise
                trained (retrieval returns a different-language LoRA).
  domain   (7): hold out a sentiment/emotion domain, train the other domains.

Partners of every held-out task (the trained cls form / reverse directions /
other domains) are forced INTO train so each hold-out is diagnostic. Train fills
to ~400 broadly across the remaining gate-passing tasks; 10 val; the rest reserved.

    python scripts/phase2_relock_split.py            # dry-run: print the picks
    python scripts/phase2_relock_split.py --lock      # write heldout_split.yaml
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_lab.library.split import Split, save_split  # noqa: E402

GATE = "results/phase1/gate_results.jsonl"
N_FORMAT, N_LANG, N_DOMAIN, N_VAL, N_TRAIN, SEED = 15, 8, 7, 10, 400, 42

CLS = {"classification", "identification", "detection", "polarity", "recognition",
       "categorization", "answerability", "disambiguation"}
GEN = {"generation", "answer", "answering", "summarization", "simplification",
       "paraphrasing", "paraphrase"}
FORMS = CLS | GEN


def _toks(name): return re.sub(r"^task\d+_", "", name).split("_")
def _stem(name):
    t = _toks(name)
    while t and t[-1] in FORMS:
        t.pop()
    return "_".join(t)
def _form(name):
    t = set(_toks(name))
    if t & CLS:
        return "cls"
    if t & GEN:
        return "gen"
    return "other"


def load_passing():
    rows = [json.loads(l) for l in open(GATE) if l.strip()]
    return {r["task_num"]: r["task_name"] for r in rows if r.get("gate") == "pass"}


def curate(names):
    """Return (held_out:set, forced_train:set, axis_map:dict[num->axis])."""
    held, forced, axis = set(), set(), {}

    # --- format: 15 stems with both cls + gen forms ----------------------
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for num, n in names.items():
        by[_stem(n)][_form(n)].append(num)
    pairs = sorted(s for s, d in by.items() if d["cls"] and d["gen"])
    for stem in pairs[:N_FORMAT]:
        gen_num = sorted(by[stem]["gen"])[0]
        cls_num = sorted(by[stem]["cls"])[0]
        held.add(gen_num); axis[gen_num] = "format"
        forced.add(cls_num)

    # --- language: 8 translation directions whose language is trained ----
    trans = {num: n for num, n in names.items() if "translation" in n}
    # direction = (src, tgt) ~ last two tokens; pick held-out dirs whose REVERSE exists
    def direction(n):
        t = _toks(n)
        return (t[-2], t[-1]) if len(t) >= 2 else (t[-1], "")
    dirs = {num: direction(n) for num, n in trans.items()}
    rev_index = collections.defaultdict(list)
    for num, (a, b) in dirs.items():
        rev_index[(a, b)].append(num)
    picked = []
    for num, (a, b) in sorted(dirs.items()):
        if len(picked) >= N_LANG:
            break
        # hold out this direction only if its REVERSE exists and is NOT itself
        # held out (so the reverse stays trainable — a same-language competitor
        # for retrieval). This makes it direction-transfer, not leave-language-out.
        reverse = [r for r in rev_index.get((b, a), []) if r not in held]
        if reverse and num not in held:
            held.add(num); axis[num] = "language"; picked.append(num)
            forced.add(sorted(reverse)[0])  # train the reverse direction
    # ensure broad translation coverage in train
    for num in sorted(trans)[:40]:
        if num not in held:
            forced.add(num)

    # --- domain: hold out the FAR domains, train amazon/yelp/sentiment140 -
    # (the doc's intent: train big sentiment domains, hold out the distant ones so
    # retrieval lands on a same-skill-other-domain LoRA — a tougher, fair competitor).
    DOMAIN_FAR = ["task512", "task843", "task1496", "task1497", "task833",
                  "task819", "task1575"]   # twitter_emotion, financial_phrasebank,
                                           # bengali x2, poem, pec, amazon_multilingual
    DOMAIN_TRAIN = ["task1312", "task1313", "task475", "task195", "task493"]  # amazon/yelp/sent140
    for num in DOMAIN_FAR:
        if num in names and num not in held:
            held.add(num); axis[num] = "domain"
    for num in DOMAIN_TRAIN:
        if num in names and num not in held:
            forced.add(num)

    forced -= held
    return held, forced, axis


def build_split(names):
    held, forced, axis = curate(names)
    pool = [t for t in sorted(names) if t not in held]
    forced = [t for t in forced if t in set(pool)]
    rng = random.Random(SEED)
    rest = [t for t in pool if t not in set(forced)]
    rng.shuffle(rest)
    n_fill = max(0, N_TRAIN - len(forced))
    train = sorted(set(forced) | set(rest[:n_fill]))
    val = sorted(rest[n_fill:n_fill + N_VAL])
    sp = Split(train=train, val=val, held_out=sorted(held), removed=[], seed=SEED)
    sp.assert_valid()
    return sp, axis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lock", action="store_true", help="write configs/phase1/heldout_split.yaml")
    args = ap.parse_args()
    names = load_passing()
    sp, axis = build_split(names)

    by_axis = collections.defaultdict(list)
    for num in sp.held_out:
        by_axis[axis.get(num, "?")].append(num)
    print(f"=== Curated held-out ({len(sp.held_out)} tasks) — lock_hash={sp.lock_hash()} ===")
    for ax in ("format", "language", "domain"):
        print(f"\n[{ax}] ({len(by_axis[ax])})")
        for num in by_axis[ax]:
            print(f"   HOLD-OUT {num:>9}  {names[num]}")
    print(f"\ntrain={len(sp.train)}  val={len(sp.val)}  held_out={len(sp.held_out)}  "
          f"reserved={len(names) - len(sp.train) - len(sp.val) - len(sp.held_out)}")

    if args.lock:
        path = save_split(sp, names)
        print(f"\n[locked] wrote {path}  lock_hash={sp.lock_hash()}")
    else:
        print("\n(dry-run — rerun with --lock to write heldout_split.yaml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
