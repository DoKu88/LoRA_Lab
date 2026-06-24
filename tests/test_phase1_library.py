"""Phase 1 library invariants (offline, no network/GPU).

Covers the per-sprint "Required testing" that doesn't need the Hub or the 7B:
manifest validation (S1), description extraction (S3), adapter compatibility +
metric inference (S2/S4), and the split leakage guard (S5).
"""

import pytest

from lora_lab.library import manifest as M
from lora_lab.library import split as SP
from lora_lab.library.descriptions import extract_definition
from lora_lab.library.gate import assert_compatible, infer_kind_metric


# ---- S1: manifest ---------------------------------------------------------
def _entry(num, **kw):
    return M.LibraryEntry(task_num=num, task_name=f"{num}_x",
                          adapter_repo=f"A/{num}", dataset_repo=f"D/{num}", **kw)


def test_manifest_round_trip(tmp_path):
    entries = [_entry("task1", description="d", split_role="train"),
               _entry("task2", status="quarantined", reason="missing_adapter")]
    p = M.save_manifest(entries, tmp_path / "lib.yaml")
    back = M.load_manifest(p)
    assert {e.task_num for e in back} == {"task1", "task2"}
    assert M.manifest_hash(back) == M.manifest_hash(entries)


def test_manifest_validate_rejects_bad_role():
    e = _entry("task1", split_role="bogus")
    with pytest.raises(AssertionError):
        M.validate([e])


def test_manifest_no_duplicate_nums():
    with pytest.raises(AssertionError):
        M.validate([_entry("task1"), _entry("task1")])


def test_coverage_reconciles():
    entries = [_entry("task1"), _entry("task2", status="quarantined", reason="x")]
    rep = M.coverage_report(entries)
    assert rep["candidate"] == rep["full_coverage"] + rep["quarantined"]


# ---- S3: description extraction -------------------------------------------
def test_extract_definition_basic():
    inp = ("Definition: Classify the sentiment as positive or negative.\n"
           "Positive Example 1 - Input: great! Output: positive\n"
           "Now complete: terrible")
    d = extract_definition(inp)
    assert d == "Classify the sentiment as positive or negative."


def test_extract_definition_absent():
    assert extract_definition("no definition here, just text") == ""


# ---- S2/S4: compatibility + metric inference ------------------------------
def _cfg(base=M.BASE_MODEL, r=16, targets=("q_proj", "k_proj", "v_proj")):
    return {"base_model_name_or_path": base, "r": r, "target_modules": list(targets)}


def test_compatible_accepts_and_returns_rank():
    assert assert_compatible(_cfg(r=43), "task1") == 43


def test_compatible_rejects_wrong_base():
    with pytest.raises(AssertionError):
        assert_compatible(_cfg(base="meta-llama/Llama-3.2-1B"), "task1")


def test_compatible_rejects_wrong_targets():
    with pytest.raises(AssertionError):
        assert_compatible(_cfg(targets=("q_proj", "o_proj")), "task1")


def test_infer_metric_classification():
    ev = [{"references": ["positive"]}, {"references": ["negative"]},
          {"references": ["neutral"]}] * 10
    assert infer_kind_metric(ev) == ("classification", "exact_match")


def test_infer_metric_generation():
    ev = [{"references": [f"a long free form answer number {i} with many words"]}
          for i in range(20)]
    assert infer_kind_metric(ev) == ("generation", "rougeL")


# ---- S5: split leakage guard ----------------------------------------------
def test_split_leakage_guard_and_disjoint():
    sp = SP.make_split([f"task{i}" for i in range(20)], n_val=2, n_heldout=3, seed=42)
    sp.assert_valid()  # raises on any overlap / leakage
    assert set(sp.held_out).isdisjoint(sp.train)
    assert len(sp.held_out) == 3 and len(sp.val) == 2


def test_split_lock_hash_stable():
    nums = [f"task{i}" for i in range(20)]
    h1 = SP.make_split(nums, seed=42).lock_hash()
    h2 = SP.make_split(nums, seed=42).lock_hash()
    assert h1 == h2


def test_split_contamination_excluded(monkeypatch):
    monkeypatch.setattr(SP, "CONTAMINATION", ["task5"])
    sp = SP.make_split([f"task{i}" for i in range(20)], seed=1)
    assert "task5" not in (set(sp.train) | set(sp.val) | set(sp.held_out))
