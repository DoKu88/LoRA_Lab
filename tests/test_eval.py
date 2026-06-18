"""Eval metric + table-schema tests (CPU-only)."""

import json

from lora_lab.eval.evaluate import _clean_generation
from lora_lab.eval.metrics import exact_match, normalize, rouge_l, score_predictions
from lora_lab.eval.table import COLUMNS, collect_rows, render_markdown


def test_clean_generation_first_nonempty_line():
    assert _clean_generation("positive\n\nNegative Example 1 - ...") == "positive"
    assert _clean_generation("\n\n  the answer  \nmore") == "the answer"
    assert _clean_generation("oneliner") == "oneliner"


# ---- metrics on a tiny known fixture --------------------------------------
def test_normalize():
    assert normalize("The  Cat.") == "cat"
    assert normalize("A dog!") == "dog"


def test_exact_match_multi_reference():
    assert exact_match("positive", ["positive"]) == 1.0
    assert exact_match("Positive.", ["positive"]) == 1.0  # normalized
    assert exact_match("4", ["four", "4"]) == 1.0  # any reference
    assert exact_match("negative", ["positive"]) == 0.0


def test_rouge_l_basic():
    assert rouge_l("the cat sat", ["the cat sat"]) == 1.0
    assert rouge_l("totally different", ["the cat sat"]) == 0.0
    partial = rouge_l("the cat", ["the cat sat on the mat"])
    assert 0.0 < partial < 1.0


def test_score_predictions_average():
    res = score_predictions(["positive", "negative"], [["positive"], ["positive"]], "exact_match")
    assert res["score"] == 0.5
    assert res["n"] == 2


def test_score_predictions_unknown_metric():
    import pytest

    with pytest.raises(ValueError):
        score_predictions(["x"], [["x"]], "bleu")


# ---- table schema ---------------------------------------------------------
def _write_summary(d, **kw):
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(kw))


def test_collect_rows_schema_and_order(tmp_path):
    runs = tmp_path / "runs"
    base = dict(base_model="org/Qwen2.5-0.5B-Instruct", task="task001",
                trainable_params=1000, pct_params=0.5, peak_vram_gb=4.0,
                wallclock_per_epoch_s=10.0, final_train_loss=0.5,
                eval_metric=0.8, eval_metric_name="exact_match", checkpoint_size_mb=7.0)
    _write_summary(runs / "lora-x", method="lora", **base)
    _write_summary(runs / "qlora-x", method="qlora", **base)
    _write_summary(runs / "full_ft-x", method="full_ft", **base)
    _write_summary(runs / "dry-x", method="lora", dry_run=True, **base)

    rows = collect_rows(runs)
    assert len(rows) == 3  # dry_run excluded
    # every row has exactly the schema columns
    for r in rows:
        assert set(r.keys()) == set(COLUMNS)
    # sorted qlora -> lora -> full_ft within a (model, task)
    assert [r["method"] for r in rows] == ["qlora", "lora", "full_ft"]


def test_render_markdown_has_header_and_rows(tmp_path):
    rows = [{c: None for c in COLUMNS}]
    rows[0].update(method="lora", base_model="org/M", task="t", trainable_params=10,
                   peak_vram_gb=4.0, eval_metric=0.5)
    md = render_markdown(rows)
    assert md.startswith("| method |")
    assert "lora" in md
    assert md.count("\n") >= 3
