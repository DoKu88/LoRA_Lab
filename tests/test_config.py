"""RunConfig round-trip and validation tests."""

import pytest

from lora_lab.config import RunConfig


def test_round_trip(tmp_path):
    c = RunConfig(method="qlora", base_model="Qwen/Qwen2.5-0.5B-Instruct", task="task001")
    p = c.save(tmp_path / "run.yaml")
    c2 = RunConfig.load(p)
    assert c.to_dict() == c2.to_dict()


def test_derived_name_and_dir():
    c = RunConfig(method="lora", base_model="HuggingFaceTB/SmolLM2-135M", task="task001")
    assert c.name == "lora-SmolLM2-135M-task001"
    assert str(c.output_dir) == "results/runs/lora-SmolLM2-135M-task001"


def test_explicit_run_name_wins():
    c = RunConfig(run_name="custom")
    assert c.name == "custom"


def test_invalid_method_raises():
    with pytest.raises(ValueError):
        RunConfig(method="bogus")


def test_invalid_wandb_mode_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"logging": {"wandb_mode": "nope"}})


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        RunConfig.from_dict({"method": "lora", "bogus_key": 1})


def test_nested_from_dict_types():
    c = RunConfig.from_dict(
        {"method": "qlora", "hparams": {"lr": 1e-3, "batch_size": 2}}
    )
    assert c.hparams.lr == 1e-3
    assert c.hparams.batch_size == 2
    # untouched nested fields keep their defaults
    assert c.lora.r == 8
