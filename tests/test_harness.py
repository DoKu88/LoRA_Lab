"""Sprint 3 harness tests — run on CPU, no GPU required."""

import json

from lora_lab.config import RunConfig, apply_overrides
from lora_lab.train.dryrun import run_dry


def test_apply_overrides_coercion():
    d = {"method": "lora"}
    apply_overrides(d, ["hparams.max_steps=10", "hparams.lr=0.0003", "logging.wandb_mode=disabled"])
    assert d["hparams"]["max_steps"] == 10
    assert d["hparams"]["lr"] == 0.0003
    assert d["logging"]["wandb_mode"] == "disabled"


def test_override_bad_format_raises():
    import pytest

    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


def test_dry_run_produces_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results/mem_trace lands under tmp
    cfg = RunConfig(
        method="qlora",
        base_model="HuggingFaceTB/SmolLM2-135M",
        task="task1564_triviaqa_answer_generation",
        output_root=str(tmp_path / "runs"),
    )
    cfg.logging.wandb_mode = "disabled"
    summary = run_dry(cfg, n_steps=15)

    run_dir = cfg.output_dir
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "summary.json").exists()

    # metrics.jsonl has one line per step, each with the required fields
    lines = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 15
    rec = json.loads(lines[0])
    for key in ("step", "train_loss", "gpu_mem_gb", "tokens_per_sec", "step_time_s"):
        assert key in rec

    # loss decreases over the run
    first = json.loads(lines[0])["train_loss"]
    last = json.loads(lines[-1])["train_loss"]
    assert last < first

    # memory trace persisted to results/mem_trace/
    trace = tmp_path / "results" / "mem_trace" / f"{cfg.name}.csv"
    assert trace.exists()
    assert trace.read_text().splitlines()[0] == "step,gpu_mem_gb,gpu_mem_reserved_gb"

    assert summary["peak_vram_gb"] > 0
    assert summary["method"] == "qlora"


def test_count_parameters_on_tiny_module():
    import torch.nn as nn

    from lora_lab.train.params import count_parameters

    model = nn.Linear(10, 5)  # 10*5 + 5 = 55 params, all trainable
    for p in model.parameters():
        p.requires_grad_(True)
    info = count_parameters(model)
    assert info["total_params"] == 55
    assert info["trainable_params"] == 55
    assert abs(info["pct_params"] - 100.0) < 1e-6

    # freeze the bias -> trainable drops
    model.bias.requires_grad_(False)
    info2 = count_parameters(model)
    assert info2["trainable_params"] == 50
    assert info2["total_params"] == 55
