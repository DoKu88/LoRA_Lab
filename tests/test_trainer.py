"""Trainer-helper unit tests (CPU-only)."""

from pathlib import Path

from lora_lab.config import RunConfig
from lora_lab.train.trainer import _dir_size_mb, _num_training_steps


def _cfg(**hp):
    c = RunConfig(method="lora")
    for k, v in hp.items():
        setattr(c.hparams, k, v)
    return c


def test_num_steps_uses_max_steps_when_set():
    c = _cfg(max_steps=37, batch_size=4, grad_accum=1, num_epochs=5)
    assert _num_training_steps(1000, c) == 37


def test_num_steps_from_epochs():
    # 100 examples, eff batch 10 -> 10 steps/epoch * 2 epochs = 20
    c = _cfg(max_steps=-1, batch_size=5, grad_accum=2, num_epochs=2)
    assert _num_training_steps(100, c) == 20


def test_num_steps_grad_accum_rounds_up():
    # 100 examples, eff batch 32 -> ceil(100/32)=4 steps/epoch * 1 = 4
    c = _cfg(max_steps=-1, batch_size=16, grad_accum=2, num_epochs=1)
    assert _num_training_steps(100, c) == 4


def test_num_steps_min_one():
    c = _cfg(max_steps=-1, batch_size=256, grad_accum=1, num_epochs=1)
    assert _num_training_steps(10, c) == 1


def test_dir_size_mb(tmp_path: Path):
    (tmp_path / "a.bin").write_bytes(b"\x00" * (1024 * 1024))
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\x00" * (512 * 1024))
    assert abs(_dir_size_mb(tmp_path) - 1.5) < 0.01
