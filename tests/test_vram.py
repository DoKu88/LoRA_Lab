"""VRAM helper unit tests (CPU-safe — no GPU required)."""

from pathlib import Path

from lora_lab.utils.vram import (
    MemoryTracer,
    bytes_to_gb,
    cuda_mem_snapshot,
    phase_memory,
)


def test_bytes_to_gb():
    assert bytes_to_gb(1024**3) == 1.0
    assert bytes_to_gb(0) == 0.0


def test_snapshot_keys_present():
    snap = cuda_mem_snapshot()
    assert set(snap) == {"allocated_gb", "reserved_gb", "max_allocated_gb"}
    assert all(v >= 0.0 for v in snap.values())


def test_phase_memory_populates():
    with phase_memory("x") as info:
        pass
    assert info["label"] == "x"
    assert info["peak_gb"] >= 0.0


def test_tracer_records_and_peak():
    t = MemoryTracer()
    for step in range(5):
        t.record(step)
    assert len(t) == 5
    assert t.steps == [0, 1, 2, 3, 4]
    # On CPU the trace is all-zeros; peak is still well-defined.
    assert t.peak_gb >= 0.0


def test_tracer_peak_is_max():
    t = MemoryTracer()
    t.steps = [0, 1, 2]
    t.allocated_gb = [1.0, 3.5, 2.0]
    t.reserved_gb = [1.0, 4.0, 2.0]
    assert t.peak_gb == 3.5
    assert t.peak_reserved_gb == 4.0


def test_tracer_save_csv(tmp_path: Path):
    t = MemoryTracer()
    t.steps = [0, 10]
    t.allocated_gb = [0.5, 0.75]
    t.reserved_gb = [0.6, 0.8]
    out = t.save_csv(tmp_path / "trace.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "step,gpu_mem_gb,gpu_mem_reserved_gb"
    assert lines[1].startswith("0,0.5")
    assert len(lines) == 3
