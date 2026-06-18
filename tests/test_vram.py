"""VRAM helper unit tests (CPU-safe — no GPU required)."""

from pathlib import Path

import pytest

from lora_lab.utils.vram import (
    HostRamTracer,
    MemoryTracer,
    bytes_to_gb,
    cuda_mem_snapshot,
    host_ram_available,
    phase_memory,
    process_ram_bytes,
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


# --- Host-RAM probe (Sprint 1) ----------------------------------------------


def test_process_ram_nonneg():
    b = process_ram_bytes()
    assert b >= 0
    if host_ram_available():
        # a live python process resident set is comfortably over 1 MB
        assert b > 1024**2


def test_host_ram_tracer_records_and_trace():
    ram = HostRamTracer(interval_s=0.02).start()
    try:
        for step in range(3):
            ram.record(step)
    finally:
        ram.stop()
    assert len(ram) == 3
    assert ram.steps == [0, 1, 2]
    assert ram.peak_ram_gb >= 0.0
    assert ram.peak_ram_delta_gb >= 0.0


def test_host_ram_tracer_save_csv(tmp_path: Path):
    ram = HostRamTracer()
    ram.steps = [0, 5]
    ram.ram_gb = [1.25, 1.5]
    out = ram.save_csv(tmp_path / "ram.csv")
    lines = out.read_text().strip().splitlines()
    assert lines[0] == "step,ram_gb"
    assert lines[1].startswith("0,1.25")
    assert len(lines) == 3


@pytest.mark.skipif(not host_ram_available(), reason="psutil required")
def test_host_ram_probe_detects_known_allocation():
    """Allocate ~50 MB and confirm the probe's peak rises by roughly that much.

    This is the Sprint 1 'RAM probe validated against a known allocation' test:
    we don't demand exactness (the allocator/GC add slop), only that a real
    multi-MB allocation is clearly reflected in the peak.
    """
    ram = HostRamTracer(interval_s=0.01).start()
    blob = None
    try:
        # bytearray is a single contiguous resident allocation (not lazy)
        blob = bytearray(50 * 1024 * 1024)
        blob[:: 4096] = b"\x01" * len(blob[:: 4096])  # touch pages so they're resident
        for step in range(3):
            ram.record(step)
    finally:
        ram.stop()
    assert ram.peak_ram_delta_gb >= bytes_to_gb(30 * 1024 * 1024)  # >= ~30 MB of the 50
    assert blob is not None and len(blob) == 50 * 1024 * 1024
