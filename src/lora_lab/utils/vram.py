"""GPU memory accounting helpers.

Two jobs:

1. **Per-region peak** — wrap a code region and read back its peak
   ``torch.cuda.max_memory_allocated()`` so we know *which pool* blew up.
2. **Memory-vs-iteration trace** — sample allocated/reserved GB at each
   logged step so the run plots as a memory-vs-iteration curve, and the
   per-run peak is just ``max`` of that trace.

Everything degrades gracefully when CUDA is absent so it works on a CPU-only box.
"""

from __future__ import annotations

import csv
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:  # torch is optional at import time (dry-run / CPU CI)
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch always present in the real env
    _HAS_TORCH = False

try:  # psutil is optional (CPU CI); host-RAM probe degrades to 0.0 without it
    import psutil

    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - psutil present in the real env
    _HAS_PSUTIL = False


_BYTES_PER_GB = 1024**3


def bytes_to_gb(n: int) -> float:
    """Bytes -> gibibytes (GiB), the unit we report and plot everywhere."""
    return float(n) / _BYTES_PER_GB


def cuda_available() -> bool:
    return _HAS_TORCH and torch.cuda.is_available()


def device_capability(device: int = 0) -> tuple[int, int] | None:
    """Return the CUDA compute capability, e.g. (12, 0) for sm_120."""
    if not cuda_available():
        return None
    return torch.cuda.get_device_capability(device)


def is_blackwell_sm120(device: int = 0) -> bool:
    """True iff the active CUDA device is sm_120 (Blackwell / RTX 5090)."""
    cap = device_capability(device)
    return cap == (12, 0)


def reset_peak_memory(device: int | None = None) -> None:
    """Reset the CUDA peak-memory counter so the next phase reads clean."""
    if cuda_available():
        torch.cuda.reset_peak_memory_stats(device)


def cuda_mem_snapshot(device: int | None = None) -> dict[str, float]:
    """Current allocated/reserved and the running peak, all in GiB.

    On a CPU-only box every field is 0.0 so callers never branch on CUDA.
    """
    if not cuda_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
    return {
        "allocated_gb": bytes_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_gb": bytes_to_gb(torch.cuda.memory_reserved(device)),
        "max_allocated_gb": bytes_to_gb(torch.cuda.max_memory_allocated(device)),
    }


@contextmanager
def phase_memory(label: str, device: int | None = None) -> Iterator[dict]:
    """Measure the peak allocated memory of a code region.

    Usage::

        with phase_memory("forward") as m:
            model(**batch)
        print(m["peak_gb"])

    The dict is populated on exit; ``label`` is echoed back for logging.
    """
    reset_peak_memory(device)
    info: dict = {"label": label, "peak_gb": 0.0, "reserved_gb": 0.0}
    try:
        yield info
    finally:
        snap = cuda_mem_snapshot(device)
        info["peak_gb"] = snap["max_allocated_gb"]
        info["reserved_gb"] = snap["reserved_gb"]


@dataclass
class MemoryTracer:
    """Accumulate a (step, gpu_mem_gb) time series across training.

    Sample once per logged step via :meth:`record`. ``peak_gb`` is the max of
    the allocated trace and feeds the ``peak_vram_gb`` summary;
    :meth:`save_csv` persists the raw trace to ``results/mem_trace/`` so the
    overlaid memory-vs-iteration plot can be re-rendered offline.
    """

    device: int | None = None
    steps: list[int] = field(default_factory=list)
    allocated_gb: list[float] = field(default_factory=list)
    reserved_gb: list[float] = field(default_factory=list)

    def record(self, step: int, peak: bool = True) -> dict[str, float]:
        """Sample memory now and append to the trace; returns the sample.

        ``peak=True`` (default) records the *peak* allocated/reserved since the
        last sample, then resets the CUDA peak counters. Because the counter
        resets to the current resident value, every point still includes the
        resident model weights — so the trace is a true memory *envelope* vs.
        iteration (captures the backward-pass spike), and ``max`` over it is the
        real per-run peak. ``peak=False`` records the instantaneous value.
        """
        if peak and cuda_available():
            alloc = bytes_to_gb(torch.cuda.max_memory_allocated(self.device))
            resv = bytes_to_gb(torch.cuda.max_memory_reserved(self.device))
            torch.cuda.reset_peak_memory_stats(self.device)
            sample = {"allocated_gb": alloc, "reserved_gb": resv}
        else:
            snap = cuda_mem_snapshot(self.device)
            sample = {"allocated_gb": snap["allocated_gb"], "reserved_gb": snap["reserved_gb"]}
        self.steps.append(int(step))
        self.allocated_gb.append(sample["allocated_gb"])
        self.reserved_gb.append(sample["reserved_gb"])
        return sample

    @property
    def peak_gb(self) -> float:
        """Per-run peak = max of the allocated trace (0.0 if empty/CPU)."""
        return max(self.allocated_gb) if self.allocated_gb else 0.0

    @property
    def peak_reserved_gb(self) -> float:
        return max(self.reserved_gb) if self.reserved_gb else 0.0

    def __len__(self) -> int:
        return len(self.steps)

    def save_csv(self, path: str | Path) -> Path:
        """Persist the trace as ``step,gpu_mem_gb,gpu_mem_reserved_gb``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "gpu_mem_gb", "gpu_mem_reserved_gb"])
            for s, a, r in zip(self.steps, self.allocated_gb, self.reserved_gb):
                writer.writerow([s, f"{a:.6f}", f"{r:.6f}"])
        return path


def process_ram_bytes(include_children: bool = True) -> int:
    """Resident set size (RSS) of this process (+ children), in bytes.

    Children matter when CPU-offload and dataloader workers run optimizer state
    and pinned buffers in child processes, so the parent's RSS alone undercounts
    the host-RAM footprint. Returns 0 when psutil is unavailable.
    """
    if not _HAS_PSUTIL:
        return 0
    proc = psutil.Process()
    total = proc.memory_info().rss
    if include_children:
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
                continue
    return total


def host_ram_available() -> bool:
    return _HAS_PSUTIL


@dataclass
class HostRamTracer:
    """Sample host (system) RAM and capture its peak across a run.

    When CPU-offload is in play, peak *RAM* matters alongside peak VRAM (the
    offloaded optimizer state lives in host memory). Unlike the GPU peak (which
    CUDA tracks for us between
    samples), host RSS has no built-in high-water mark, so a **background
    sampler thread** polls it every ``interval_s`` to catch the spikes that land
    *between* logged training steps (e.g. the offloaded optimizer step on CPU).

    Usage::

        ram = HostRamTracer().start()
        ...train...
        ram.record(step)        # step-aligned point for the plot
        ...
        ram.stop()
        print(ram.peak_ram_gb)  # true high-water mark, incl. between-step spikes

    Degrades to all-zeros (never raises) when psutil is absent so the harness,
    dry-run and CPU tests still work.
    """

    interval_s: float = 0.5
    include_children: bool = True
    # step-aligned trace (for the RAM-vs-iteration plot)
    steps: list[int] = field(default_factory=list)
    ram_gb: list[float] = field(default_factory=list)
    # continuous high-water mark from the background thread
    _peak_bytes: int = 0
    _baseline_bytes: int = 0
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_evt: threading.Event | None = field(default=None, repr=False)

    def _sample_bytes(self) -> int:
        return process_ram_bytes(self.include_children)

    def _run(self) -> None:  # pragma: no cover - timing-dependent thread body
        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            self._peak_bytes = max(self._peak_bytes, self._sample_bytes())
            self._stop_evt.wait(self.interval_s)

    def start(self) -> "HostRamTracer":
        """Snapshot the baseline RSS and launch the background sampler."""
        now = self._sample_bytes()
        self._baseline_bytes = now
        self._peak_bytes = now
        if _HAS_PSUTIL:
            self._stop_evt = threading.Event()
            self._thread = threading.Thread(
                target=self._run, name="host-ram-sampler", daemon=True
            )
            self._thread.start()
        return self

    def record(self, step: int) -> float:
        """Append a step-aligned RAM sample (GiB) to the trace; returns it.

        Also folds the reading into the running peak so the high-water mark is
        correct even if the background thread is disabled.
        """
        b = self._sample_bytes()
        self._peak_bytes = max(self._peak_bytes, b)
        gb = bytes_to_gb(b)
        self.steps.append(int(step))
        self.ram_gb.append(gb)
        return gb

    def stop(self) -> "HostRamTracer":
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 4)
            self._thread = None
        return self

    @property
    def peak_ram_gb(self) -> float:
        """Peak total RSS (GiB) seen over the run (0.0 without psutil)."""
        return bytes_to_gb(self._peak_bytes)

    @property
    def baseline_ram_gb(self) -> float:
        """RSS (GiB) at ``start()`` — subtract for the run's *marginal* RAM."""
        return bytes_to_gb(self._baseline_bytes)

    @property
    def peak_ram_delta_gb(self) -> float:
        """Peak RAM above the start baseline — the run's own RAM footprint."""
        return bytes_to_gb(max(0, self._peak_bytes - self._baseline_bytes))

    def __len__(self) -> int:
        return len(self.steps)

    def __enter__(self) -> "HostRamTracer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def save_csv(self, path: str | Path) -> Path:
        """Persist the step-aligned trace as ``step,ram_gb``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "ram_gb"])
            for s, r in zip(self.steps, self.ram_gb):
                writer.writerow([s, f"{r:.6f}"])
        return path
