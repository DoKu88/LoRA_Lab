"""Phase 0.5 benchmark entrypoint — one fixed protocol, many techniques.

``benchmark(config)`` runs the fixed measurement protocol with the strategy
selected by ``config.technique.name`` and returns the trade-off-table row
(peak VRAM, peak RAM, wall-clock/step, tokens/s, fits, ...). Every strategy
goes through the same instrumentation (GPU + host-RAM tracers, W&B) so the rows
stay apples-to-apples.

Each technique is a *strategy* registered in ``STRATEGIES``. Sprints 2-5 fill
these in one at a time; until then a technique raises a clear NotImplementedError
naming its sprint, so the entrypoint and dispatch are testable now and the
overnight runner fails a single technique gracefully (logged, skipped) rather
than crashing the batch.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

from ..config import RunConfig

# Techniques that run the optimizer step on CPU via DeepSpeed and therefore need
# CUDA_HOME pointed at a real toolkit so the CPUAdam op JIT-compiles (see
# docs/phase-0.5-toolchain note / memory). VRAM-direct + FSDP don't need it.
_NEEDS_CUDA_HOME = {"zero_offload", "zero_infinity"}
_DEFAULT_CUDA_HOME = os.path.dirname(os.path.dirname(sys.executable))  # env prefix


def _ensure_cuda_home(config: RunConfig) -> None:
    if config.technique.name in _NEEDS_CUDA_HOME and not os.environ.get("CUDA_HOME"):
        candidate = _DEFAULT_CUDA_HOME
        if os.path.exists(os.path.join(candidate, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = candidate
            print(f"[benchmark] CUDA_HOME auto-set to {candidate} for "
                  f"{config.technique.name}")
        else:
            print(f"[benchmark] WARNING: {config.technique.name} needs CUDA_HOME "
                  f"but no nvcc found at {candidate}; offload op may fail to build")


# --- strategies -------------------------------------------------------------


def _strategy_baseline(config: RunConfig) -> dict:
    """On-GPU full-FT: bf16 weights + (paged 8-bit) AdamW + grad checkpointing.

    The Sprint 2 smoke test confirmed this fits Mistral-7B in ~27 GB VRAM at
    ~1.7 s/step (with levers.use_8bit_adam=True + gradient_checkpointing=True).
    """
    from .strategies.manual import run_baseline

    return run_baseline(config)


def _not_implemented(sprint: str) -> Callable[[RunConfig], dict]:
    def _stub(config: RunConfig) -> dict:
        raise NotImplementedError(
            f"technique '{config.technique.name}' is implemented in {sprint} "
            f"(see docs/phase-0.5-sprint-plan.md)"
        )

    return _stub


def _galore(config: RunConfig) -> dict:
    from .strategies.galore import run_galore
    return run_galore(config)


def _lomo(config: RunConfig) -> dict:
    from .strategies.lomo import run_lomo
    return run_lomo(config)


def _badam(config: RunConfig) -> dict:
    from .strategies.badam import run_badam
    return run_badam(config)


def _offload(config: RunConfig) -> dict:
    from .strategies.offload import run_offload
    return run_offload(config)


def _mezo(config: RunConfig) -> dict:
    from .strategies.mezo import run_mezo
    return run_mezo(config)


def _fsdp(config: RunConfig) -> dict:
    from .strategies.fsdp import run_fsdp
    return run_fsdp(config)


# Registry: technique name -> strategy fn. Sprints 2-5 replace the stubs.
STRATEGIES: dict[str, Callable[[RunConfig], dict]] = {
    "baseline": _strategy_baseline,
    "zero_offload": _offload,
    "fsdp_offload": _fsdp,
    "galore": _galore,
    "qgalore": _galore,
    "lomo": _lomo,
    "adalomo": _lomo,
    "badam": _badam,
    "mezo": _mezo,
    "zero_infinity": _not_implemented("Sprint 5"),
}


def benchmark(config: RunConfig) -> dict:
    """Run one technique under the fixed protocol; return its trade-off row.

    The returned summary is the trainer/strategy summary annotated with the
    technique name and a ``fits`` flag (peak VRAM <= 32 GB and peak RAM <= 96 GB).
    Strategy exceptions propagate to the caller, which decides whether to record
    ``fits=no`` and continue (the overnight runner does exactly that).
    """
    technique = config.technique.name
    if technique not in STRATEGIES:
        raise ValueError(f"unknown technique {technique!r}; "
                         f"known: {sorted(STRATEGIES)}")
    _ensure_cuda_home(config)
    print(f"[benchmark] technique={technique} model={config.base_model} "
          f"task={config.task}")
    summary = STRATEGIES[technique](config)
    summary.setdefault("technique", technique)
    # A strategy may set `fits` authoritatively (e.g. the offload preflight,
    # which compares against *available* RAM, not the nominal 96 GB). Only
    # derive it from the measured peaks when the strategy didn't decide.
    if "fits" not in summary:
        summary["fits"] = bool(
            summary.get("peak_vram_gb", 0) <= 32.0
            and summary.get("peak_ram_gb", 0) <= 96.0
        )
    return summary
