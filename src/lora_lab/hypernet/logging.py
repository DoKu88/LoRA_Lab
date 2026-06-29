"""Build a ``RunLogger`` for a ``HyperConfig`` run.

Logs every run local-first (``metrics.jsonl``) with best-effort W&B. ``RunLogger``
reads a ``RunConfig``-shaped object; this thin shim exposes the same surface from
a ``HyperConfig``. The W&B *group* is the run stage, *job_type* the objective.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ..train.run_logger import RunLogger


class _LoggerCfg:
    """Minimal RunConfig-ish view of a HyperConfig for RunLogger."""

    def __init__(self, cfg, stage: str):
        self._cfg = cfg
        self.output_dir = Path(cfg.output_root) / cfg.name
        self.logging = SimpleNamespace(
            wandb_mode=cfg.wandb_mode,
            wandb_project=cfg.wandb_project,
            wandb_entity=getattr(cfg, "wandb_entity", None),
        )
        self.model_slug = cfg.base_model.split("/")[-1]
        self.task = cfg.objective       # W&B job_type
        self.method = stage             # W&B group (run stage)

    def to_dict(self):
        return self._cfg.to_dict()

    def save(self, path):
        return self._cfg.save(path)


def build_run_logger(cfg, *, stage: str = "train") -> RunLogger:
    """Construct a RunLogger for a HyperConfig run."""
    return RunLogger(_LoggerCfg(cfg, stage))
