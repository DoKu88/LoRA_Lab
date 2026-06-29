"""Standardized run logging — local-first, W&B best-effort.

Every run writes, under ``results/runs/{name}/``:
  * ``config.yaml``   — the full config snapshot (reproducibility)
  * ``metrics.jsonl`` — one JSON line per logged step (loss, tok/s, step
                        time, gpu_mem_gb, ...) — the per-step time series
  * ``summary.json``  — the final scalar summary (peak VRAM, params, etc.)

W&B is a thin, optional layer (sprint plan S3): if ``wandb`` is missing or
auth is unavailable, the run still executes and logs locally. Online mode
degrades to offline on any init failure rather than blocking the run.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, config, output_dir: str | Path | None = None):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._metrics_fh = self.metrics_path.open("w")
        self.summary: dict[str, Any] = {}

        # Snapshot the exact config used (round-trips back into RunConfig).
        config.save(self.output_dir / "config.yaml")

        self._wandb = None
        self._wandb_mode = config.logging.wandb_mode
        self._init_wandb()

    # ---- W&B (best-effort) --------------------------------------------
    def _wandb_run_name(self) -> str:
        """W&B display name: ``MM_DD_YYYY_HR_MM_SEC_<model>_<task>``.

        Timestamp-prefixed so re-runs of the same cell are distinct runs in the
        W&B UI (the local results/runs/ dir stays ``{method}-{model}-{task}``).
        Method is preserved as the W&B ``group`` rather than in the name.
        """
        ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        return f"{ts}_{self.config.model_slug}_{self.config.task}"

    def _init_wandb(self) -> None:
        if self._wandb_mode == "disabled":
            print("[wandb] disabled")
            return
        wb_name = self._wandb_run_name()
        try:
            import wandb

            # offline never needs creds; online falls back to offline on error.
            run = wandb.init(
                project=self.config.logging.wandb_project,
                entity=self.config.logging.wandb_entity,
                name=wb_name,
                group=self.config.method,
                job_type=self.config.task,
                mode=self._wandb_mode,
                dir=str(self.output_dir),
                config=self.config.to_dict(),
            )
            self._wandb = wandb
            print(f"[wandb] init OK (mode={self._wandb_mode}) -> {getattr(run, 'name', '?')}")
        except Exception as e:  # noqa: BLE001 - never let logging block a run
            print(f"[wandb] unavailable ({type(e).__name__}: {e}); continuing local-only")
            if self._wandb_mode == "online":
                # one retry in offline mode so the data is still captured locally
                try:
                    os.environ["WANDB_MODE"] = "offline"
                    import wandb

                    wandb.init(
                        project=self.config.logging.wandb_project,
                        name=wb_name,
                        mode="offline",
                        dir=str(self.output_dir),
                        config=self.config.to_dict(),
                    )
                    self._wandb = wandb
                    self._wandb_mode = "offline"
                    print("[wandb] fell back to offline")
                except Exception:  # noqa: BLE001
                    self._wandb = None

    # ---- logging API ---------------------------------------------------
    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one step's metrics locally and (best-effort) to W&B."""
        record = {"step": int(step), **metrics}
        self._metrics_fh.write(json.dumps(record) + "\n")
        self._metrics_fh.flush()
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=int(step))
            except Exception:  # noqa: BLE001
                pass

    def set_summary(self, **kwargs: Any) -> None:
        self.summary.update(kwargs)
        if self._wandb is not None:
            try:
                for k, v in kwargs.items():
                    self._wandb.run.summary[k] = v
            except Exception:  # noqa: BLE001
                pass

    def log_artifact_path(self, path: str | Path, name: str) -> None:
        """Record a produced artifact path in the summary (e.g. mem trace)."""
        self.summary.setdefault("artifacts", {})[name] = str(path)

    def finish(self) -> Path:
        """Flush summary.json and close W&B. Returns the summary path."""
        self._metrics_fh.close()
        summary_path = self.output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(self.summary, f, indent=2, default=str)
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:  # noqa: BLE001
                pass
        return summary_path

    # context-manager sugar
    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()
