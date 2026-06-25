"""Sprint 5 — train our *own* oracle LoRAs for the held-out tasks.

These are the adapters Phase 2 evaluates generated LoRAs *against* and Phase 3
compares feature geometry *with*, so we train them ourselves (fully under our
control, reproducible) rather than reuse the downloaded Lots-of-LoRAs adapter.
Recipe matches the library: rank-16 LoRA on q/k/v_proj of Mistral-7B.

Reuses the Phase-0 stack: ``train(RunConfig)`` + ``evaluate_checkpoint``. Since
the trainer reads tasks from ``configs/tasks.yaml``, we extend that manifest
(don't fork it) with the held-out tasks before training.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..config import RunConfig
from ..eval.evaluate import evaluate_checkpoint
from ..train.trainer import train

TASKS_MANIFEST = "configs/tasks.yaml"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
ORACLE_ROOT = "results/phase1/oracles"


def ensure_tasks_registered(tasks: list[dict], manifest: str = TASKS_MANIFEST) -> None:
    """Idempotently add held-out tasks to configs/tasks.yaml.

    ``tasks``: list of {name, hf_repo, kind, metric, description}.
    """
    path = Path(manifest)
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    raw = raw or {}
    existing = raw.setdefault("tasks", {})
    for t in tasks:
        if t["name"] in existing:
            continue
        existing[t["name"]] = {
            "hf_repo": t["hf_repo"],
            "kind": t["kind"],
            "metric": t["metric"],
            "description": t["description"],
        }
    with path.open("w") as f:
        yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)


def train_oracle(
    task_name: str,
    metric: str,
    *,
    max_train_samples: int = 500,
    max_steps: int = 250,
    lr: float = 2e-4,
    seed: int = 42,
) -> dict:
    """Train one rank-16 oracle LoRA and eval it on the held-out test split."""
    cfg = RunConfig(
        method="lora",
        base_model=BASE_MODEL,
        task=task_name,
        run_name=f"oracle-{task_name}",
        output_root=ORACLE_ROOT,
    )
    cfg.lora.r = 16
    cfg.lora.alpha = 32
    cfg.lora.target_modules = ["q_proj", "k_proj", "v_proj"]
    cfg.hparams.lr = lr
    cfg.hparams.batch_size = 4
    cfg.hparams.grad_accum = 1
    cfg.hparams.max_steps = max_steps
    cfg.hparams.max_train_samples = max_train_samples
    cfg.hparams.seed = seed
    cfg.eval.metric = metric
    cfg.eval.max_eval_samples = 120
    cfg.logging.wandb_mode = "offline"
    cfg.logging.wandb_project = "lora-lab-phase1"

    summary = train(cfg)
    ckpt = summary["checkpoint_dir"]
    scored = evaluate_checkpoint(ckpt, BASE_MODEL, task_name, metric,
                                 max_eval_samples=120,
                                 max_new_tokens=16 if metric == "exact_match" else 64)
    return {
        "task_name": task_name,
        "metric": metric,
        "oracle_score": round(scored["score"], 4),
        "checkpoint": ckpt,
        "checkpoint_mb": summary.get("checkpoint_size_mb"),
        "peak_vram_gb": summary.get("peak_vram_gb"),
        "final_train_loss": summary.get("final_train_loss"),
        "steps": summary.get("steps"),
    }
