"""Matrix presets: model ladder + per-(model, method) config construction.

Keeps the {method × model × task} sweep reproducible and apples-to-apples:
each model pins batch/seq/accumulation, and full-FT-specific memory levers
(8-bit Adam always; gradient checkpointing on the larger models) are applied
uniformly so the comparison stays controlled.

Models are identified everywhere by their HF id (e.g. ``google/gemma-2-2b-it``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .config import VALID_METHODS, RunConfig

# ---- model ladder ---------------------------------------------------------
# Models are identified by their HF id. Ungated (Phase 0) + gated (Sprint 7,
# needs HF_TOKEN). Both lists are ordered smallest -> up.
UNGATED_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
]
GATED_MODELS = [
    "google/gemma-2-2b-it",
    "meta-llama/Llama-3.2-1B-Instruct",
]
MODELS = UNGATED_MODELS + GATED_MODELS  # every valid model id

# execution order for smallest-first (by param count, not list order)
_ORDER = {
    m: i for i, m in enumerate([
        "HuggingFaceTB/SmolLM2-135M",       # 135M
        "Qwen/Qwen2.5-0.5B-Instruct",       # 494M
        "meta-llama/Llama-3.2-1B-Instruct",  # 1.24B
        "Qwen/Qwen2.5-1.5B-Instruct",       # 1.54B
        "google/gemma-2-2b-it",             # 2.61B
    ])
}

# ---- per-model training shape --------------------------------------------
# batch/accum/seq sized so full FT fits under 32 GB on the 5090.
_SHAPE = {
    "HuggingFaceTB/SmolLM2-135M": dict(batch_size=8, grad_accum=1, max_seq_len=384, full_ft_gc=False),
    "Qwen/Qwen2.5-0.5B-Instruct": dict(batch_size=4, grad_accum=2, max_seq_len=384, full_ft_gc=False),
    "Qwen/Qwen2.5-1.5B-Instruct": dict(batch_size=2, grad_accum=4, max_seq_len=384, full_ft_gc=True),
    "google/gemma-2-2b-it": dict(batch_size=1, grad_accum=8, max_seq_len=384, full_ft_gc=True),
    "meta-llama/Llama-3.2-1B-Instruct": dict(batch_size=2, grad_accum=4, max_seq_len=384, full_ft_gc=True),
}

# full FT needs a much lower LR than the adapters.
_LR = {"full_ft": 2e-5, "lora": 2e-4, "qlora": 2e-4}

# default sweep axes (used when a MatrixConfig leaves them unset)
DEFAULT_TASKS = [
    "task1564_triviaqa_answer_generation",
    "task843_financial_phrasebank_classification",
    "task512_twitter_emotion_classification",
    "task1344_glue_entailment_classification",
    "task639_multi_woz_user_utterance_generation",
]
DEFAULT_METHODS = ["qlora", "lora", "full_ft"]


def order_key(model_id: str) -> int:
    return _ORDER.get(model_id, 99)


def build_config(
    model_id: str,
    task: str,
    method: str,
    *,
    max_train_samples: int = -1,
    max_steps: int = -1,
    num_epochs: float = 1.0,
    max_eval_samples: int = 200,
    wandb_mode: str = "offline",
    wandb_project: str = "lora-lab-phase0",
    wandb_entity: str | None = None,
    lora_rank: int = 16,
) -> RunConfig:
    """Construct a fully-specified RunConfig for one matrix cell."""
    shape = _SHAPE[model_id]
    cfg = RunConfig(
        method=method,
        base_model=model_id,
        task=task,
    )
    cfg.hparams.lr = _LR[method]
    cfg.hparams.batch_size = shape["batch_size"]
    cfg.hparams.grad_accum = shape["grad_accum"]
    cfg.hparams.max_seq_len = shape["max_seq_len"]
    cfg.hparams.num_epochs = num_epochs
    cfg.hparams.max_steps = max_steps
    cfg.lora.r = lora_rank
    cfg.lora.alpha = lora_rank * 2
    cfg.full_ft.gradient_checkpointing = bool(shape["full_ft_gc"]) and method == "full_ft"
    cfg.hparams.max_train_samples = max_train_samples
    cfg.eval.max_eval_samples = max_eval_samples
    cfg.logging.wandb_mode = wandb_mode
    cfg.logging.wandb_project = wandb_project
    cfg.logging.wandb_entity = wandb_entity
    return cfg


# ---- matrix-level config (YAML-driven sweep) ------------------------------
@dataclass
class MatrixConfig:
    """The knobs for one ``run_matrix`` sweep — the CLI mirror, YAML-round-trippable.

    Either set ``models`` explicitly (HF ids from ``MODELS``) or leave it empty
    and pick a ``tier`` preset. The remaining fields are passed straight through
    to ``build_config`` per cell. Round-trips ``load -> run -> reproduce`` like
    ``RunConfig``; the resolved instance is saved next to the run logs.
    """

    models: list[str] = field(default_factory=list)  # explicit HF ids; [] => use tier
    tier: str = "ungated"  # ungated | gated | all
    tasks: list[str] = field(default_factory=lambda: list(DEFAULT_TASKS))
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    max_train_samples: int = -1
    max_steps: int = -1
    epochs: float = 1.0
    max_eval_samples: int = 200
    lora_rank: int = 16
    wandb_mode: str = "offline"  # online | offline | disabled
    wandb_project: str = "lora-lab-phase0"
    wandb_entity: str | None = None  # None => your W&B default entity
    skip_eval: bool = False
    no_aggregate: bool = False
    output_root: str = "results/runs"

    def validate(self) -> None:
        if self.tier not in ("ungated", "gated", "all"):
            raise ValueError(f"tier must be ungated|gated|all, got {self.tier!r}")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(
                f"wandb_mode must be online|offline|disabled, got {self.wandb_mode!r}"
            )
        bad_models = [m for m in self.models if m not in MODELS]
        if bad_models:
            raise ValueError(
                f"unknown models {bad_models}; valid HF ids: {MODELS}"
            )
        bad_methods = [m for m in self.methods if m not in VALID_METHODS]
        if bad_methods:
            raise ValueError(
                f"unknown methods {bad_methods}; valid: {list(VALID_METHODS)}"
            )
        if not self.tasks:
            raise ValueError("tasks must be non-empty")

    def resolved_models(self) -> list[str]:
        """Expand ``models``/``tier`` into the smallest-first HF-id list to run."""
        if self.models:
            ids = list(self.models)
        elif self.tier == "all":
            ids = UNGATED_MODELS + GATED_MODELS
        elif self.tier == "gated":
            ids = GATED_MODELS
        else:
            ids = UNGATED_MODELS
        return sorted(ids, key=order_key)

    # ---- (de)serialization (mirrors RunConfig) ------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MatrixConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown matrix-config keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MatrixConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)
