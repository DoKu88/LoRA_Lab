"""Config system for the run harness.

One YAML per run parameterizing ``{method × model × task × hparams}``.
Configs round-trip: ``load → run → reproduce``. Nested dataclasses keep the
method-specific knobs (LoRA rank, NF4 settings, full-FT memory levers)
grouped and self-documenting.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_METHODS = ("full_ft", "lora", "qlora")

# Phase 0.5 full-FT techniques (the trade-off-table rows). All are full-parameter
# fine-tunes of Mistral-7B; they differ in how optimizer/grad state is kept small
# enough to fit (on-GPU shrink vs. CPU offload). See docs/phase-0.5-sprint-plan.md.
VALID_TECHNIQUES = (
    "baseline",       # plain full-FT (fits only on small models; harness validation)
    "zero_offload",   # DeepSpeed ZeRO-2/3 + CPU optimizer offload (Sprint 2)
    "fsdp_offload",   # FSDP full-shard + CPU offload (Sprint 3)
    "galore",         # GaLore low-rank gradient projection (Sprint 4)
    "qgalore",        # Q-GaLore = GaLore + quantization (Sprint 4)
    "lomo",           # LOMO fused backward+update (Sprint 4)
    "adalomo",        # AdaLOMO (Sprint 4)
    "badam",          # BAdam block-coordinate (Sprint 5)
    "mezo",           # MeZO zeroth-order, forward-only (Sprint 5)
    "zero_infinity",  # ZeRO-Infinity NVMe offload — fallback only (Sprint 5)
)


@dataclass
class LoraParams:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    # Default targets the attention projections; overridable per model family.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class QuantParams:
    """4-bit NF4 settings (QLoRA only)."""

    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass
class FullFtParams:
    """Memory levers for the full-FT stress rung (sprint plan S4).

    ``gradient_checkpointing`` defaults OFF so the small-model comparison stays
    apples-to-apples with LoRA/QLoRA (which don't checkpoint); the matrix turns
    it ON for the larger full-FT rungs that need it to fit under 32 GB.
    """

    use_8bit_adam: bool = True
    gradient_checkpointing: bool = False


@dataclass
class TechniqueParams:
    """Phase 0.5 technique selector + per-technique knobs.

    ``name`` picks the training strategy (see VALID_TECHNIQUES). The rest are
    technique-specific and ignored by strategies that don't use them, so one
    schema covers every row of the trade-off table.
    """

    name: str = "baseline"
    # --- GaLore / Q-GaLore ---
    galore_rank: int = 128
    galore_update_proj_gap: int = 200
    galore_scale: float = 0.25
    galore_proj_type: str = "std"
    # --- BAdam ---
    badam_switch_every: int = 100  # optimizer steps per block before switching
    badam_switch_mode: str = "ascending"
    # --- LOMO / AdaLOMO ---
    lomo_clip_grad_norm: float = 0.0  # >0 enables LOMO's two-pass grad clipping
    # --- MeZO ---
    mezo_eps: float = 1e-3
    mezo_lr: float = 1e-6
    # --- offload (ZeRO / FSDP / NVMe) ---
    zero_stage: int = 2
    nvme_path: str | None = None  # ZeRO-Infinity only


@dataclass
class LeverParams:
    """Stackable memory optimizations, toggled individually for the Sprint 6
    ablation study. Each is measured in isolation (one flag changed vs. an
    anchor) so we know its standalone VRAM/RAM/speed contribution.

    Note: ``use_8bit_adam`` and ``gradient_checkpointing`` also live on
    FullFtParams (Phase 0 used them there); for Phase 0.5 the ablation toggles
    them here so the lever set is self-contained and one config diff = one flag.
    """

    use_8bit_adam: bool = False
    gradient_checkpointing: bool = False
    activation_offload: bool = False  # push activations to CPU
    drop_fp32_master: bool = False    # train without the fp32 weight shadow
    optimizer_offload: bool = True    # the offload itself (offload techniques)


@dataclass
class HParams:
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 1
    num_epochs: float = 1.0
    max_steps: int = -1  # -1 => use num_epochs
    max_train_samples: int = -1  # -1 => use the full train split
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    seed: int = 42


@dataclass
class LoggingParams:
    log_every: int = 10
    mem_trace_every: int = 1  # sample GPU memory every N optimizer steps
    wandb_project: str = "lora-lab-phase0"
    wandb_mode: str = "offline"  # online | offline | disabled
    wandb_entity: str | None = None


@dataclass
class EvalParams:
    metric: str = "rougeL"  # rougeL | exact_match
    max_eval_samples: int = 200
    gen_max_new_tokens: int = 128
    batch_size: int = 8


@dataclass
class RunConfig:
    """Everything needed to reproduce one run."""

    method: str = "lora"
    base_model: str = "HuggingFaceTB/SmolLM2-135M"
    task: str = "task setup-required"
    run_name: str | None = None  # default derived: {method}-{model}-{task}
    output_root: str = "results/runs"

    hparams: HParams = field(default_factory=HParams)
    lora: LoraParams = field(default_factory=LoraParams)
    quant: QuantParams = field(default_factory=QuantParams)
    full_ft: FullFtParams = field(default_factory=FullFtParams)
    # Phase 0.5 only: technique selects the full-FT strategy; levers are the
    # stackable optimizations the ablation study toggles. Both default to the
    # no-op baseline so Phase 0 configs are unaffected.
    technique: TechniqueParams = field(default_factory=TechniqueParams)
    levers: LeverParams = field(default_factory=LeverParams)
    logging: LoggingParams = field(default_factory=LoggingParams)
    eval: EvalParams = field(default_factory=EvalParams)

    # ---- derived / convenience ----------------------------------------
    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {self.method!r}"
            )
        if self.logging.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(
                f"wandb_mode must be online|offline|disabled, "
                f"got {self.logging.wandb_mode!r}"
            )
        if self.technique.name not in VALID_TECHNIQUES:
            raise ValueError(
                f"technique.name must be one of {VALID_TECHNIQUES}, "
                f"got {self.technique.name!r}"
            )

    @property
    def model_slug(self) -> str:
        return self.base_model.split("/")[-1]

    @property
    def name(self) -> str:
        return self.run_name or f"{self.method}-{self.model_slug}-{self.task}"

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.name

    # ---- (de)serialization --------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunConfig":
        return _from_dict(cls, d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI overrides ("3" -> 3, "true" -> True)."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(d: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides to a config dict in place.

    e.g. ``hparams.max_steps=10`` or ``logging.wandb_mode=disabled``.
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        parts = key.split(".")
        node = d
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
    return d


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict.

    Unknown keys raise, so a typo in a YAML field is caught at load time
    rather than silently ignored. ``from __future__ import annotations``
    turns field types into strings, so we resolve them via ``get_type_hints``.
    """
    if not is_dataclass(cls):
        return d
    import typing

    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(field_map)
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name in field_map:
        if name not in d:
            continue
        value = d[name]
        ftype = hints.get(name)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _from_dict(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
