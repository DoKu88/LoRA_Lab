"""Phase 2 hypernetwork run config (one YAML per run; round-trips load->run).

Mirrors the Phase-0 ``RunConfig`` pattern but for the meta-training run. Captures
everything needed to reproduce a hypernetwork run: the frozen base + target LoRA
shape (which must match the Phase-1 library for Phase-3 comparability), the
hypernetwork architecture (parameterization + conditioning dims), the objective
(reconstruction warmup vs SFT), and the data split. The committed YAMLs in
``configs/phase2/`` are the staged launch points — the user reviews the design
choice (esp. ``parameterization``) then launches with one command.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_OBJECTIVES = ("reconstruction", "sft")
VALID_PARAM = ("vera", "lowrank", "full")


@dataclass
class HyperConfig:
    # --- frozen base + target LoRA shape (match the Phase-1 library) ---------
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    load_in_4bit: bool = True          # QLoRA-style frozen base
    r: int = 16
    alpha: int = 32

    # --- hypernetwork architecture ------------------------------------------
    parameterization: str = "vera"     # vera | lowrank | full (the S2 ladder)
    encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    d_layer: int = 16
    d_module: int = 8
    trunk_hidden: int = 256

    # --- objective + optimization -------------------------------------------
    objective: str = "sft"             # reconstruction (warmup) | sft (the gate run)
    lr: float = 1e-4
    max_steps: int = 2000
    batch_size: int = 4
    grad_accum: int = 1
    max_seq_len: int = 512
    seed: int = 42
    gradient_checkpointing: bool = True   # base-backprop memory lever (SFT)
    warmup_from: str | None = None        # path to a recon-warmup hypernet checkpoint

    # --- data (Phase-1 artifacts) -------------------------------------------
    library_path: str = "configs/phase1/library.yaml"
    split_path: str = "configs/phase1/heldout_split.yaml"

    # --- runtime / logging ---------------------------------------------------
    device: str = "cuda"
    output_root: str = "results/phase2/runs"
    run_name: str | None = None
    wandb_project: str = "lora-lab-phase2"
    wandb_mode: str = "offline"

    def __post_init__(self) -> None:
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {VALID_OBJECTIVES}, got {self.objective!r}")
        if self.parameterization not in VALID_PARAM:
            raise ValueError(f"parameterization must be one of {VALID_PARAM}, got {self.parameterization!r}")
        if self.wandb_mode not in ("online", "offline", "disabled"):
            raise ValueError(f"wandb_mode invalid: {self.wandb_mode!r}")

    @property
    def name(self) -> str:
        slug = self.base_model.split("/")[-1]
        return self.run_name or f"{self.objective}-{self.parameterization}-{slug}"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HyperConfig":
        field_map = {f.name for f in fields(cls)}
        unknown = set(d) - field_map
        if unknown:
            raise ValueError(f"unknown HyperConfig keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HyperConfig":
        with Path(path).open() as f:
            return cls.from_dict(yaml.safe_load(f) or {})


assert is_dataclass(HyperConfig)
