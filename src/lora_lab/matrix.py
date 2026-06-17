"""Matrix presets: model ladder + per-(tier, method) config construction.

Keeps the {method × model × task} sweep reproducible and apples-to-apples:
each rung pins batch/seq/accumulation, and full-FT-specific memory levers
(8-bit Adam always; gradient checkpointing on the larger rungs) are applied
uniformly so the comparison stays controlled.
"""

from __future__ import annotations

from .config import RunConfig

# ---- model ladder ---------------------------------------------------------
# Ungated (Phase 0) + gated (Sprint 7, needs HF_TOKEN). Keyed smallest -> up.
MODELS: dict[str, str] = {
    "tiny": "HuggingFaceTB/SmolLM2-135M",
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "mid": "Qwen/Qwen2.5-1.5B-Instruct",
    # --- gated (Sprint 7) ---
    "gemma2b": "google/gemma-2-2b-it",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
}
UNGATED_TIERS = ["tiny", "small", "mid"]
GATED_TIERS = ["gemma2b", "llama1b"]

# ladder ordering for smallest-first execution
_TIER_ORDER = {t: i for i, t in enumerate(["tiny", "small", "llama1b", "gemma2b", "mid"])}

# ---- per-tier training shape ---------------------------------------------
# batch/accum/seq sized so full FT fits under 32 GB on the 5090.
_TIER_SHAPE = {
    "tiny": dict(batch_size=8, grad_accum=1, max_seq_len=384, full_ft_gc=False),
    "small": dict(batch_size=4, grad_accum=2, max_seq_len=384, full_ft_gc=False),
    "mid": dict(batch_size=2, grad_accum=4, max_seq_len=384, full_ft_gc=True),
    "gemma2b": dict(batch_size=1, grad_accum=8, max_seq_len=384, full_ft_gc=True),
    "llama1b": dict(batch_size=2, grad_accum=4, max_seq_len=384, full_ft_gc=True),
}

# full FT needs a much lower LR than the adapters.
_LR = {"full_ft": 2e-5, "lora": 2e-4, "qlora": 2e-4}


def tier_of(model_key: str) -> str:
    return model_key


def order_key(model_key: str) -> int:
    return _TIER_ORDER.get(model_key, 99)


def build_config(
    model_key: str,
    task: str,
    method: str,
    *,
    max_train_samples: int = -1,
    max_steps: int = -1,
    num_epochs: float = 1.0,
    max_eval_samples: int = 200,
    wandb_mode: str = "offline",
    lora_rank: int = 16,
) -> RunConfig:
    """Construct a fully-specified RunConfig for one matrix cell."""
    shape = _TIER_SHAPE[model_key]
    cfg = RunConfig(
        method=method,
        base_model=MODELS[model_key],
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
    return cfg
