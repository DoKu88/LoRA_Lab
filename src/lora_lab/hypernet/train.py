"""Train the Text-to-LoRA hypernetwork — the classic training loop for both
objectives (reconstruction and generalization).

The loop is the textbook five steps; the only per-objective difference (compare
generated weights vs. run the frozen base) is wrapped in ``loss_fn`` and chosen
once before the loop:

    for step in range(n_steps):
        descriptions, targets = train_data.batch(...)   # 1. load samples
        loss = loss_fn(descriptions, targets)           # 2. predict + 3. compare
        loss.backward(); optimizer.step()               # 4. backprop + update
        if time_to_validate: validate(...)              # 5. validate on val split
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ..utils.run_logger import RunLogger
from ..utils.vram import cuda_mem_snapshot, reset_peak_memory
from .apply import LoRARegistry, inject, remove, target_specs
from .data import GeneralizationSampler, LibraryReconSampler, SyntheticReconSampler
from .model import HyperLoRAGenerator, MeanPoolEncoder, delta_w


def build_model(cfg):
    """Load (base, tokenizer, encoder, generator, specs); warm-start if configured."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.load_in_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=bnb,
                                                    device_map={"": 0})
    else:
        dtype = torch.float32 if cfg.device == "cpu" else torch.bfloat16
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=dtype).to(cfg.device)
    base.eval()
    for param in base.parameters():
        param.requires_grad_(False)
    if cfg.gradient_checkpointing and cfg.objective == "generalization":
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    encoder = MeanPoolEncoder(cfg.encoder_model, device=cfg.device)
    specs = target_specs(base, cfg.target_modules)
    generator = HyperLoRAGenerator(specs, task_dim=encoder.dim, rank=cfg.rank, alpha=cfg.alpha,
                                   parameterization=cfg.parameterization, layer_dim=cfg.layer_dim,
                                   module_dim=cfg.module_dim, trunk_hidden=cfg.trunk_hidden).to(cfg.device)
    if cfg.warmup_from and Path(cfg.warmup_from).exists():
        generator.load_state_dict(torch.load(cfg.warmup_from, map_location=cfg.device))
        print(f"[train] warm-started from {cfg.warmup_from}")
    return base, tokenizer, encoder, generator, specs


def reconstruction_error(adapter, target, *, scaling, eps=1e-6):
    """Mean relative Frobenius error ‖ΔW_gen − ΔW_tgt‖/‖ΔW_tgt‖ over shared keys.

    Relative (not absolute) so a target LoRA's tiny per-element values don't
    vanish the gradient; 1.0 when the generated ΔW is 0 (the no-learning floor).
    """
    keys = [key for key in adapter if key in target]
    if not keys:
        raise ValueError("no shared target keys between generated and target adapters")
    total = adapter[keys[0]][0].new_zeros(())
    for key in keys:
        delta_gen = delta_w(*adapter[key], scaling)
        delta_tgt = delta_w(*target[key], scaling)
        total = total + (delta_gen - delta_tgt).norm() / (delta_tgt.norm() + eps)
    return total / len(keys)


def make_loss_fn(cfg, generator, base, encoder, registry, *, device, scaling):
    """Return ``loss_fn(descriptions, targets)`` for the configured objective.

    reconstruction: generate a LoRA per description, compare ΔW to the target
                    library LoRA (no base forward) — mean over the batch.
    generalization: generate one LoRA, apply it to the frozen base, run the task
                    batch, and take the cross-entropy on the masked labels
                    (supervised fine-tuning through the frozen base).
    """
    if cfg.objective == "reconstruction":
        def loss_fn(descriptions, targets):
            embeddings = encoder.encode(descriptions).to(device)
            total = embeddings.new_zeros(())
            for index, target in enumerate(targets):
                adapter = generator(embeddings[index])
                target = {key: (a.to(device), b.to(device)) for key, (a, b) in target.items()}
                total = total + reconstruction_error(adapter, target, scaling=scaling)
            return total / len(targets)
    else:  # generalization
        def loss_fn(descriptions, batch):
            embedding = encoder.encode(descriptions).to(device).squeeze(0)
            registry.set_adapter(generator(embedding))          # apply the generated LoRA
            batch = {key: value.to(device) for key, value in batch.items()}
            return base(**batch).loss                            # CE on the masked labels
    return loss_fn


@torch.no_grad()
def validate(loss_fn, generator, val_data, *, batch_size, n_batches) -> float:
    """Mean loss over ``n_batches`` of the validation split (no gradients)."""
    generator.eval()
    total = 0.0
    for _ in range(n_batches):
        descriptions, targets = val_data.batch(batch_size)
        total += float(loss_fn(descriptions, targets))
    generator.train()
    return total / n_batches


def _load_samples(cfg, tokenizer, specs, *, split, synthetic):
    if synthetic:
        return SyntheticReconSampler(specs, rank=cfg.rank, seed=cfg.seed)
    if cfg.objective == "reconstruction":
        return LibraryReconSampler(cfg.split_path, cfg.library_path, split=split, seed=cfg.seed)
    return GeneralizationSampler(cfg.split_path, cfg.library_path, tokenizer,
                                 split=split, max_seq_len=cfg.max_seq_len, seed=cfg.seed)


def _build_logger(cfg, stage):
    """RunLogger over a HyperConfig (local metrics.jsonl + best-effort W&B)."""
    view = SimpleNamespace(
        output_dir=Path(cfg.output_root) / cfg.name,
        logging=SimpleNamespace(wandb_mode=cfg.wandb_mode, wandb_project=cfg.wandb_project,
                                wandb_entity=getattr(cfg, "wandb_entity", None)),
        model_slug=cfg.base_model.split("/")[-1],
        task=cfg.objective,
        method=stage,
        to_dict=cfg.to_dict,
        save=cfg.save,
    )
    return RunLogger(view)


def train(cfg, *, steps=None, synthetic=False, stage=None):
    """Train the hypernetwork for ``cfg.objective`` and save the checkpoint."""
    base, tokenizer, encoder, generator, specs = build_model(cfg)
    device, scaling = cfg.device, generator.scaling

    train_data = _load_samples(cfg, tokenizer, specs, split="train", synthetic=synthetic)
    val_data = None if synthetic else _load_samples(cfg, tokenizer, specs, split="val", synthetic=False)
    if val_data is not None and len(val_data) == 0:
        val_data = None

    optimizer = torch.optim.Adam(generator.parameters(), lr=cfg.lr)
    logger = _build_logger(cfg, stage or cfg.objective)

    # Generalization applies the generated LoRA to the frozen base via these hooks;
    # the reconstruction objective compares weights directly and needs no injection.
    registry = LoRARegistry()
    handles = inject(base, cfg.target_modules, registry, scaling=scaling) if cfg.objective == "generalization" else []
    loss_fn = make_loss_fn(cfg, generator, base, encoder, registry, device=device, scaling=scaling)

    n_steps = steps or cfg.max_steps
    if device == "cuda":
        reset_peak_memory()
    print(f"[train] {cfg.name}: {n_steps} steps, objective={cfg.objective}, "
          f"params={generator.num_params():,}, targets={len(specs)}")

    generator.train()
    losses: list[float] = []
    bar = tqdm(range(n_steps), desc=f"train [{cfg.objective}]", unit="step")
    try:
        for step in bar:
            # 1. load a batch of (description, ground-truth target) samples
            descriptions, targets = train_data.batch(cfg.batch_size)

            # 2. predict the output LoRA + 3. compare to the ground truth -> loss
            loss = loss_fn(descriptions, targets)

            # 4. backprop and update the hypernetwork (the base stays frozen)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))

            # 5. periodically validate on the held-out val split
            metrics = {"train/loss": round(losses[-1], 6), "grad_norm": round(float(grad_norm), 4)}
            if val_data is not None and (step + 1) % cfg.val_every == 0:
                metrics["val/loss"] = round(
                    validate(loss_fn, generator, val_data,
                             batch_size=cfg.batch_size, n_batches=cfg.val_batches), 6)
            if device == "cuda":
                metrics["gpu_mem_gb"] = round(cuda_mem_snapshot()["allocated_gb"], 4)
            logger.log_metrics(step, metrics)
            bar.set_postfix(loss=f"{losses[-1]:.5f}")
    finally:
        bar.close()
        remove(base, handles)

    return _save_and_summarize(cfg, generator, specs, logger, losses, device)


def _save_and_summarize(cfg, generator, specs, logger, losses, device):
    out_dir = logger.output_dir
    torch.save(generator.state_dict(), out_dir / "hypernet.pt")
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0
    logger.set_summary(name=cfg.name, objective=cfg.objective, parameterization=cfg.parameterization,
                       steps=len(losses), hypernet_params=generator.num_params(), n_targets=len(specs),
                       loss_first=round(losses[0], 6), loss_last=round(losses[-1], 6),
                       peak_vram_gb=round(peak_vram_gb, 4))
    logger.log_artifact_path(out_dir / "hypernet.pt", "hypernet_checkpoint")
    logger.finish()
    print(f"[train] done: loss {losses[0]:.4f} -> {losses[-1]:.4f}; saved {out_dir}/hypernet.pt")
    return dict(logger.summary)
