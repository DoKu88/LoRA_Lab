"""Model + optimizer construction for the three methods.

All three share one interface: ``build_model_and_tokenizer(config)`` returns a
ready-to-train ``(model, tokenizer)`` and ``build_optimizer(config, model)``
returns the matching optimizer. The only thing that varies is *how the base is
loaded and which params are trainable*:

  full_ft  bf16 base, every weight trainable, (8-bit Adam + grad checkpointing)
  lora     bf16 base frozen, small LoRA adapter trainable (PEFT)
  qlora    4-bit NF4 base frozen, LoRA adapter trainable (PEFT)
"""

from __future__ import annotations

import torch

from ..config import RunConfig

_DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _load_tokenizer(config: RunConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(config.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _lora_config(config: RunConfig):
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_model_and_tokenizer(config: RunConfig):
    from transformers import AutoModelForCausalLM

    tok = _load_tokenizer(config)
    method = config.method
    use_gc = method == "full_ft" and config.full_ft.gradient_checkpointing

    if method == "qlora":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=config.quant.load_in_4bit,
            bnb_4bit_quant_type=config.quant.quant_type,
            bnb_4bit_use_double_quant=config.quant.double_quant,
            bnb_4bit_compute_dtype=_DTYPE[config.quant.compute_dtype],
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model, quantization_config=bnb, device_map={"": 0},
            dtype=torch.bfloat16,
        )
    else:
        # Full FT keeps fp32 master weights (updated under bf16 autocast) —
        # pure-bf16 weight updates are too imprecise and diverge. LoRA's frozen
        # base can stay bf16 since it is never updated.
        load_dtype = torch.float32 if method == "full_ft" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(config.base_model, dtype=load_dtype)
        if torch.cuda.is_available():
            model = model.to("cuda")

    # Gradient checkpointing requires the KV cache off.
    if use_gc:
        model.config.use_cache = False

    if method in ("lora", "qlora"):
        from peft import get_peft_model, prepare_model_for_kbit_training

        if method == "qlora":
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=False
            )
        model = get_peft_model(model, _lora_config(config))
        model.print_trainable_parameters()
    else:  # full_ft — everything trainable
        for p in model.parameters():
            p.requires_grad_(True)
        if use_gc:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()

    return model, tok


def build_optimizer(config: RunConfig, model):
    """8-bit paged AdamW for full FT (memory lever); plain AdamW otherwise."""
    params = [p for p in model.parameters() if p.requires_grad]
    lr = config.hparams.lr
    wd = config.hparams.weight_decay

    if config.method == "full_ft" and config.full_ft.use_8bit_adam:
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
