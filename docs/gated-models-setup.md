# Sprint 7 — Gated-model follow-on (Gemma-2-2B / Llama-3.2)

Phase 0 runs on an **ungated** model ladder so the whole pipeline is
reproducible with no credentials. This sprint extends the *exact same harness*
to the **gated** bases once a HuggingFace token is available. Nothing in the
code changes — only `base_model` (already wired in `src/lora_lab/matrix.py`)
and a one-time token + license acceptance.

## 1. Get access (one-time, manual)

1. Create a token at <https://huggingface.co/settings/tokens> (read scope).
2. Accept each model's license on its HF page while logged in:
   - Gemma-2-2B-Instruct: <https://huggingface.co/google/gemma-2-2b-it>
   - Llama-3.2-1B-Instruct: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>
3. Authenticate the environment (run it yourself in the shell — interactive):

   ```bash
   # in the prompt, the leading ! runs it in this session:
   ! huggingface-cli login
   # or, non-interactively:
   ! export HF_TOKEN=hf_xxx   # then launch the run in the same shell
   ```

## 2. Run the gated tier

```bash
conda run -n lora_lab python scripts/run_matrix.py --tier gated \
    --max-train-samples 500 --epochs 3 --max-eval-samples 100
# or a single model key:
conda run -n lora_lab python scripts/run_matrix.py --models gemma2b \
    --tasks task843_financial_phrasebank_classification
# or the full ladder (ungated + gated) in one go:
conda run -n lora_lab python scripts/run_matrix.py --tier all \
    --max-train-samples 500 --epochs 3 --max-eval-samples 100
```

`--tier` accepts `ungated` (default — `tiny small mid`), `gated` (`gemma2b
llama1b`), or `all` (the full five-rung ladder). `--tier all` requires the
`HF_TOKEN` + accepted licenses below, since it includes the gated rungs.

This appends gated rows to `results/comparison.csv`/`.parquet`/`.md` and
renders the Gemma/Llama memory-vs-iteration plots alongside the ungated ones
(`scripts/build_table.py` re-aggregates everything under `results/runs/`).

## 3. Notes / expectations

- **Gemma-2-2B full FT is the real VRAM stress test.** Its tier preset
  (`gemma2b` in `matrix.py`) already enables 8-bit Adam + gradient
  checkpointing + batch size 1 / grad-accum 8 to stay under 32 GB. The peak
  comes from the persisted memory trace; the matrix runner warns if any cell
  exceeds the 32 GB ceiling.
- **LoRA target modules**: Gemma-2 and Llama-3.2 both use the
  `q/k/v/o_proj` names already in the default `lora.target_modules`, so no
  config change is needed. (Optionally add `gate_proj/up_proj/down_proj` to
  adapt the MLPs too.)
- Gemma-2 uses logit soft-capping and a larger MLP; if full FT is unstable,
  lower the LR (`--`-style override via a per-run YAML) before widening scope.
- Everything else (data pipeline, metrics, plots, table) is model-agnostic and
  already validated on the ungated ladder.
