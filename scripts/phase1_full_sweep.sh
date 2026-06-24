#!/usr/bin/env bash
# Phase 1 — full library sweep over all full-coverage tasks (~1116).
#
# Resumable end-to-end:
#   1. descriptions  — fill the SNI Definition for every task (downloads datasets)
#   2. gate          — adapter-vs-base quality gate on the resident 7B (the GPU-heavy part)
#   3. report        — aggregate gate_results.jsonl into library_quality.{csv,parquet,md}
#   4. finalize      — fold gate outcomes (kind/metric/rank + below-tau quarantines)
#                      into the manifest; recompute the version hash
#
# Per-task isolation means one bad adapter/dataset never aborts the batch; both
# `descriptions` and `gate` skip work already recorded, so re-running resumes.
# The locked held-out split is NOT touched here (expanding it to T2L's 479/11/10
# is a separate, reviewed step after the full gate lands).
set -uo pipefail
cd "$(dirname "$0")/.."

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lora_lab
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG_DIR=results/phase1
mkdir -p "$LOG_DIR"
echo "[sweep] start $(date -u +%FT%TZ)"

echo "[sweep] step 1/4 — descriptions (full)"
python -u scripts/phase1_library.py descriptions 2>&1 | tee "$LOG_DIR/descriptions_full.log"

echo "[sweep] step 2/4 — gate (full)"
python -u scripts/phase1_library.py gate --eval-samples 120 2>&1 | tee "$LOG_DIR/gate_full.log"

echo "[sweep] step 3/4 — report"
python -u scripts/phase1_finalize.py report 2>&1 | tee "$LOG_DIR/report_full.log"

echo "[sweep] step 4/4 — finalize manifest"
python -u scripts/phase1_finalize.py finalize 2>&1 | tee "$LOG_DIR/finalize_full.log"

echo "[sweep] done $(date -u +%FT%TZ)"
