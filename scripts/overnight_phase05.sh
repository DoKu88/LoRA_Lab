#!/bin/bash
# Phase 0.5 overnight orchestration: chains the remaining GPU work sequentially
# (single GPU, so they must serialize). Detached; survives the session.
#   1. wait for the in-flight task843 eval matrix to finish
#   2. task1344 (RTE entailment) eval matrix — the harder second task
#   3. method-combination matrix (combos)
# Each stage's runner is failure-tolerant (fits=no/no_row on OOM, partial-safe).
set -u
cd /home/shadow1/Projects/LoRA_Lab/.claude/worktrees/mistral7B
RUN="conda run -n lora_lab python -u scripts/run_phase05_matrix.py"
log() { echo "[overnight $(date +%H:%M:%S)] $*"; }

log "waiting for task843 eval matrix (MATRIX_EXIT in /tmp/phase05_eval_matrix.log)..."
until grep -q "MATRIX_EXIT" /tmp/phase05_eval_matrix.log 2>/dev/null; do sleep 30; done
log "task843 eval matrix done."

log "STAGE 2: task1344 (RTE entailment) — 6 working techniques"
$RUN --task task1344_glue_entailment_classification --eval-samples 248 \
     --only baseline_paged8bit galore qgalore lomo adalomo badam \
     --timeout-min 45 > /tmp/phase05_task1344.log 2>&1
log "task1344 matrix done (exit $?)."

log "STAGE 3: method-combination matrix"
$RUN --combos --timeout-min 45 > /tmp/phase05_combos.log 2>&1
log "combos matrix done (exit $?)."

touch /tmp/phase05_overnight_DONE
log "ALL STAGES COMPLETE."
