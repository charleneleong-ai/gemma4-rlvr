#!/usr/bin/env bash
# Re-run the remaining top-5 scorers (#5 beta=0.02, #13 lr=2e-6) so they get
# per-experiment LoRA snapshots under gemma_4_lora/exp_<N>/ and metrics.heldout
# + metrics.regression in results.jsonl. The earlier reruns covered #4/#10/#15
# (now #19/#20/#21).
set -euo pipefail
cd /workspace/gemma4_rl

LOG=logs/rerun_top5_remaining_$(date -u +%Y%m%dT%H%M%SZ).log
echo "[$(date -uIs)] starting remaining top-5 reruns; tail -F $LOG" | tee -a "$LOG"

run() {
  local label="$1"; shift
  echo "[$(date -uIs)] === $label === $*" | tee -a "$LOG"
  ./.venv/bin/python -u train.py train -c train_fast -d "$label" \
    --max-steps 80 --patience 8 --plateau-window 10 --plateau-delta 0.05 \
    "$@" >> "$LOG" 2>&1
  echo "[$(date -uIs)] === done: $label ===" | tee -a "$LOG"
}

run "[rerun EARLY_KILL→exp#5] lower KL penalty (beta=0.02)"  --beta 0.02
run "[rerun EARLY_KILL→exp#13] smaller LR (lr=2e-6)"          --learning-rate 2.0e-6

echo "[$(date -uIs)] all remaining reruns finished" | tee -a "$LOG"
