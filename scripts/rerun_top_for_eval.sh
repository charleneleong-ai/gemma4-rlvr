#!/usr/bin/env bash
# Re-run the 3 clean-exit top scorers so we (a) capture per-experiment LoRA
# snapshots under gemma_4_lora/exp_<N>/ and (b) populate metrics.heldout +
# metrics.regression in results.jsonl. Train.py runs without the autoresearch
# triage wrapper so the eval block always reaches.
set -euo pipefail
cd /workspace/gemma4_rl

LOG=logs/rerun_top_$(date -u +%Y%m%dT%H%M%SZ).log
echo "[$(date -uIs)] starting top-3 reruns; tail -F $LOG" | tee -a "$LOG"

run() {
  local label="$1"; shift
  echo "[$(date -uIs)] === $label === $*" | tee -a "$LOG"
  ./.venv/bin/python -u train.py train -c train_fast -d "$label" \
    --max-steps 80 --patience 8 --plateau-window 10 --plateau-delta 0.05 \
    "$@" >> "$LOG" 2>&1
  echo "[$(date -uIs)] === done: $label ===" | tee -a "$LOG"
}

run "[rerun BASELINE→exp#4] 0.5x learning rate"             --learning-rate 5.0e-6
run "[rerun KEEP→exp#10] num_generations=8 (default lr)"     --num-generations 8
run "[rerun DISCARD→exp#15] lr=5e-6 + num_generations=8"     --num-generations 8 --learning-rate 5.0e-6

echo "[$(date -uIs)] all reruns finished" | tee -a "$LOG"
