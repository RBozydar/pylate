#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/ml_models/lightonai/ColBERT-Zero}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/tmp/pylate-hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/rbw/repo/pylate/output}"
SCRIPT_PATH="${SCRIPT_PATH:-examples/train/ColBERT-zero/reason_moderncolbert.py}"

VALIDATION_SIZE="${VALIDATION_SIZE:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-25}"
EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"
TEMPERATURE="${TEMPERATURE:-1.0}"
BEST_BATCH_SIZE="${BEST_BATCH_SIZE:-2048}"
MAX_STEPS="${MAX_STEPS:-100}"

cd /home/rbw/repo/pylate

run_sweep_entry() {
  local name="$1"
  local learning_rate="$2"
  local output_dir="${OUTPUT_ROOT}/${name}"

  echo "==> Starting ${name}"
  echo "    bs=${BEST_BATCH_SIZE} max_steps=${MAX_STEPS} lr=${learning_rate} temp=${TEMPERATURE}"
  echo "    output_dir=${output_dir}"

  uv run python "${SCRIPT_PATH}" \
    --model "${MODEL_PATH}" \
    --max-steps "${MAX_STEPS}" \
    --validation-size "${VALIDATION_SIZE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --bs "${BEST_BATCH_SIZE}" \
    --eval-bs "${BEST_BATCH_SIZE}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --lr "${learning_rate}" \
    --temp "${TEMPERATURE}" \
    --num-workers "${NUM_WORKERS}" \
    --logging-steps "${LOGGING_STEPS}" \
    --eval-steps "${EVAL_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --save-total-limit "${SAVE_TOTAL_LIMIT}" \
    --report-to "${REPORT_TO}" \
    --run-name "${name}" \
    --dataset-cache-dir "${DATASET_CACHE_DIR}" \
    --output-dir "${output_dir}"
}

run_sweep_entry "hq-lr-bs${BEST_BATCH_SIZE}-lr2e5-temp1" 2e-5
run_sweep_entry "hq-lr-bs${BEST_BATCH_SIZE}-lr5e5-temp1" 5e-5
run_sweep_entry "hq-lr-bs${BEST_BATCH_SIZE}-lr8e5-temp1" 8e-5
run_sweep_entry "hq-lr-bs${BEST_BATCH_SIZE}-lr1e4-temp1" 1e-4
