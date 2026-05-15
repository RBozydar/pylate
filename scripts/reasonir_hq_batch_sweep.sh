#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/ml_models/lightonai/ColBERT-Zero}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/tmp/pylate-hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/rbw/repo/pylate/output}"
SCRIPT_PATH="${SCRIPT_PATH:-examples/train/ColBERT-zero/reason_moderncolbert.py}"

VALIDATION_SIZE="${VALIDATION_SIZE:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-$SAVE_STEPS}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"
TEMPERATURE="${TEMPERATURE:-1.0}"

cd /home/rbw/repo/pylate

run_sweep_entry() {
  local name="$1"
  local batch_size="$2"
  local max_steps="$3"
  local learning_rate="$4"
  local output_dir="${OUTPUT_ROOT}/${name}"

  echo "==> Starting ${name}"
  echo "    bs=${batch_size} max_steps=${max_steps} lr=${learning_rate} temp=${TEMPERATURE}"
  echo "    output_dir=${output_dir}"

  uv run python "${SCRIPT_PATH}" \
    --model "${MODEL_PATH}" \
    --max-steps "${max_steps}" \
    --validation-size "${VALIDATION_SIZE}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --bs "${batch_size}" \
    --eval-bs "${batch_size}" \
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

# Roughly equalize examples seen (~102k) while scaling LR with batch size.
# run_sweep_entry "hq-batch-bs256-lr1e5-temp1" 256 400 1e-5
# run_sweep_entry "hq-batch-bs512-lr2e5-temp1" 512 200 2e-5
# run_sweep_entry "hq-batch-bs1024-lr4e5-temp1" 1024 100 4e-5
# run_sweep_entry "hq-batch-bs2048-lr8e5-temp1" 2048 50 8e-5
run_sweep_entry "hq-batch-bs4096-lr1e4-temp1" 4096 25 1e-4
run_sweep_entry "hq-batch-bs8192-lr3.2e4-temp1" 8192 25 3.2e-4
