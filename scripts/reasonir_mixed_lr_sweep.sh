#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/ml_models/lightonai/ColBERT-Zero}"
DATA_ROOT="${DATA_ROOT:-/mnt/ml_models/datasets/ReasonIR/synthetic_data}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/tmp/pylate-hf-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/rbw/repo/pylate/output}"
SCRIPT_PATH="${SCRIPT_PATH:-examples/train/ColBERT-zero/reasonir.py}"

DATASETS="${DATASETS:-mixed}"
PROMPT_ID="${PROMPT_ID:-hq_gen}"
GENERATOR="${GENERATOR:-balanced-v1}"
VALIDATION_SIZE="${VALIDATION_SIZE:-0.01}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${BATCH_SIZE}}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
LR_VALUES="${LR_VALUES:-5e-6,1e-5,3e-5,5e-5}"
RUN_PREFIX="${RUN_PREFIX:-reasonir-mixed-lr-bs${BATCH_SIZE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"

cd /home/rbw/repo/pylate

run_sweep_entry() {
  local name="$1"
  local learning_rate="$2"
  local output_dir="${OUTPUT_ROOT}/${name}"

  echo "==> Starting ${name}"
  echo "    dataset=${DATASETS}/${PROMPT_ID}/${GENERATOR}"
  echo "    epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${learning_rate}"
  echo "    output_dir=${output_dir}"

  uv run python -u "${SCRIPT_PATH}" \
    --model "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --prompt-id "${PROMPT_ID}" \
    --generator "${GENERATOR}" \
    --validation-size "${VALIDATION_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${learning_rate}" \
    --bs "${BATCH_SIZE}" \
    --eval-bs "${EVAL_BATCH_SIZE}" \
    --mini-batch-size "${MINI_BATCH_SIZE}" \
    --fp16 \
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

IFS=',' read -r -a lr_values <<< "${LR_VALUES}"

for learning_rate in "${lr_values[@]}"; do
  if [[ -z "${learning_rate// }" ]]; then
    continue
  fi

  clean_learning_rate="${learning_rate// /}"
  run_sweep_entry \
    "${RUN_PREFIX}-lr${clean_learning_rate//./p}" \
    "${clean_learning_rate}"
done
