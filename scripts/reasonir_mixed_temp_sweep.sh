#!/usr/bin/env bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/ml_models/lightonai/ColBERT-Zero}"
DATA_ROOT="${DATA_ROOT:-/mnt/ml_models/datasets/ReasonIR/synthetic_data}"
TMPDIR="${TMPDIR:-${HOME}/.temp}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-${TMPDIR}/pylate-hf-cache}"
WANDB_ROOT="${WANDB_ROOT:-/mnt/ml_models/wandb}"
WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}/runs}"
WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_ROOT}/data}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_ROOT}/artifacts}"
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
LEARNING_RATE="${LEARNING_RATE:-8e-5}"
TEMP_VALUES="${TEMP_VALUES:-0.1,0.25,0.5,0.05}"
RUN_PREFIX="${RUN_PREFIX:-reasonir-mixed-temp-bs${BATCH_SIZE}-lr${LEARNING_RATE}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-5}"
EVAL_STEPS="${EVAL_STEPS:-${SAVE_STEPS}}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"
REPORT_TO="${REPORT_TO:-wandb}"

cd /home/rbw/repo/pylate
mkdir -p "${TMPDIR}" "${DATASET_CACHE_DIR}" "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}"
export TMPDIR
export WANDB_DIR WANDB_DATA_DIR WANDB_CACHE_DIR WANDB_ARTIFACT_DIR

format_temp_token() {
  python - "$1" <<'PY'
import math
import sys

value = float(sys.argv[1])
if math.isclose(value, round(value)):
    print(int(round(value)))
else:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    print(text.replace(".", ""))
PY
}

run_sweep_entry() {
  local name="$1"
  local temperature="$2"
  local output_dir="${OUTPUT_ROOT}/${name}"

  echo "==> Starting ${name}"
  echo "    dataset=${DATASETS}/${PROMPT_ID}/${GENERATOR}"
  echo "    epochs=${EPOCHS} bs=${BATCH_SIZE} lr=${LEARNING_RATE} temp=${temperature}"
  echo "    output_dir=${output_dir}"

  uv run python -u "${SCRIPT_PATH}" \
    --model "${MODEL_PATH}" \
    --data-root "${DATA_ROOT}" \
    --datasets "${DATASETS}" \
    --prompt-id "${PROMPT_ID}" \
    --generator "${GENERATOR}" \
    --validation-size "${VALIDATION_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --temp "${temperature}" \
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

IFS=',' read -r -a temp_values <<< "${TEMP_VALUES}"

for temperature in "${temp_values[@]}"; do
  if [[ -z "${temperature// }" ]]; then
    continue
  fi

  clean_temperature="${temperature// /}"
  temp_token="$(format_temp_token "${clean_temperature}")"
  run_sweep_entry \
    "${RUN_PREFIX}-temp${temp_token}" \
    "${clean_temperature}"
done
