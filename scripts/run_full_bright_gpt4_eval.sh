#!/usr/bin/env bash

set -euo pipefail

cd /home/rbw/repo/pylate

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

WANDB_ROOT="${WANDB_ROOT:-/mnt/ml_models/wandb}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}/runs}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_ROOT}/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_ROOT}/artifacts}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${WANDB_ROOT}/eval-json}"

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
RUN_NAME="${RUN_NAME:-}"
REPORT_TO="${REPORT_TO:-wandb}"

CACHE_DIR="${CACHE_DIR:-/mnt/ml_models/cache/pylate-bright-cache}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
QUERY_ENCODE_BATCH_SIZE="${QUERY_ENCODE_BATCH_SIZE:-32}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-256}"
CORPUS_CHUNK_SIZE="${CORPUS_CHUNK_SIZE:-256}"
DOCUMENT_CACHE_CHUNK_SIZE="${DOCUMENT_CACHE_CHUNK_SIZE:-${CORPUS_CHUNK_SIZE}}"
CLEANUP_DOCUMENT_CACHE="${CLEANUP_DOCUMENT_CACHE:-1}"
TOP_K="${TOP_K:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-ColBERT-Zero}"
WANDB_ENTITY="${WANDB_ENTITY:-rbw}"
WANDB_GROUP="${WANDB_GROUP:-reasonir-mixed-gpt4-full}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-bright-eval}"
TASKS="${TASKS:-}"
TASK_CONFIGS="${TASK_CONFIGS:-}"

mkdir -p "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" "${EVAL_OUTPUT_ROOT}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  echo "Example:" >&2
  echo "  MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \\" >&2
  echo "  OUTPUT_JSON=/mnt/ml_models/wandb/eval-json/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \\" >&2
  echo "  RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \\" >&2
  echo "  ./scripts/run_full_bright_gpt4_eval.sh" >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="eval-$(basename "${MODEL_PATH}")-gpt4-full"
fi

if [[ -z "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="${EVAL_OUTPUT_ROOT}/${RUN_NAME}.json"
fi

slugify() {
  printf '%s' "$1" | tr ',/' '__' | tr -cd 'A-Za-z0-9._-'
}

run_eval() {
  local tasks="$1"
  local query_batch_size="$2"
  local query_encode_batch_size="$3"
  local document_batch_size="$4"
  local corpus_chunk_size="$5"
  local document_cache_chunk_size="$6"
  local run_name_suffix="${7:-}"

  local run_name="${RUN_NAME}"
  if [[ -n "${run_name_suffix}" ]]; then
    run_name="${RUN_NAME}-${run_name_suffix}"
  fi

  local args=(
    uv run python examples/evaluation/bright_reasonir.py
    --model_name_or_path "${MODEL_PATH}"
    --reasoning gpt4
    --use_reason_moderncolbert_gpt4_lengths
    --query_batch_size "${query_batch_size}"
    --query_encode_batch_size "${query_encode_batch_size}"
    --document_batch_size "${document_batch_size}"
    --corpus_chunk_size "${corpus_chunk_size}"
    --document_cache_chunk_size "${document_cache_chunk_size}"
    --top_k "${TOP_K}"
    --cache_dir "${CACHE_DIR}"
    --output_json "${OUTPUT_JSON}"
    --report-to "${REPORT_TO}"
  )

  if [[ "${CLEANUP_DOCUMENT_CACHE}" == "1" ]]; then
    args+=(--cleanup-document-cache)
  fi

  if [[ -n "${tasks}" ]]; then
    args+=(--tasks "${tasks}")
  fi

  if [[ "${REPORT_TO}" != "none" ]]; then
    args+=(
      --wandb-project "${WANDB_PROJECT}"
      --wandb-entity "${WANDB_ENTITY}"
      --wandb-run-name "${run_name}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-job-type "${WANDB_JOB_TYPE}"
    )
  fi

  "${args[@]}"
}

if [[ -z "${TASK_CONFIGS}" ]]; then
  run_eval \
    "${TASKS}" \
    "${QUERY_BATCH_SIZE}" \
    "${QUERY_ENCODE_BATCH_SIZE}" \
    "${DOCUMENT_BATCH_SIZE}" \
    "${CORPUS_CHUNK_SIZE}" \
    "${DOCUMENT_CACHE_CHUNK_SIZE}"
  exit 0
fi

IFS=';' read -r -a task_config_entries <<< "${TASK_CONFIGS}"

for config_entry in "${task_config_entries[@]}"; do
  if [[ -z "${config_entry// }" ]]; then
    continue
  fi

  IFS='|' read -r tasks query_batch query_encode_batch document_batch corpus_chunk document_cache_chunk <<< "${config_entry}"

  if [[ -z "${tasks:-}" || -z "${query_batch:-}" || -z "${query_encode_batch:-}" || -z "${document_batch:-}" || -z "${corpus_chunk:-}" ]]; then
    echo "Invalid TASK_CONFIGS entry: ${config_entry}" >&2
    echo "Expected format: tasks|query_batch|query_encode_batch|document_batch|corpus_chunk|document_cache_chunk" >&2
    exit 1
  fi

  if [[ -z "${document_cache_chunk:-}" ]]; then
    document_cache_chunk="${corpus_chunk}"
  fi

  run_eval \
    "${tasks}" \
    "${query_batch}" \
    "${query_encode_batch}" \
    "${document_batch}" \
    "${corpus_chunk}" \
    "${document_cache_chunk}" \
    "$(slugify "${tasks}")"
done
