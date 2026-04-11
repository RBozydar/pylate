#!/usr/bin/env bash

set -euo pipefail

cd /home/rbw/repo/pylate

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODEL_PATH="${MODEL_PATH:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
RUN_NAME="${RUN_NAME:-}"
REPORT_TO="${REPORT_TO:-wandb}"

CACHE_DIR="${CACHE_DIR:-/mnt/ml_models/cache/pylate-bright-cache}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
QUERY_ENCODE_BATCH_SIZE="${QUERY_ENCODE_BATCH_SIZE:-32}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-256}"
CORPUS_CHUNK_SIZE="${CORPUS_CHUNK_SIZE:-256}"
TOP_K="${TOP_K:-1000}"
WANDB_PROJECT="${WANDB_PROJECT:-ColBERT-Zero}"
WANDB_ENTITY="${WANDB_ENTITY:-rbw}"
WANDB_GROUP="${WANDB_GROUP:-reasonir-mixed-gpt4-full}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-bright-eval}"
TASKS="${TASKS:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  echo "Example:" >&2
  echo "  MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \\" >&2
  echo "  OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \\" >&2
  echo "  RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \\" >&2
  echo "  ./scripts/run_full_bright_gpt4_eval.sh" >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="eval-$(basename "${MODEL_PATH}")-gpt4-full"
fi

if [[ -z "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="/home/rbw/repo/pylate/output/${RUN_NAME}.json"
fi

ARGS=(
  uv run python examples/evaluation/bright_reasonir.py
  --model_name_or_path "${MODEL_PATH}"
  --reasoning gpt4
  --use_reason_moderncolbert_gpt4_lengths
  --query_batch_size "${QUERY_BATCH_SIZE}"
  --query_encode_batch_size "${QUERY_ENCODE_BATCH_SIZE}"
  --document_batch_size "${DOCUMENT_BATCH_SIZE}"
  --corpus_chunk_size "${CORPUS_CHUNK_SIZE}"
  --top_k "${TOP_K}"
  --cache_dir "${CACHE_DIR}"
  --output_json "${OUTPUT_JSON}"
  --cleanup-document-cache
  --report-to "${REPORT_TO}"
)

if [[ -n "${TASKS}" ]]; then
  ARGS+=(--tasks "${TASKS}")
fi

if [[ "${REPORT_TO}" != "none" ]]; then
  ARGS+=(
    --wandb-project "${WANDB_PROJECT}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-run-name "${RUN_NAME}"
    --wandb-group "${WANDB_GROUP}"
    --wandb-job-type "${WANDB_JOB_TYPE}"
  )
fi

"${ARGS[@]}"
