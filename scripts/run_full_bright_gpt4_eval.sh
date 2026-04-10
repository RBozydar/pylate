#!/usr/bin/env bash

set -euo pipefail

cd /home/rbw/repo/pylate

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

CACHE_DIR="${CACHE_DIR:-/mnt/ml_models/cache/pylate-bright-cache}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
QUERY_ENCODE_BATCH_SIZE="${QUERY_ENCODE_BATCH_SIZE:-32}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-256}"
CORPUS_CHUNK_SIZE="${CORPUS_CHUNK_SIZE:-256}"
TOP_K="${TOP_K:-1000}"
WANDB_GROUP="${WANDB_GROUP:-reasonir-hq-gpt4-gate-full}"

run_eval() {
  local model_path="$1"
  local output_json="$2"
  local run_name="$3"

  uv run python examples/evaluation/bright_reasonir.py \
    --model_name_or_path "${model_path}" \
    --reasoning gpt4 \
    --use_reason_moderncolbert_gpt4_lengths \
    --query_batch_size "${QUERY_BATCH_SIZE}" \
    --query_encode_batch_size "${QUERY_ENCODE_BATCH_SIZE}" \
    --document_batch_size "${DOCUMENT_BATCH_SIZE}" \
    --corpus_chunk_size "${CORPUS_CHUNK_SIZE}" \
    --top_k "${TOP_K}" \
    --cache_dir "${CACHE_DIR}" \
    --output_json "${output_json}" \
    --cleanup-document-cache \
    --report-to wandb \
    --wandb-project ColBERT-Zero \
    --wandb-entity rbw \
    --wandb-run-name "${run_name}" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-job-type bright-eval
}

run_eval \
  /mnt/ml_models/lightonai/ColBERT-Zero \
  /home/rbw/repo/pylate/output/bright-base-colbert-zero-gpt4-full.json \
  eval-colbert-zero-gpt4-full-r4

run_eval \
  /home/rbw/repo/pylate/output/hq-batch-bs2048-lr8e5-temp1/checkpoint-50 \
  /home/rbw/repo/pylate/output/hq-batch-bs2048-lr8e5-temp1-checkpoint-50-gpt4-full.json \
  eval-hq-batch-bs2048-lr8e5-temp1-checkpoint-50-gpt4-full

run_eval \
  /mnt/ml_models/lightonai/ColBERT-Zero-reason-training/hq-batch-bs4096-lr1e4-temp1/checkpoint-5 \
  /home/rbw/repo/pylate/output/hq-batch-bs4096-lr1e4-temp1-checkpoint-5-gpt4-full.json \
  eval-hq-batch-bs4096-lr1e4-temp1-checkpoint-5-gpt4-full
