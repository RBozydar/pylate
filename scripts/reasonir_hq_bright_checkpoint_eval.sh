#!/usr/bin/env bash

set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/rbw/repo/pylate/output}"
EVAL_SCRIPT="${EVAL_SCRIPT:-examples/evaluation/bright_reasonir.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/rbw/repo/pylate/output}"
BRIGHT_CACHE_DIR="${BRIGHT_CACHE_DIR:-/tmp/pylate-bright-cache}"
TASKS="${TASKS:-biology,economics,robotics,pony}"
REASONING="${REASONING:-none}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-16}"
QUERY_ENCODE_BATCH_SIZE="${QUERY_ENCODE_BATCH_SIZE:-32}"
DOCUMENT_BATCH_SIZE="${DOCUMENT_BATCH_SIZE:-128}"
CORPUS_CHUNK_SIZE="${CORPUS_CHUNK_SIZE:-1024}"
TOP_K="${TOP_K:-1000}"
BEST_BATCH_SIZE="${BEST_BATCH_SIZE:-1024}"
BEST_LR="${BEST_LR:-1e-5}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-ColBERT-Zero}"
WANDB_ENTITY="${WANDB_ENTITY:-rbw}"
WANDB_GROUP="${WANDB_GROUP:-reasonir-hq-bright-checkpoints}"
WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-bright-eval}"
WANDB_TAGS="${WANDB_TAGS:-bright,subset,reasonir-hq,checkpoint}"
INCLUDE_FINAL="${INCLUDE_FINAL:-1}"

cd /home/rbw/repo/pylate

default_runs=(
  "hq-batch-bs256-lr1e5-temp1"
  "hq-batch-bs512-lr2e5-temp1"
  "hq-batch-bs1024-lr4e5-temp1"
  "hq-batch-bs2048-lr8e5-temp1"
  "hq-lr-bs${BEST_BATCH_SIZE}-lr1e6-temp1"
  "hq-lr-bs${BEST_BATCH_SIZE}-lr5e6-temp1"
  "hq-lr-bs${BEST_BATCH_SIZE}-lr1e5-temp1"
  "hq-lr-bs${BEST_BATCH_SIZE}-lr5e5-temp1"
  "hq-temp-bs${BEST_BATCH_SIZE}-lr${BEST_LR//./}-temp002"
  "hq-temp-bs${BEST_BATCH_SIZE}-lr${BEST_LR//./}-temp005"
  "hq-temp-bs${BEST_BATCH_SIZE}-lr${BEST_LR//./}-temp01"
)

run_eval() {
  local run_name="$1"
  local model_path="$2"
  local checkpoint_name
  checkpoint_name="$(basename "${model_path}")"
  local output_json="${OUTPUT_ROOT}/${run_name}-${checkpoint_name}-bright-subset.json"

  if [[ ! -d "${model_path}" ]]; then
    echo "==> Skipping ${run_name}/${checkpoint_name}: model path not found at ${model_path}"
    return 0
  fi

  echo "==> Evaluating ${run_name}/${checkpoint_name}"
  echo "    model_path=${model_path}"
  echo "    output_json=${output_json}"

  uv run python "${EVAL_SCRIPT}" \
    --model_name_or_path "${model_path}" \
    --tasks "${TASKS}" \
    --reasoning "${REASONING}" \
    --query_batch_size "${QUERY_BATCH_SIZE}" \
    --query_encode_batch_size "${QUERY_ENCODE_BATCH_SIZE}" \
    --document_batch_size "${DOCUMENT_BATCH_SIZE}" \
    --corpus_chunk_size "${CORPUS_CHUNK_SIZE}" \
    --top_k "${TOP_K}" \
    --cache_dir "${BRIGHT_CACHE_DIR}" \
    --output_json "${output_json}" \
    --report-to "${REPORT_TO}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-run-name "eval-${run_name}-${checkpoint_name}-bright-subset" \
    --wandb-group "${WANDB_GROUP}" \
    --wandb-job-type "${WANDB_JOB_TYPE}" \
    --wandb-tags "${WANDB_TAGS}"
}

run_all_checkpoints_for_run() {
  local run_name="$1"
  local run_dir="${MODEL_ROOT}/${run_name}"

  if [[ ! -d "${run_dir}" ]]; then
    echo "==> Skipping ${run_name}: run directory not found at ${run_dir}"
    return 0
  fi

  local checkpoint_found=0
  while IFS= read -r model_path; do
    checkpoint_found=1
    run_eval "${run_name}" "${model_path}"
  done < <(find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

  if [[ "${checkpoint_found}" == "0" ]]; then
    echo "==> No checkpoint-* directories found for ${run_name}"
  fi

  if [[ "${INCLUDE_FINAL}" == "1" ]] && [[ -d "${run_dir}/final" ]]; then
    run_eval "${run_name}" "${run_dir}/final"
  fi
}

if [[ "$#" -gt 0 ]]; then
  for run_name in "$@"; do
    run_all_checkpoints_for_run "${run_name}"
  done
else
  for run_name in "${default_runs[@]}"; do
    run_all_checkpoints_for_run "${run_name}"
  done
fi
