#!/usr/bin/env bash

set -euo pipefail

cd /home/rbw/repo/pylate

export TMPDIR="${TMPDIR:-/home/rbw/.temp}"
export CACHE_DIR="${CACHE_DIR:-/home/rbw/.temp/pylate-bright-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/home/rbw/.temp/uv-cache}"
export WANDB_ROOT="${WANDB_ROOT:-/mnt/ml_models/wandb}"
export WANDB_DIR="${WANDB_DIR:-${WANDB_ROOT}/runs}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_ROOT}/data}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_ROOT}/cache}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_ROOT}/artifacts}"
export EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${WANDB_ROOT}/eval-json}"
export CLEANUP_DOCUMENT_CACHE="${CLEANUP_DOCUMENT_CACHE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TASK_CONFIGS="${TASK_CONFIGS:-earth_science|32|128|256|256|256;biology,robotics,stackoverflow,aops,theoremqa_questions|64|128|256|512|512;economics,psychology,sustainable_living,leetcode,pony,theoremqa_theorems|64|128|256|1024|1024}"

mkdir -p "${TMPDIR}" "${CACHE_DIR}" "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_ARTIFACT_DIR}" "${EVAL_OUTPUT_ROOT}" output/logs

if [[ ! -d "${CACHE_DIR}/xlangai___bright" && -d /mnt/ml_models/cache/pylate-bright-cache/xlangai___bright ]]; then
  rsync -a /mnt/ml_models/cache/pylate-bright-cache/xlangai___bright "${CACHE_DIR}/"
fi

for name in \
  reasonir-mixed-temp-bs2048-lr8e-5-temp01 \
  reasonir-mixed-temp-bs2048-lr8e-5-temp025 \
  reasonir-mixed-temp-bs2048-lr8e-5-temp05 \
  reasonir-mixed-temp-bs2048-lr8e-5-temp005; do
  json="${EVAL_OUTPUT_ROOT}/${name}-gpt4-full.json"
  if [[ -f "${json}" ]] && [[ "$(jq '.task_metrics | keys | length' "${json}")" == "12" ]]; then
    echo "$(date -Is) skip_complete=${name}"
    continue
  fi

  echo "$(date -Is) eval_start=${name}"
  MODEL_PATH="/home/rbw/repo/pylate/output/${name}/final" \
  OUTPUT_JSON="${json}" \
  RUN_NAME="eval-${name}-gpt4-full" \
  WANDB_GROUP="reasonir-mixed-temp-gpt4-full" \
  bash scripts/run_full_bright_gpt4_eval.sh
  status=$?
  echo "$(date -Is) eval_done=${name} status=${status}"
  if [[ "${status}" != "0" ]]; then
    exit "${status}"
  fi
done
