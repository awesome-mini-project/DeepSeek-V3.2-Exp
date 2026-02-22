#!/usr/bin/env bash
set -euo pipefail

# Run DSA trace collection on ShareGPT-style conversations.
#
# Required env:
# - CKPT_PATH
#
# Optional env:
# - CONFIG (default: inference/config_671B_v3.2.json)
# - MP (default: 8)
# - LIMIT (default: 64)
# - BATCH_SIZE (default: 1)
# - KV_BLOCK_SIZE (default: 16)
# - MAX_NEW_TOKENS (default: 64)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
CONFIG="${CONFIG:-${ROOT_DIR}/inference/config_671B_v3.2.json}"
MP="${MP:-8}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

# Force HuggingFace caches to live under the repo (avoid ~/.cache/huggingface).
DATA_ROOT="${DATAS_DIR:-${DATA_ROOT:-${ROOT_DIR}/data}}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# Make CONFIG robust when user provides a relative path.
if [[ "${CONFIG}" != /* ]]; then
  if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then
    CONFIG="${ROOT_DIR}/${CONFIG}"
  elif [[ -f "${ROOT_DIR}/inference/${CONFIG}" ]]; then
    CONFIG="${ROOT_DIR}/inference/${CONFIG}"
  fi
fi

OUT_DIR="${ROOT_DIR}/outputs/sharegpt_$(date +%s)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHAREGPT_JSON="${SHAREGPT_JSON:-}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-anon8231489123/ShareGPT_Vicuna_unfiltered}"
SHAREGPT_HF_FILE="${SHAREGPT_HF_FILE:-ShareGPT_V3_unfiltered_cleaned_split.json}"
TURN_MODE="${TURN_MODE:-full}"

SCRIPT="${ROOT_DIR}/inference/run_dataset.py"
ARGS=(
  --ckpt-path "${CKPT_PATH}"
  --config "${CONFIG}"
  --dataset sharegpt
  --split train
  --limit "${LIMIT}"
  --batch-size "${BATCH_SIZE}"
  --kv-block-size "${KV_BLOCK_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --trace-out "${OUT_DIR}"
  --sharegpt-turn-mode "${TURN_MODE}"
  --sharegpt-hf-file "${SHAREGPT_HF_FILE}"
)

if [[ -n "${SHAREGPT_JSON}" ]]; then
  ARGS+=(--sharegpt-json "${SHAREGPT_JSON}")
else
  ARGS+=(--sharegpt-dataset "${SHAREGPT_DATASET}")
fi

if [[ "${MP}" -gt 1 ]]; then
  torchrun --nproc-per-node "${MP}" "${SCRIPT}" "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}"
fi

echo "Done. Trace outputs: ${OUT_DIR}"

