#!/usr/bin/env bash
set -euo pipefail

# Run DSA trace collection under a BurstGPT arrival trace (synthetic prompts).
#
# Required env:
# - CKPT_PATH
# - BURSTGPT_CSV: path to a BurstGPT CSV file (see scripts/datasets/download_burstgpt.sh)
#
# Optional env:
# - CONFIG (default: inference/config_671B_v3.2.json)
# - MP (default: 8)
# - LIMIT (default: 256)
# - BATCH_SIZE (default: 1)
# - KV_BLOCK_SIZE (default: 16)
# - MAX_NEW_TOKENS_CAP (default: 64)
# - TEMPERATURE (default: 0.6; run_burstgpt.py uses hardcoded 0.6, env var for docs only)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
BURSTGPT_CSV="${BURSTGPT_CSV:?Please set BURSTGPT_CSV}"
CONFIG="${CONFIG:-${ROOT_DIR}/inference/config_671B_v3.2.json}"
MP="${MP:-8}"
LIMIT="${LIMIT:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
MAX_NEW_TOKENS_CAP="${MAX_NEW_TOKENS_CAP:-64}"
TEMPERATURE="${TEMPERATURE:-0.6}"

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

OUT_DIR="${ROOT_DIR}/outputs/burstgpt_$(date +%s)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT="${ROOT_DIR}/inference/run_burstgpt.py"
ARGS=(
  --ckpt-path "${CKPT_PATH}"
  --config "${CONFIG}"
  --burstgpt-csv "${BURSTGPT_CSV}"
  --limit "${LIMIT}"
  --batch-size "${BATCH_SIZE}"
  --kv-block-size "${KV_BLOCK_SIZE}"
  --max-new-tokens-cap "${MAX_NEW_TOKENS_CAP}"
  --trace-out "${OUT_DIR}"
)

if [[ "${MP}" -gt 1 ]]; then
  torchrun --nproc-per-node "${MP}" "${SCRIPT}" "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}"
fi

echo "Done. Trace outputs: ${OUT_DIR}"

