#!/usr/bin/env bash
set -euo pipefail

# Run DSA trace collection on ShareGPT-style conversations.
#
# Required env:
# - CKPT_PATH
#
# Optional env:
# - CONFIG (default: inference/config_671B_v3.2.json)
# - MP (default: 1)
# - LIMIT (default: 64)
# - BATCH_SIZE (default: 1)
# - KV_BLOCK_SIZE (default: 16)
# - MAX_NEW_TOKENS (default: 64)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
CONFIG="${CONFIG:-${ROOT_DIR}/inference/config_671B_v3.2.json}"
MP="${MP:-1}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"

OUT_DIR="${ROOT_DIR}/outputs/sharegpt_$(date +%s)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SHAREGPT_JSON="${SHAREGPT_JSON:-}"
SHAREGPT_DATASET="${SHAREGPT_DATASET:-anon8231489123/ShareGPT_Vicuna_unfiltered}"
TURN_MODE="${TURN_MODE:-full}"

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/inference/run_dataset.py"
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
)

if [[ -n "${SHAREGPT_JSON}" ]]; then
  CMD+=(--sharegpt-json "${SHAREGPT_JSON}")
else
  CMD+=(--sharegpt-dataset "${SHAREGPT_DATASET}")
fi

if [[ "${MP}" -gt 1 ]]; then
  torchrun --nproc-per-node "${MP}" "${CMD[@]}"
else
  "${CMD[@]}"
fi

echo "Done. Trace outputs: ${OUT_DIR}"

