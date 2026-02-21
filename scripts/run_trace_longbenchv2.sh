#!/usr/bin/env bash
set -euo pipefail

# Run DSA trace collection on LongBench v2.
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
# - MAX_NEW_TOKENS (default: 32)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
CONFIG="${CONFIG:-${ROOT_DIR}/inference/config_671B_v3.2.json}"
MP="${MP:-8}"
LIMIT="${LIMIT:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

OUT_DIR="${ROOT_DIR}/outputs/longbenchv2_$(date +%s)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SCRIPT="${ROOT_DIR}/inference/run_dataset.py"
ARGS=(
  --ckpt-path "${CKPT_PATH}"
  --config "${CONFIG}"
  --dataset longbenchv2
  --split train
  --limit "${LIMIT}"
  --batch-size "${BATCH_SIZE}"
  --kv-block-size "${KV_BLOCK_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --trace-out "${OUT_DIR}"
)

if [[ "${MP}" -gt 1 ]]; then
  torchrun --nproc-per-node "${MP}" "${SCRIPT}" "${ARGS[@]}"
else
  "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}"
fi

echo "Done. Trace outputs: ${OUT_DIR}"

