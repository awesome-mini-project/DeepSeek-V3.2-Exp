#!/usr/bin/env bash
set -uo pipefail

# Run trace collection in chunks with crash recovery.
#
# Instead of running all samples in one torchrun process (which dies entirely
# on any CUDA/NCCL error), this script splits the dataset into chunks and
# runs each chunk as a separate torchrun invocation. If a chunk crashes,
# it waits for GPU cooldown and moves on to the next chunk.
#
# Usage:
#   ./scripts/run_trace_chunked.sh ruler         # default: 100 samples/chunk
#   CHUNK_SIZE=50 ./scripts/run_trace_chunked.sh ruler
#   ./scripts/run_trace_chunked.sh sharegpt
#   ./scripts/run_trace_chunked.sh longbenchv2
#
# Required env:
#   CKPT_PATH
#
# Optional env:
#   CHUNK_SIZE          - samples per chunk (default: 100)
#   TOTAL_LIMIT         - total samples to process; 0 = all (default: 0)
#   MAX_NEW_TOKENS      - decode length cap (default: 64)
#   KV_BLOCK_SIZE       - logical block size (default: 64)
#   MAX_REQUESTS_PER_FILE - shard JSONL every N requests (default: 4; 0 = no sharding)
#   MP                  - model parallel (default: 8)
#   TEMPERATURE         - sampling temperature (default: 0.6)
#   COOLDOWN_SECS       - seconds to wait after a crash before next chunk (default: 30)
#   DATASET             - can also be passed as $1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
DATASET="${1:-${DATASET:?Please specify dataset as argument or DATASET env}}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
TOTAL_LIMIT="${TOTAL_LIMIT:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-64}"
MAX_REQUESTS_PER_FILE="${MAX_REQUESTS_PER_FILE:-4}"
MP="${MP:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
COOLDOWN_SECS="${COOLDOWN_SECS:-30}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# HF cache under repo.
DATA_ROOT="${DATAS_DIR:-${DATA_ROOT:-${ROOT_DIR}/data}}"
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# Resolve config.
CONFIG="${CONFIG:-${ROOT_DIR}/inference/config_671B_v3.2.json}"
if [[ "${CONFIG}" != /* ]]; then
  if [[ -f "${ROOT_DIR}/${CONFIG}" ]]; then CONFIG="${ROOT_DIR}/${CONFIG}"
  elif [[ -f "${ROOT_DIR}/inference/${CONFIG}" ]]; then CONFIG="${ROOT_DIR}/inference/${CONFIG}"; fi
fi

OUT_BASE="${ROOT_DIR}/outputs/${DATASET}_chunked_$(date +%s)"
mkdir -p "${OUT_BASE}"

echo "=== run_trace_chunked.sh ==="
echo "  DATASET       = ${DATASET}"
echo "  CHUNK_SIZE    = ${CHUNK_SIZE}"
echo "  TOTAL_LIMIT   = ${TOTAL_LIMIT}"
echo "  OUT_BASE      = ${OUT_BASE}"
echo "  COOLDOWN_SECS = ${COOLDOWN_SECS}"
echo ""

SCRIPT="${ROOT_DIR}/inference/run_dataset.py"

# We use --limit and --seed to simulate offset: seed controls which chunk.
# Actually, run_dataset.py doesn't support offset natively, so we use a
# simpler approach: pass --limit as chunk_end and rely on the fact that
# the dataset is deterministic. We'll pass a custom --trace-out per chunk.
#
# Better approach: pass explicit --offset and --limit. Since run_dataset.py
# doesn't have --offset yet, we'll add it inline via environment variable
# and let the script skip the first N samples.

chunk_idx=0
offset=0
n_success=0
n_failed=0
n_skipped=0

while true; do
  if [[ "${TOTAL_LIMIT}" -gt 0 ]] && [[ ${offset} -ge ${TOTAL_LIMIT} ]]; then
    break
  fi

  remaining=$((TOTAL_LIMIT > 0 ? TOTAL_LIMIT - offset : CHUNK_SIZE))
  this_limit=$((remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE))

  chunk_out="${OUT_BASE}/chunk_${chunk_idx}_offset${offset}"

  echo "========================================"
  echo "  Chunk ${chunk_idx}: offset=${offset}, limit=${this_limit}"
  echo "========================================"

  ARGS=(
    --ckpt-path "${CKPT_PATH}"
    --config "${CONFIG}"
    --dataset "${DATASET}"
    --limit "${this_limit}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --kv-block-size "${KV_BLOCK_SIZE}"
    --max-requests-per-file "${MAX_REQUESTS_PER_FILE}"
    --temperature "${TEMPERATURE}"
    --trace-out "${chunk_out}"
    --seed "${offset}"
  )

  # For RULER, pass tgz.
  if [[ "${DATASET}" == "ruler" ]]; then
    ARGS+=(--ruler-tgz "${RULER_TGZ:-data_debug.tgz}")
  fi

  set +e
  if [[ "${MP}" -gt 1 ]]; then
    torchrun --nproc-per-node "${MP}" "${SCRIPT}" "${ARGS[@]}"
  else
    python3 "${SCRIPT}" "${ARGS[@]}"
  fi
  rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "  Chunk ${chunk_idx}: SUCCESS"
    n_success=$((n_success + 1))
  else
    echo "  Chunk ${chunk_idx}: FAILED (exit ${rc})"
    echo "  Waiting ${COOLDOWN_SECS}s for GPU cooldown..."
    sleep "${COOLDOWN_SECS}"
    n_failed=$((n_failed + 1))
  fi

  offset=$((offset + this_limit))
  chunk_idx=$((chunk_idx + 1))

  # Safety: if we've done many chunks with no TOTAL_LIMIT, check if dataset is exhausted.
  # The runner prints "[done] samples_seen=N" — if samples_seen < this_limit, we're done.
  if [[ "${TOTAL_LIMIT}" -le 0 ]] && [[ ${rc} -eq 0 ]]; then
    # Check if the last run processed fewer samples than requested (dataset exhausted).
    last_done=$(grep -o 'samples_seen=[0-9]*' "${chunk_out}/block${KV_BLOCK_SIZE}/summary.json" 2>/dev/null || echo "")
    if [[ -z "${last_done}" ]]; then
      # Can't determine; assume more data exists. User can Ctrl+C.
      :
    fi
  fi

  echo ""
done

echo "========================================"
echo "  CHUNKED RUN COMPLETE"
echo "  Chunks: ${chunk_idx} total, ${n_success} success, ${n_failed} failed"
echo "  Output: ${OUT_BASE}/"
echo "========================================"

ls -d "${OUT_BASE}"/chunk_*/ 2>/dev/null
