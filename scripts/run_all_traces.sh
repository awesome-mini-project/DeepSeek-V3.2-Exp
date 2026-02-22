#!/usr/bin/env bash
set -euo pipefail

# Run DSA trace collection on all four datasets in one go.
#
# Usage:
#   ./scripts/run_all_traces.sh              # default mode (64 samples each)
#   MODE=smoke ./scripts/run_all_traces.sh   # smoke test (2 samples each, fast)
#   MODE=full  ./scripts/run_all_traces.sh   # full dataset (all samples)
#
# Required env:
#   CKPT_PATH   - converted checkpoint directory
#
# Optional env:
#   MODE          - "default" / "smoke" / "full" (default: "default")
#   CKPT_PATH     - model checkpoint path (required)
#   KV_BLOCK_SIZE - logical block size (default: 64)
#   MP            - model parallel world size (default: 8)
#   TEMPERATURE   - sampling temperature (default: 0.6; 0 = deterministic)
#   BURSTGPT_LIMIT - how many rows to export from BurstGPT CSV (default: 2000)
#   SKIP_DOWNLOAD  - set to 1 to skip BurstGPT CSV download if already exists

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
MODE="${MODE:-default}"
BURSTGPT_LIMIT="${BURSTGPT_LIMIT:-2000}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

case "${MODE}" in
  smoke)
    export LIMIT=2
    export MAX_NEW_TOKENS=8
    echo "=== SMOKE TEST MODE: 2 samples, 8 max_new_tokens ==="
    ;;
  full)
    export LIMIT=0
    export MAX_NEW_TOKENS=0
    echo "=== FULL MODE: all samples, generate until EOS ==="
    ;;
  *)
    export LIMIT=64
    export MAX_NEW_TOKENS=64
    echo "=== DEFAULT MODE: 64 samples, 64 max_new_tokens ==="
    ;;
esac

echo ""
echo "CKPT_PATH       = ${CKPT_PATH}"
echo "MODE            = ${MODE}"
echo "LIMIT           = ${LIMIT}"
echo "MAX_NEW_TOKENS  = ${MAX_NEW_TOKENS}"
echo "KV_BLOCK_SIZE   = ${KV_BLOCK_SIZE:-64}"
echo "MP              = ${MP:-8}"
echo "TEMPERATURE     = ${TEMPERATURE:-0.6}"
echo ""

run_one() {
  local name="$1"
  shift
  echo "========================================"
  echo "  [${name}] starting..."
  echo "========================================"
  local t0
  t0=$(date +%s)
  "$@"
  local t1
  t1=$(date +%s)
  echo "  [${name}] done in $(( t1 - t0 ))s"
  echo ""
}

run_one "RULER" "${ROOT_DIR}/scripts/run_trace_ruler.sh"

run_one "LongBench-v2" "${ROOT_DIR}/scripts/run_trace_longbenchv2.sh"

run_one "ShareGPT" "${ROOT_DIR}/scripts/run_trace_sharegpt.sh"

BURSTGPT_CSV="${ROOT_DIR}/data/burstgpt/burstgpt_train_limit${BURSTGPT_LIMIT}.csv"
if [[ "${SKIP_DOWNLOAD}" != "1" ]] || [[ ! -f "${BURSTGPT_CSV}" ]]; then
  echo "[BurstGPT] Downloading CSV (LIMIT=${BURSTGPT_LIMIT})..."
  LIMIT="${BURSTGPT_LIMIT}" "${ROOT_DIR}/scripts/datasets/download_burstgpt.sh"
fi
export BURSTGPT_CSV
run_one "BurstGPT" "${ROOT_DIR}/scripts/run_trace_burstgpt.sh"

echo "========================================"
echo "  ALL DONE"
echo "========================================"
echo ""
echo "Outputs are in: ${ROOT_DIR}/outputs/"
ls -dt "${ROOT_DIR}"/outputs/*/ 2>/dev/null | head -10
