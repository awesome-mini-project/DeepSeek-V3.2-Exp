#!/usr/bin/env bash
set -uo pipefail
# Note: we do NOT use set -e globally; error handling is per-dataset via run_one().

# Run DSA trace collection on all four datasets in one go.
#
# Two independent dimensions control the run scale:
#
#   DATA=smoke|default|full    How many samples to take from each dataset
#   GEN=short|default|full     How many tokens to generate per sample
#
# Combinations:
#   DATA=smoke   GEN=short    → 2 samples, 8 tokens   (pipeline smoke test, ~30s)
#   DATA=default GEN=default  → 64 samples, 64 tokens  (daily trace collection)
#   DATA=full    GEN=default  → all samples, 64 tokens  (full data, controlled decode)
#   DATA=default GEN=full     → 64 samples, until EOS   (natural generation length)
#   DATA=full    GEN=full     → all samples, until EOS   (exhaustive, may take hours)
#
# Shorthand MODE (backward compat):
#   MODE=smoke   → DATA=smoke  GEN=short
#   MODE=default → DATA=default GEN=default
#   MODE=full    → DATA=full   GEN=full
#
# Required env:
#   CKPT_PATH     - converted checkpoint directory
#
# Optional env:
#   DATA / GEN         - see above (default: "default" / "default")
#   MODE               - shorthand; overridden by explicit DATA/GEN
#   KV_BLOCK_SIZE      - logical block size (default: 64)
#   MP                 - model parallel world size (default: 8)
#   TEMPERATURE        - sampling temperature (default: 0.6; 0 = deterministic)
#   BURSTGPT_LIMIT     - rows to export from BurstGPT HF (default: 2000)
#   SKIP_DOWNLOAD      - set to 1 to skip BurstGPT CSV download if it exists
#   CONTINUE_ON_ERROR  - set to 1 to keep running remaining datasets if one fails (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CKPT_PATH="${CKPT_PATH:?Please set CKPT_PATH}"
MODE="${MODE:-default}"
BURSTGPT_LIMIT="${BURSTGPT_LIMIT:-2000}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# --- Apply MODE shorthand if DATA/GEN not explicitly set ---
if [[ -z "${DATA:-}" ]]; then
  case "${MODE}" in
    smoke)   DATA=smoke   ;;
    full)    DATA=full    ;;
    *)       DATA=default ;;
  esac
fi
if [[ -z "${GEN:-}" ]]; then
  case "${MODE}" in
    smoke)   GEN=short   ;;
    full)    GEN=full    ;;
    *)       GEN=default ;;
  esac
fi

# --- Validate DATA ---
case "${DATA}" in
  smoke)   export LIMIT=2   ;;
  default) export LIMIT=64  ;;
  full)    export LIMIT=0   ;;
  *)
    echo "ERROR: invalid DATA='${DATA}'. Must be one of: smoke, default, full" >&2
    exit 1
    ;;
esac

# --- Validate GEN ---
case "${GEN}" in
  short)   export MAX_NEW_TOKENS=8  ;;
  default) export MAX_NEW_TOKENS=64 ;;
  full)    export MAX_NEW_TOKENS=0  ;;
  *)
    echo "ERROR: invalid GEN='${GEN}'. Must be one of: short, default, full" >&2
    exit 1
    ;;
esac

echo "=== run_all_traces.sh ==="
echo "  DATA=${DATA}  (LIMIT=${LIMIT})"
echo "  GEN=${GEN}   (MAX_NEW_TOKENS=${MAX_NEW_TOKENS})"
echo "  CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo ""
echo "  CKPT_PATH       = ${CKPT_PATH}"
echo "  KV_BLOCK_SIZE   = ${KV_BLOCK_SIZE:-64}"
echo "  MP              = ${MP:-8}"
echo "  TEMPERATURE     = ${TEMPERATURE:-0.6}"
echo ""

FAILED=()

run_one() {
  local name="$1"
  shift
  echo "========================================"
  echo "  [${name}] starting..."
  echo "========================================"
  local t0 rc
  t0=$(date +%s)
  set +e
  "$@"
  rc=$?
  set -e
  local t1
  t1=$(date +%s)
  if [[ ${rc} -ne 0 ]]; then
    echo "  [${name}] FAILED (exit code ${rc}) after $(( t1 - t0 ))s"
    FAILED+=("${name}")
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "  Stopping. Set CONTINUE_ON_ERROR=1 to skip failures and continue."
      exit ${rc}
    fi
  else
    echo "  [${name}] done in $(( t1 - t0 ))s"
  fi
  echo ""
}

# --- Run all four datasets ---

run_one "RULER" "${ROOT_DIR}/scripts/run_trace_ruler.sh"

run_one "LongBench-v2" "${ROOT_DIR}/scripts/run_trace_longbenchv2.sh"

run_one "ShareGPT" "${ROOT_DIR}/scripts/run_trace_sharegpt.sh"

# BurstGPT: download CSV if needed, then run.
BURSTGPT_CSV="${ROOT_DIR}/data/burstgpt/burstgpt_train_limit${BURSTGPT_LIMIT}.csv"
if [[ "${SKIP_DOWNLOAD}" != "1" ]] || [[ ! -f "${BURSTGPT_CSV}" ]]; then
  echo "[BurstGPT] Downloading CSV (LIMIT=${BURSTGPT_LIMIT})..."
  LIMIT="${BURSTGPT_LIMIT}" "${ROOT_DIR}/scripts/datasets/download_burstgpt.sh"
fi
export BURSTGPT_CSV
run_one "BurstGPT" "${ROOT_DIR}/scripts/run_trace_burstgpt.sh"

# --- Summary ---
echo "========================================"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  DONE WITH FAILURES: ${FAILED[*]}"
else
  echo "  ALL DONE (no failures)"
fi
echo "========================================"
echo ""
echo "Outputs are in: ${ROOT_DIR}/outputs/"
ls -dt "${ROOT_DIR}"/outputs/*/ 2>/dev/null | head -10

if [[ ${#FAILED[@]} -gt 0 ]]; then
  exit 1
fi
