#!/usr/bin/env bash
set -euo pipefail

# Download BurstGPT traces via Hugging Face (or your configured source).
# Dataset: lzzmm/BurstGPT (arXiv:2401.17644)
#
# Notes:
# - The full dataset is large. Use LIMIT to export a small CSV for quick experiments.
# - The exported CSV keeps the common columns used by `inference/run_burstgpt.py`.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATAS_DIR:-${DATA_ROOT:-${ROOT_DIR}/data}}"

# Force HuggingFace caches to live under the repo (avoid ~/.cache/huggingface).
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

OUT_DIR="${DATA_ROOT}/burstgpt"
DATASET_NAME="${DATASET_NAME:-lzzmm/BurstGPT}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-2000}"

mkdir -p "${OUT_DIR}"
export OUT_DIR DATASET_NAME SPLIT LIMIT

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import csv
import os
from datasets import load_dataset

out_dir = os.environ["OUT_DIR"]
name = os.environ["DATASET_NAME"]
split = os.environ["SPLIT"]
limit = int(os.environ.get("LIMIT", "2000"))

ds = load_dataset(name, split=split)
path = os.path.join(out_dir, f"burstgpt_{split}_limit{limit}.csv")

cols = ["Log Type", "Timestamp", "Request tokens", "Response tokens", "Total tokens", "Model"]
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    n = 0
    for ex in ds:
        row = {k: ex.get(k, "") for k in cols}
        w.writerow(row)
        n += 1
        if limit > 0 and n >= limit:
            break
print("Wrote", path, "rows=", n)
PY

echo "Done. Output: ${OUT_DIR}"

