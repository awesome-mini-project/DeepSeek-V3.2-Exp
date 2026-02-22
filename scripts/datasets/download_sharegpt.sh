#!/usr/bin/env bash
set -euo pipefail

# Download a ShareGPT-style conversation dataset via Hugging Face.
# Default: anon8231489123/ShareGPT_Vicuna_unfiltered

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATAS_DIR:-${DATA_ROOT:-${ROOT_DIR}/data}}"

# Force HuggingFace caches to live under the repo (avoid ~/.cache/huggingface).
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

OUT_DIR="${DATA_ROOT}/sharegpt"
DATASET_NAME="${DATASET_NAME:-anon8231489123/ShareGPT_Vicuna_unfiltered}"
SPLIT="${SPLIT:-train}"

mkdir -p "${OUT_DIR}"
export OUT_DIR DATASET_NAME SPLIT

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import os
from datasets import load_dataset

out_dir = os.environ["OUT_DIR"]
name = os.environ["DATASET_NAME"]
split = os.environ["SPLIT"]

ds = load_dataset(name, split=split)
path = os.path.join(out_dir, f"sharegpt_{split}.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")
print("Wrote", path, "rows=", len(ds))
PY

echo "Done. Output: ${OUT_DIR}"

