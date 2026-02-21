#!/usr/bin/env bash
set -euo pipefail

# Download LongBench v2 via Hugging Face.
# Source: zai-org/LongBench-v2 (arXiv:2412.15204)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/longbenchv2"

mkdir -p "${OUT_DIR}"
export OUT_DIR

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import os
from datasets import load_dataset

out_dir = os.environ["OUT_DIR"]
ds = load_dataset("zai-org/LongBench-v2", split="train")
path = os.path.join(out_dir, "longbenchv2_train.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")
print("Wrote", path, "rows=", len(ds))
PY

echo "Done. Output: ${OUT_DIR}"

