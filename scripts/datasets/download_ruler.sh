#!/usr/bin/env bash
set -euo pipefail

# Download RULER dataset via Hugging Face.
# Source: allenai/ruler_data (arXiv:2404.06654)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/ruler"

mkdir -p "${OUT_DIR}"
export OUT_DIR

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import os
from datasets import load_dataset

out_dir = os.environ.get("OUT_DIR")
ds = load_dataset("allenai/ruler_data", split="train")
path = os.path.join(out_dir, "ruler_train.jsonl")
with open(path, "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(__import__("json").dumps(ex, ensure_ascii=False) + "\n")
print("Wrote", path, "rows=", len(ds))
PY

echo "Done. Output: ${OUT_DIR}"

