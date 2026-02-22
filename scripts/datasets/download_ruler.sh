#!/usr/bin/env bash
set -euo pipefail

# Download + extract RULER dataset via Hugging Face.
# Source: allenai/ruler_data (arXiv:2404.06654)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATAS_DIR:-${DATA_ROOT:-${ROOT_DIR}/data}}"

# Force HuggingFace caches to live under the repo (avoid ~/.cache/huggingface).
HF_HOME="${HF_HOME:-${DATA_ROOT}/huggingface}"
export HF_HOME
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"

OUT_DIR="${DATA_ROOT}/ruler"
RULER_TGZ="${RULER_TGZ:-data_debug.tgz}"  # or data_100_samples.tgz

mkdir -p "${OUT_DIR}"
export OUT_DIR RULER_TGZ

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import os
import tarfile
from huggingface_hub import hf_hub_download

out_dir = os.environ.get("OUT_DIR")
tgz = os.environ.get("RULER_TGZ", "data_debug.tgz")
tgz_path = hf_hub_download(repo_id="allenai/ruler_data", filename=tgz, repo_type="dataset")
extract_dir = os.path.join(out_dir, f"extracted_{tgz.replace('.tgz','')}")
os.makedirs(extract_dir, exist_ok=True)
marker = os.path.join(extract_dir, ".extracted.ok")
if not os.path.exists(marker):
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(path=extract_dir)
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
print("Extracted", tgz, "to", extract_dir)
PY

echo "Done. Output: ${OUT_DIR}"

