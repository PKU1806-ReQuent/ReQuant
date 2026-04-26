#!/usr/bin/env bash
set -euo pipefail

# Download PTQ datasets as full Hugging Face dataset repos into
# ${DATASETS_DIR}/<local_name>, so that data_utils.py can load them fully
# offline via `load_dataset(local_dir, config, split=...)`.
#
# Layout produced (example DATASETS_DIR=/apdcephfs_fsgm/.../datasets):
#   ${DATASETS_DIR}/wikitext/
#   ${DATASETS_DIR}/LLM_compression_calibration/
#   ${DATASETS_DIR}/ultrachat_2k/
#   ${DATASETS_DIR}/NuminaMath-1.5/
#
# Usage:
#   DATASETS_DIR=/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets \
#   HF_ENDPOINT=https://hf-mirror.com \
#   PYTHON_BIN=/data/miniconda3/envs/requant/bin/python \
#   bash scripts/cache_ptq_local_datasets.sh
#
# After download, a symlink ${REPO_ROOT}/datasets -> ${DATASETS_DIR} is
# created (idempotent), so data_utils.py's `./datasets/<name>` lookup just
# works.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/requant/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

DATASETS_DIR="${DATASETS_DIR:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets}"
mkdir -p "${DATASETS_DIR}"

echo "[Info] python      : ${PYTHON_BIN}"
echo "[Info] HF_ENDPOINT : ${HF_ENDPOINT:-<default>}"
echo "[Info] DATASETS_DIR: ${DATASETS_DIR}"

DATASETS_DIR="${DATASETS_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
import sys
import traceback

from huggingface_hub import snapshot_download
from datasets import load_dataset

ROOT = os.environ["DATASETS_DIR"]

# (local_name, hf_repo_id, verify_fn)
# verify_fn probes a tiny split via load_dataset to make sure the local copy
# is a fully-usable dataset repo (not just raw jsonl).
TARGETS = [
    ("wikitext",
     "Salesforce/wikitext",
     lambda p: load_dataset(p, "wikitext-2-raw-v1", split="test[:2]")),
    ("LLM_compression_calibration",
     "neuralmagic/LLM_compression_calibration",
     lambda p: load_dataset(p, split="train[:2]")),
    ("ultrachat_2k",
     "HuggingFaceH4/ultrachat_200k",
     lambda p: load_dataset(p, split="test_sft[:2]")),
    ("NuminaMath-1.5",
     "AI-MO/NuminaMath-1.5",
     lambda p: load_dataset(p, split="train[:2]")),
]

failed = []
for local_name, repo_id, probe in TARGETS:
    dst = os.path.join(ROOT, local_name)
    print(f"\n[Step] {repo_id} -> {dst}")
    os.makedirs(dst, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dst,
            local_dir_use_symlinks=False,
            max_workers=8,
        )
        # sanity-check that datasets.load_dataset can read it offline.
        ds = probe(dst)
        n = len(ds)
        print(f"[OK] {local_name}: local load returns {n} rows from tiny probe split")
    except Exception as e:
        traceback.print_exc()
        failed.append((local_name, repo_id, repr(e)))

print("\n==== PTQ dataset download summary ====")
print(f"Root: {ROOT}")
for local_name, repo_id, _ in TARGETS:
    mark = "FAIL" if any(f[0] == local_name for f in failed) else "OK  "
    print(f"  [{mark}] {local_name:30s} <- {repo_id}")

if failed:
    print("\nFailures:")
    for name, rid, err in failed:
        print(f"  - {name} ({rid}): {err}")
    sys.exit(1)
PY

# Create/refresh ./datasets -> ${DATASETS_DIR} so the repo lookup path works.
LINK="${REPO_ROOT}/datasets"
if [[ -L "${LINK}" ]]; then
  CURRENT="$(readlink "${LINK}")"
  if [[ "${CURRENT}" != "${DATASETS_DIR}" ]]; then
    echo "[Info] updating symlink ${LINK} -> ${DATASETS_DIR} (was ${CURRENT})"
    rm "${LINK}"
    ln -s "${DATASETS_DIR}" "${LINK}"
  fi
elif [[ -e "${LINK}" ]]; then
  echo "[Warn] ${LINK} exists and is not a symlink; leaving it alone."
else
  ln -s "${DATASETS_DIR}" "${LINK}"
  echo "[Info] created symlink ${LINK} -> ${DATASETS_DIR}"
fi

echo "[Done] local PTQ datasets are ready under ${DATASETS_DIR}."
ls -lh "${DATASETS_DIR}"
