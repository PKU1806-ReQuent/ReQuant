#!/usr/bin/env bash
set -euo pipefail

# Download PTQ datasets into the local paths used by ReQuant:
#   ./datasets/wikitext
#   ./datasets/ultrachat_2k
#   ./datasets/NuminaMath-1.5
#
# Usage:
#   bash scripts/cache_ptq_local_datasets.sh
#
# Optional:
#   HF_ENDPOINT=https://hf-mirror.com \
#   PYTHON_BIN=/root/miniconda3/envs/requant/bin/python \
#   bash scripts/cache_ptq_local_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/requant/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

echo "[Info] Using python: ${PYTHON_BIN}"
echo "[Info] HF_ENDPOINT=${HF_ENDPOINT:-<default>}"
echo "[Info] target dir: ${REPO_ROOT}/datasets"

mkdir -p datasets/wikitext datasets/ultrachat_2k datasets/NuminaMath-1.5

"${PYTHON_BIN}" - <<'PY'
from datasets import load_dataset

print("[Step] wikitext-2-raw-v1 -> datasets/wikitext")
load_dataset("wikitext", "wikitext-2-raw-v1", split="train").to_json("datasets/wikitext/train.jsonl")
load_dataset("wikitext", "wikitext-2-raw-v1", split="validation").to_json("datasets/wikitext/validation.jsonl")
load_dataset("wikitext", "wikitext-2-raw-v1", split="test").to_json("datasets/wikitext/test.jsonl")

print("[Step] HuggingFaceH4/ultrachat_200k -> datasets/ultrachat_2k")
# data_utils.py reads the train_sft and test_sft splits.
load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:3000]").to_json("datasets/ultrachat_2k/train_sft.jsonl")
load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:300]").to_json("datasets/ultrachat_2k/test_sft.jsonl")

print("[Step] AI-MO/NuminaMath-TIR -> datasets/NuminaMath-1.5")
load_dataset("AI-MO/NuminaMath-TIR", split="train[:5000]").to_json("datasets/NuminaMath-1.5/train.jsonl")

print("[Done] local PTQ datasets are ready.")
PY

echo "[Check] Output files:"
ls -lh datasets/wikitext datasets/ultrachat_2k datasets/NuminaMath-1.5
