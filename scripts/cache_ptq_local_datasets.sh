#!/usr/bin/env bash
set -euo pipefail

# Mirror the Hugging Face dataset repos used by utils/data_utils.py into the
# local ./datasets/<name> directories expected by the codebase. Each local
# directory keeps the full HF repo layout (parquet + README + dataset_infos
# + per-config metadata), so the existing calls like
#   load_dataset("./datasets/wikitext", "wikitext-2-raw-v1", split="train")
#   load_dataset("./datasets/ultrachat_2k", split="train_sft[:256]")
# keep working unchanged and fully offline afterwards.
#
# NOTE: Writing per-split JSONL via .to_json() is NOT a drop-in replacement for
# the repo layout: load_dataset() cannot read a named config (e.g.
# "wikitext-2-raw-v1") from flat JSONL files, and HF split-slicing like
# "train_sft[:256]" requires the Arrow/parquet-backed layout. Always mirror the
# repo (parquet) via huggingface_hub.snapshot_download.
#
# Usage:
#   bash scripts/cache_ptq_local_datasets.sh
# Optional environment overrides (examples):
#   HF_ENDPOINT=https://hf-mirror.com \
#   HF_TOKEN=<token>                             # for gated datasets
#   SKIP_NEURALMAGIC=1                           # skip neuralmagic/LLM_compression_calibration
#   SKIP_NUMINAMATH=1                            # skip AI-MO/NuminaMath-1.5 (~large)
#   PYTHON_BIN=/path/to/python \
#   bash scripts/cache_ptq_local_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SKIP_NEURALMAGIC="${SKIP_NEURALMAGIC:-0}"
SKIP_NUMINAMATH="${SKIP_NUMINAMATH:-0}"

echo "[Info] Using python: ${PYTHON_BIN}"
echo "[Info] HF_ENDPOINT=${HF_ENDPOINT:-<default>}"
echo "[Info] target dir: ${REPO_ROOT}/datasets"
echo "[Info] SKIP_NEURALMAGIC=${SKIP_NEURALMAGIC} SKIP_NUMINAMATH=${SKIP_NUMINAMATH}"

export SKIP_NEURALMAGIC SKIP_NUMINAMATH

"${PYTHON_BIN}" - <<'PY'
import os
import sys
import traceback

from huggingface_hub import snapshot_download
from datasets import load_dataset

# (repo_id, local_subdir, optional sanity-check kwargs for load_dataset)
PLAN = [
    {
        "repo_id": "wikitext",
        "local": "datasets/wikitext",
        "verify": dict(name="wikitext-2-raw-v1", split="validation"),
    },
    {
        "repo_id": "HuggingFaceH4/ultrachat_200k",
        "local": "datasets/ultrachat_2k",
        "verify": dict(split="test_sft[:4]"),
    },
    {
        "repo_id": "AI-MO/NuminaMath-1.5",
        "local": "datasets/NuminaMath-1.5",
        "verify": dict(split="train[:4]"),
        "skip_env": "SKIP_NUMINAMATH",
    },
    {
        "repo_id": "neuralmagic/LLM_compression_calibration",
        "local": "datasets/LLM_compression_calibration",
        "verify": dict(split="train[:4]"),
        "skip_env": "SKIP_NEURALMAGIC",
    },
]

# Only grab the files needed for datasets.load_dataset() from a repo mirror:
#  - parquet shards (primary data)
#  - json metadata (dataset_infos.json, dataset_info.json, per-config config.json, .jsonl when used)
#  - README.md (optional but nice)
#  - arrow / csv fallbacks (some small datasets)
ALLOW_PATTERNS = [
    "*.parquet",
    "*.json",
    "*.jsonl",
    "*.arrow",
    "*.csv",
    "README.md",
    "LICENSE*",
]

ok = []
failed = []
skipped = []

for item in PLAN:
    repo_id = item["repo_id"]
    local_dir = item["local"]
    verify = item.get("verify", {})
    skip_env = item.get("skip_env")
    if skip_env and os.environ.get(skip_env, "0") == "1":
        skipped.append(f"{repo_id} (skip via {skip_env}=1)")
        print(f"[SKIP] {repo_id}  (env {skip_env}=1)")
        continue

    print(f"\n[Step] Mirror {repo_id}  ->  {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            allow_patterns=ALLOW_PATTERNS,
        )
    except Exception as e:
        failed.append((repo_id, repr(e)))
        print(f"[FAIL] snapshot_download {repo_id}: {e}")
        traceback.print_exc(limit=2)
        continue

    # Sanity-check that load_dataset can read it back with the exact call shape
    # used by utils/data_utils.py (i.e. path + optional config + HF split slice).
    try:
        kwargs = dict(verify)
        split = kwargs.pop("split", None)
        name = kwargs.pop("name", None)
        if name is not None:
            ds = load_dataset(local_dir, name, split=split)
        else:
            ds = load_dataset(local_dir, split=split)
        print(f"[OK]   {repo_id}  verify(load_dataset {verify}) -> n={len(ds)}")
        ok.append(repo_id)
    except Exception as e:
        failed.append((repo_id, f"verify failed: {e!r}"))
        print(f"[WARN] {repo_id} mirrored but verify failed: {e}")
        traceback.print_exc(limit=2)

print("\n==== Mirror Summary ====")
print(f"Success: {len(ok)}")
for x in ok:
    print(f"  - {x}")
if skipped:
    print(f"Skipped: {len(skipped)}")
    for x in skipped:
        print(f"  - {x}")
if failed:
    print(f"Failed:  {len(failed)}")
    for repo, err in failed:
        print(f"  - {repo}: {err}")
    sys.exit(1)
PY

echo
echo "[Check] Mirrored dataset layouts (top-level only):"
for d in datasets/wikitext datasets/ultrachat_2k datasets/NuminaMath-1.5 datasets/LLM_compression_calibration; do
    if [[ -d "${d}" ]]; then
        echo "  ${d}:"
        ls -lh "${d}" | head -n 20 | sed 's/^/    /'
    fi
done

echo "[Done] Local PTQ datasets are ready."
