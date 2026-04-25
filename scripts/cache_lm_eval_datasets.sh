#!/usr/bin/env bash
set -euo pipefail

# Pre-cache common lm-eval datasets in the local Hugging Face datasets cache.
# Usage:
#   bash scripts/cache_lm_eval_datasets.sh
# Optional:
#   HTTP_PROXY=http://162.105.146.48:7890 HTTPS_PROXY=http://162.105.146.48:7890 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   bash scripts/cache_lm_eval_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gptq/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

echo "[Info] Using python: ${PYTHON_BIN}"
echo "[Info] HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
echo "[Info] HF_ENDPOINT=${HF_ENDPOINT:-<default>}"
echo "[Info] http_proxy=${http_proxy:-${HTTP_PROXY:-<empty>}}"
echo "[Info] https_proxy=${https_proxy:-${HTTPS_PROXY:-<empty>}}"

"${PYTHON_BIN}" - <<'PY'
import traceback
from datasets import load_dataset

# (dataset_id_candidates, config, split_to_touch)
# The split is only touched to trigger download/cache. Dataset IDs are tried in order.
TARGETS = [
    (("piqa",), None, "validation"),
    (("Rowan/hellaswag", "hellaswag"), None, "validation"),
    (("allenai/ai2_arc", "ai2_arc"), "ARC-Easy", "validation"),
    (("allenai/ai2_arc", "ai2_arc"), "ARC-Challenge", "validation"),
    (("winogrande",), "winogrande_debiased", "validation"),
    (("allenai/openbookqa", "openbookqa"), "main", "validation"),
    (("social_i_qa",), None, "validation"),
    (("google/boolq", "boolq"), None, "validation"),
    (("EleutherAI/lambada_openai",), None, "test"),
    (("ceval/ceval-exam",), None, "validation"),
]

ok = []
failed = []

for ds_names, cfg, split in TARGETS:
    done = False
    last_err = None
    for ds_name in ds_names:
        tag = f"{ds_name}" + (f"[{cfg}]" if cfg else "")
        try:
            print(f"[Cache] {tag} split={split}")
            ds = load_dataset(ds_name, cfg, split=split)
            _ = len(ds)
            ok.append(tag)
            print(f"[OK] {tag}")
            done = True
            break
        except Exception as e:
            last_err = e
            print(f"[WARN] {tag} failed: {e}")
            traceback.print_exc(limit=1)
    if not done:
        failed.append((str(ds_names), repr(last_err)))

print("\n==== Cache Summary ====")
print(f"Success: {len(ok)}")
for x in ok:
    print(f"  - {x}")
print(f"Failed: {len(failed)}")
for x, err in failed:
    print(f"  - {x}: {err}")
PY

echo "[Done] Dataset cache prefetch finished."
