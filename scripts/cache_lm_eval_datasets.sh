#!/usr/bin/env bash
set -euo pipefail

# Pre-cache common lm-eval datasets into a persistent HF cache directory on
# the shared disk so subsequent runs can work fully offline.
#
# Usage:
#   HF_HOME=/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache \
#   HF_ENDPOINT=https://hf-mirror.com \
#   bash scripts/cache_lm_eval_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-/data/miniconda3/envs/requant/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

# Default cache location: shared disk so it is reusable across jobs.
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"
mkdir -p "${HF_HOME}"

echo "[Info] python      : ${PYTHON_BIN}"
echo "[Info] HF_HOME     : ${HF_HOME}"
echo "[Info] HF_ENDPOINT : ${HF_ENDPOINT:-<default>}"
echo "[Info] http_proxy  : ${http_proxy:-${HTTP_PROXY:-<empty>}}"
echo "[Info] https_proxy : ${https_proxy:-${HTTPS_PROXY:-<empty>}}"

"${PYTHON_BIN}" - <<'PY'
import traceback
from datasets import load_dataset

# (dataset_id_candidates, config, split_to_touch, extra_kwargs)
# The split is only touched to trigger download/cache. Dataset IDs are tried in order.
TARGETS = [
    (("piqa",), None, "validation", {"trust_remote_code": True}),
    (("Rowan/hellaswag", "hellaswag"), None, "validation", {}),
    (("allenai/ai2_arc", "ai2_arc"), "ARC-Easy", "validation", {}),
    (("allenai/ai2_arc", "ai2_arc"), "ARC-Challenge", "validation", {}),
    (("winogrande",), "winogrande_debiased", "validation", {"trust_remote_code": True}),
    (("allenai/openbookqa", "openbookqa"), "main", "validation", {}),
    (("social_i_qa", "allenai/social_i_qa"), None, "validation", {"trust_remote_code": True}),
    (("google/boolq", "boolq"), None, "validation", {}),
    (("EleutherAI/lambada_openai",), None, "test", {}),
]

ok = []
failed = []

for ds_names, cfg, split, extra in TARGETS:
    done = False
    last_err = None
    for ds_name in ds_names:
        tag = f"{ds_name}" + (f"[{cfg}]" if cfg else "")
        try:
            print(f"[Cache] {tag} split={split} kwargs={extra}")
            ds = load_dataset(ds_name, cfg, split=split, **extra)
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

# ceval/ceval-exam has many configs (one per subject). The lm_eval task
# 'ceval-valid' iterates them all, so we pre-cache every config.
try:
    from datasets import get_dataset_config_names
    configs = get_dataset_config_names("ceval/ceval-exam")
    print(f"[Cache] ceval/ceval-exam configs: {len(configs)}")
    for cfg in configs:
        try:
            load_dataset("ceval/ceval-exam", cfg, split="val")
            ok.append(f"ceval/ceval-exam[{cfg}]")
        except Exception as e:
            print(f"[WARN] ceval/ceval-exam[{cfg}]: {e}")
            failed.append((f"ceval/ceval-exam[{cfg}]", repr(e)))
    print("[OK] ceval/ceval-exam (all configs attempted)")
except Exception as e:
    print(f"[WARN] ceval enumerate failed: {e}")
    failed.append(("ceval/ceval-exam", repr(e)))

print("\n==== Cache Summary ====")
print(f"Success: {len(ok)}")
for x in ok:
    print(f"  - {x}")
print(f"Failed: {len(failed)}")
for x, err in failed:
    print(f"  - {x}: {err}")
PY

echo "[Done] Dataset cache prefetch finished (HF_HOME=${HF_HOME})."
