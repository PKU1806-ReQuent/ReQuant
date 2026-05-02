#!/usr/bin/env bash
set -euo pipefail

# Pre-cache the lm-eval-harness datasets used by utils.eval_utils.qa_eval into
# the local Hugging Face datasets cache (HF_HOME). This is a superset of what
# the run-time eval needs so subsequent `--lm_eval` runs work fully offline.
#
# The task list mirrors qa_eval's defaults:
#   piqa, hellaswag, arc_easy, arc_challenge, winogrande, openbookqa,
#   social_iqa, boolq, lambada_openai, ceval-valid (52 subjects)
#
# Usage:
#   bash scripts/cache_lm_eval_datasets.sh
# Optional environment overrides (examples):
#   HTTP_PROXY=http://proxy.example.com:7890 HTTPS_PROXY=http://proxy.example.com:7890 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   bash scripts/cache_lm_eval_datasets.sh
#
# NOTE: (dataset_path, dataset_name, splits) are resolved by parsing the task
# YAMLs shipped with the installed lm-eval-harness (offline, no HEAD requests),
# so the cached datasets stay aligned with what the eval actually requests
# (parquet-backed HF datasets).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "[Info] Using python: ${PYTHON_BIN}"
echo "[Info] HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}"
echo "[Info] HF_ENDPOINT=${HF_ENDPOINT:-<default>}"
echo "[Info] http_proxy=${http_proxy:-${HTTP_PROXY:-<empty>}}"
echo "[Info] https_proxy=${https_proxy:-${HTTPS_PROXY:-<empty>}}"

"${PYTHON_BIN}" - <<'PY'
import os
import glob
import traceback
from typing import Optional

import yaml
from datasets import load_dataset
import lm_eval

# Task list must stay in sync with utils/eval_utils.py::qa_eval.
WANT_TASKS = [
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "openbookqa",
    "social_iqa",
    "boolq",
    "lambada_openai",
    "ceval-valid",  # expands to all ceval-valid_* subject YAMLs
]

LM_EVAL_DIR = os.path.dirname(lm_eval.__file__)
TASKS_ROOT = os.path.join(LM_EVAL_DIR, "tasks")


def _load_yaml(path: str) -> dict:
    """Tolerant YAML load: skip !function tags and other custom ones."""
    class _Loader(yaml.SafeLoader):
        pass

    def _ignore_unknown(loader, tag, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return None

    _Loader.add_multi_constructor("!", _ignore_unknown)
    with open(path) as f:
        return yaml.load(f, Loader=_Loader) or {}


# Collect every task YAML keyed by its ``task`` field. Skip YAMLs whose ``task``
# is a list (group configs) -- the per-task children are picked up separately.
TASK_YAMLS: dict[str, str] = {}
INCLUDE_RESOLVE: dict[str, str] = {}
for yaml_path in glob.glob(os.path.join(TASKS_ROOT, "**/*.yaml"), recursive=True):
    try:
        data = _load_yaml(yaml_path)
    except Exception:
        continue
    if not isinstance(data, dict):
        continue
    name = data.get("task")
    if isinstance(name, str):
        TASK_YAMLS[name] = yaml_path
# Non-YAML include files (like ceval's _default_ceval_yaml) live next to task
# YAMLs; index them by basename so we can resolve "include:" references.
for fp in glob.glob(os.path.join(TASKS_ROOT, "**/_default_*yaml"), recursive=True):
    INCLUDE_RESOLVE[os.path.basename(fp)] = fp
for fp in glob.glob(os.path.join(TASKS_ROOT, "**/*.yaml"), recursive=True):
    INCLUDE_RESOLVE[os.path.basename(fp)] = fp


def _merge_included(cfg: dict, yaml_path: str) -> dict:
    """Inline ``include:`` parent so dataset_path/dataset_name/splits propagate."""
    inc = cfg.get("include")
    if not inc:
        return cfg
    # Resolve relative to the YAML's own directory first, then the global index.
    candidates = [os.path.join(os.path.dirname(yaml_path), inc), INCLUDE_RESOLVE.get(inc)]
    for parent in candidates:
        if parent and os.path.isfile(parent):
            try:
                parent_cfg = _merge_included(_load_yaml(parent), parent)
            except Exception:
                parent_cfg = {}
            merged = dict(parent_cfg)
            merged.update(cfg)  # child overrides parent
            return merged
    return cfg


def _expand(want: str) -> list[str]:
    if want in TASK_YAMLS:
        return [want]
    expanded = sorted(n for n in TASK_YAMLS if n.startswith(want + "_"))
    if not expanded:
        print(f"[WARN] task '{want}' not present in this lm-eval install; skipped.")
    return expanded


# Deduplicate (dataset_path, dataset_name, split) so shared parents (e.g.
# arc_easy+arc_challenge -> allenai/ai2_arc; all 52 ceval subjects ->
# ceval/ceval-exam) don't trigger repeat downloads.
want_triples: list[tuple[str, Optional[str], str]] = []
seen: set[tuple[str, Optional[str], str]] = set()

for top in WANT_TASKS:
    for name in _expand(top):
        yaml_path = TASK_YAMLS[name]
        try:
            cfg = _merge_included(_load_yaml(yaml_path), yaml_path)
        except Exception as e:
            print(f"[WARN] parse failed for {yaml_path}: {e}")
            continue
        path = cfg.get("dataset_path")
        if not path:
            continue
        subname = cfg.get("dataset_name")
        # Skip non-str subnames (e.g. !function generators).
        if subname is not None and not isinstance(subname, str):
            subname = None
        for split_key in ("validation_split", "test_split", "training_split", "fewshot_split"):
            split = cfg.get(split_key)
            if not isinstance(split, str):
                continue
            key = (path, subname, split)
            if key in seen:
                continue
            seen.add(key)
            want_triples.append(key)

print(f"\n==== Caching {len(want_triples)} dataset slices ====\n")

ok: list[str] = []
failed: list[tuple[str, str]] = []
for path, subname, split in want_triples:
    tag = f"{path}" + (f"[{subname}]" if subname else "") + f":{split}"
    try:
        print(f"[Cache] {tag}")
        ds = load_dataset(path, subname, split=split)
        _ = len(ds)
        ok.append(tag)
        print(f"[OK]    {tag}  (n={len(ds)})")
    except Exception as e:
        failed.append((tag, repr(e)))
        print(f"[WARN]  {tag} failed: {e}")
        traceback.print_exc(limit=1)

print("\n==== Cache Summary ====")
print(f"Success: {len(ok)}")
print(f"Failed:  {len(failed)}")
for tag, err in failed:
    print(f"  - {tag}: {err}")
PY

echo "[Done] Dataset cache prefetch finished."
