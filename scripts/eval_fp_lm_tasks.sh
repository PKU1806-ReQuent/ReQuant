#!/usr/bin/env bash
# Run full-precision lm-eval tasks:
# arc_challenge, arc_easy, boolq, ceval-valid, hellaswag,
# lambada_openai, openbookqa, piqa, social_iqa, winogrande

set -euo pipefail

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-4B}
DEVICE=${2:-0}
USE_HF_MIRROR=${USE_HF_MIRROR:-1}
HF_MIRROR_URL=${HF_MIRROR_URL:-https://hf-mirror.com}
# 默认走实验室代理，避免 lm-eval 里带 commit SHA 的 dataset script 绕过 HF_ENDPOINT 直连 huggingface.co
# 如需禁用代理，传 PROXY_URL="" 运行
PROXY_URL=${PROXY_URL-http://162.105.146.48:7890}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-4}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
TASKS=${TASKS:-}
PPL_CHUNK_SIZE=${PPL_CHUNK_SIZE:-16}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export CUDA_VISIBLE_DEVICES=${DEVICE}
export PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gptq/bin/python}"

if [[ "${USE_HF_MIRROR}" == "1" ]]; then
  export HF_ENDPOINT="${HF_MIRROR_URL}"
  export HUGGINGFACE_HUB_ENDPOINT="${HF_MIRROR_URL}"
fi

if [[ -n "${PROXY_URL}" ]]; then
  export http_proxy="${PROXY_URL}"
  export https_proxy="${PROXY_URL}"
  export HTTP_PROXY="${PROXY_URL}"
  export HTTPS_PROXY="${PROXY_URL}"
  export no_proxy="${no_proxy:-localhost,127.0.0.1,::1}"
  export NO_PROXY="${NO_PROXY:-${no_proxy}}"
fi

echo "[eval_fp_lm_tasks] MODEL_PATH=${MODEL_PATH}" >&2
echo "[eval_fp_lm_tasks] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
echo "[eval_fp_lm_tasks] LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE}" >&2
echo "[eval_fp_lm_tasks] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" >&2
echo "[eval_fp_lm_tasks] PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF}" >&2
echo "[eval_fp_lm_tasks] PPL_CHUNK_SIZE=${PPL_CHUNK_SIZE}" >&2
echo "[eval_fp_lm_tasks] USE_HF_MIRROR=${USE_HF_MIRROR}" >&2
if [[ "${USE_HF_MIRROR}" == "1" ]]; then
  echo "[eval_fp_lm_tasks] HF_ENDPOINT=${HF_ENDPOINT}" >&2
fi
if [[ -n "${PROXY_URL}" ]]; then
  echo "[eval_fp_lm_tasks] PROXY_URL=${PROXY_URL}" >&2
fi
if [[ -n "${TASKS}" ]]; then
  echo "[eval_fp_lm_tasks] TASKS=${TASKS}" >&2
fi

CMD=(
  "${PYTHON_BIN}" "${REPO_ROOT}/eval_fp_lm_tasks.py"
  --model "${MODEL_PATH}"
  --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}"
  --ppl_chunk_size "${PPL_CHUNK_SIZE}"
)
if [[ -n "${TASKS}" ]]; then
  CMD+=(--tasks "${TASKS}")
fi

exec "${CMD[@]}"
