#!/bin/bash
# Run lm-eval for an existing quantized checkpoint.
#
# Usage:
#   bash scripts/eval_lm_tasks.sh <HF model path> <quantized .pt path> [GPU or GPU list]
# Example:
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Qwen3/Qwen3-0.6B \
#     ./outputs/Qwen3-0.6B/gptaq_refined_dp_w4_ns512_a0.5_s3_c2_dp2.pt 0
# Multi-GPU:
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Qwen3/Qwen3-0.6B \
#     ./outputs/Qwen3-0.6B/gptaq_dp/gptaq_dp_w4_ns512_a0.25_dp4_rot.pt 0,1,2,3
# Lower this when lm-eval runs out of memory:
#   LM_EVAL_BATCH_SIZE=8 bash scripts/eval_lm_tasks.sh <model> <ckpt.pt> <GPU>
# Checkpoints with "_rot" in the filename automatically enable --rotate; set ROTATE=1 to force it.
# Use USE_LAB_PROXY=1 when datasets need to be downloaded through the lab proxy.
#   USE_LAB_PROXY=1 bash scripts/eval_lm_tasks.sh <model> <ckpt.pt> <GPU>

MODEL_PATH=${1:?Usage: $0 <model path> <checkpoint.pt> [GPU or GPU list]}
QMODEL=${2:?Usage: $0 <model path> <checkpoint.pt> [GPU or GPU list]}
DEVICE=${3:-0}

if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(pwd)/${MODEL_PATH}"
fi
if [[ "${QMODEL}" != /* ]]; then
    QMODEL="$(pwd)/${QMODEL}"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export CUDA_VISIBLE_DEVICES=${DEVICE}
PYTHON_BIN="${PYTHON_BIN:-python}"
SEQ_LEN="${SEQ_LEN:-2048}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-32}"
if [[ "${DEVICE}" == *,* ]]; then
    LM_EVAL_PLACEMENT="${LM_EVAL_PLACEMENT:-dispatch}"
else
    LM_EVAL_PLACEMENT="${LM_EVAL_PLACEMENT:-auto}"
fi
if [[ "${USE_LAB_PROXY:-0}" == "1" ]]; then
    LAB_HTTP_PROXY="${LAB_HTTP_PROXY:-http://162.105.146.48:7890}"
    export http_proxy="${LAB_HTTP_PROXY}"
    export https_proxy="${LAB_HTTP_PROXY}"
    export HTTP_PROXY="${LAB_HTTP_PROXY}"
    export HTTPS_PROXY="${LAB_HTTP_PROXY}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
fi
EXTRA_ARGS=()
if [[ "${ROTATE:-0}" == "1" || "$(basename "${QMODEL}")" == *_rot* ]]; then
    EXTRA_ARGS+=(--rotate)
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/eval_lm_tasks.py" \
    --model "${MODEL_PATH}" \
    --load_qmodel_path "${QMODEL}" \
    --seq_len "${SEQ_LEN}" \
    --placement "${LM_EVAL_PLACEMENT}" \
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"
