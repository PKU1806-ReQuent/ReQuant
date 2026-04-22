#!/bin/bash
# 对已保存的量化 checkpoint 单独跑 lm-eval（与 gptaq.sh --lm_eval 相同任务表）
#
# 用法:
#   bash scripts/eval_lm_tasks.sh <HF模型路径> <量化.pt路径> [GPU或GPU列表]
# 例:
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Qwen3/Qwen3-0.6B \
#     ./outputs/Qwen3-0.6B/gptaq_refined_dp_w4_ns512_a0.5_s3_c2_dp2.pt 0
# 多卡：
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Qwen3/Qwen3-0.6B \
#     ./outputs/Qwen3-0.6B/gptaq_dp/gptaq_dp_w4_ns512_a0.25_dp4_rot.pt 0,1,2,3
# lm-eval 显存紧时（如 boolq loglikelihood OOM）:
#   LM_EVAL_BATCH_SIZE=8 bash scripts/eval_lm_tasks.sh <模型> <ckpt.pt> <GPU>
# 带 SpinQuant rotate 训练的 ckpt（如文件名含 _rot）会自动加 --rotate；
# 也可显式设置 ROTATE=1 强制开启。
# 需要联网下载/缓存数据集时，可挂实验室代理：
#   USE_LAB_PROXY=1 bash scripts/eval_lm_tasks.sh <模型> <ckpt.pt> <GPU>

MODEL_PATH=${1:?用法: $0 <模型目录> <checkpoint.pt> [GPU或GPU列表]}
QMODEL=${2:?用法: $0 <模型目录> <checkpoint.pt> [GPU或GPU列表]}
DEVICE=${3:-0}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export CUDA_VISIBLE_DEVICES=${DEVICE}
PYTHON_BIN="${PYTHON_BIN:-python}"
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
if [[ "${ROTATE:-0}" == "1" || "${QMODEL}" == *_rot.pt ]]; then
    EXTRA_ARGS+=(--rotate)
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/eval_lm_tasks.py" \
    --model "${MODEL_PATH}" \
    --load_qmodel_path "${QMODEL}" \
    --placement "${LM_EVAL_PLACEMENT}" \
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"
