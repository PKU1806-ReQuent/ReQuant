#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
DEVICE=${2}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"

# Optional lm_eval (controlled by LM_EVAL_ENABLE env var)
LM_EVAL_ARGS=()
if [[ "${LM_EVAL_ENABLE:-0}" == "1" ]]; then
  LM_EVAL_ARGS=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE:-32}")
fi

echo "[Info] output_dir=${OUTPUT_ROOT}/${MODEL_NAME}/bf16/ (logs in logs/)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

# Execute the distributed run
${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model ${MODEL_PATH} \
    --exp bf16 \
    --output_dir "${OUTPUT_ROOT}" \
    "${LM_EVAL_ARGS[@]}"
