#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
DEVICE=${2}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
LM_EVAL=${LM_EVAL:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

EXTRA_ARGS=()
if [[ "${LM_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
    echo "[Info] lm_eval=1 batch_size=${LM_EVAL_BATCH_SIZE}"
else
    echo "[Info] lm_eval=0"
fi

# Execute the distributed run
python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model ${MODEL_PATH} \
    --exp bf16 \
    "${EXTRA_ARGS[@]}"
