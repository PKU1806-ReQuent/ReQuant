#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
NUM_GROUPS=${2}     # 4
DEVICE=${3}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=1024
SEQ_LEN=2048
ROTATE=${ROTATE:-0}
ROTATE_TAG=""
if [[ "${ROTATE}" == "1" ]]; then
    ROTATE_TAG="_rot"
    [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]] && ROTATE_TAG="${ROTATE_TAG}_optrot"
fi
# Quantized checkpoint (state_dict + w_quantizers); dirname is created by ptq
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/gptq_guided_w4_g${NUM_GROUPS}${ROTATE_TAG}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

echo "[Info] Quantized model will be saved to: ${SAVE_QMODEL_PATH}"

EXTRA_ARGS=()
if [[ "${ROTATE}" == "1" ]]; then
    EXTRA_ARGS+=(--rotate)
    if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
        EXTRA_ARGS+=(--optimized_rotation_path "${OPTIMIZED_ROTATION_PATH}")
    fi
fi

# Execute the distributed run
python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model ${MODEL_PATH} \
    --exp gptq_guided \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptq_guided --w_bits 4 --w_clip --num_groups ${NUM_GROUPS} --act_order \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --lm_eval --lm_eval_batch_size 32 \
