#!/bin/bash

# Input arguments
MODEL_PATH=${1}     # ./modelzoo/Qwen3/Qwen3-0.6B
NUM_GROUPS=${2}     # 4
DEVICE=${3}         # 0

MODEL_NAME=$(basename ${MODEL_PATH})
N_SAMPLES=1024
SEQ_LEN=2048
# Quantized checkpoint (state_dict + w_quantizers); dirname is created by ptq
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/gptq_guided_w4_g${NUM_GROUPS}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}

echo "[Info] Quantized model will be saved to: ${SAVE_QMODEL_PATH}"

# Execute the distributed run
python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model ${MODEL_PATH} \
    --exp gptq_guided \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptq_guided --w_bits 4 --w_clip --num_groups ${NUM_GROUPS} --act_order \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --lm_eval --lm_eval_batch_size 32 \
