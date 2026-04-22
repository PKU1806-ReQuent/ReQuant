#!/bin/bash

# 用法: bash scripts/gptaq.sh [模型路径] [GPU 编号]
# 改 GPTAQ 强度: 编辑下方 ALPHA（会写入文件名，避免与旧 run 混淆）

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
DEVICE=${2:-0}

MODEL_NAME=$(basename "${MODEL_PATH}")
N_SAMPLES=768
SEQ_LEN=2048
ALPHA=0.25

SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/gptaq_w4_ns${N_SAMPLES}_a${ALPHA}.pt"

export CUDA_VISIBLE_DEVICES=${DEVICE}
export HF_ENDPOINT="https://hf-mirror.com"
PYTHON_BIN="/root/miniconda3/envs/gptq/bin/python"

echo "[Info] model=${MODEL_PATH} CUDA_VISIBLE_DEVICES=${DEVICE} alpha=${ALPHA}"
echo "[Info] Quantized model will be saved to: ${SAVE_QMODEL_PATH}"

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model "${MODEL_PATH}" \
    --exp gptaq \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptaq --w_bits 4 --w_clip --act_order \
    --alpha ${ALPHA} \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --lm_eval --lm_eval_batch_size 32
