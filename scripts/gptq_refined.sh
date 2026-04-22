#!/bin/bash

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
# 默认避开常占用的 1 号卡；指定卡: bash scripts/gptq_refined.sh <模型路径> <GPU编号>
DEVICE=${2:-2}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=gptaq_refined
N_SAMPLES=512
SEQ_LEN=2048

ALPHA=0.25
REFINE_SWEEPS=4
REFINE_CANDIDATES=1
REFINE_MIN_GAIN_EPS=${REFINE_MIN_GAIN_EPS:-0}

SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/gptaq_refined_w4_ns${N_SAMPLES}_a${ALPHA}_s${REFINE_SWEEPS}_c${REFINE_CANDIDATES}.pt"

export CUDA_VISIBLE_DEVICES=${DEVICE}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((29510 + DEVICE))}"
export HF_ENDPOINT="https://hf-mirror.com"
PYTHON_BIN="/root/miniconda3/envs/gptq/bin/python"

echo "[Info] model=${MODEL_PATH} CUDA_VISIBLE_DEVICES=${DEVICE} refine_min_gain_eps=${REFINE_MIN_GAIN_EPS} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs 在 logs/ 子目录)"
echo "[Info] Quantized model will be saved to: ${SAVE_QMODEL_PATH}"

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=1 ./ptq.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples ${N_SAMPLES} --seq_len ${SEQ_LEN} \
    --w_method gptaq_refined --w_bits 4 --w_clip --act_order \
    --alpha ${ALPHA} --refine_sweeps ${REFINE_SWEEPS} --refine_candidates ${REFINE_CANDIDATES} \
    --refine_min_gain_eps ${REFINE_MIN_GAIN_EPS} \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    --lm_eval --lm_eval_batch_size 32
