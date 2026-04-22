#!/bin/bash
# GPTAQ（非 Refined）：多卡数据并行累计 H / dXXT（逻辑见 gptq_utils/gptaq_dp_utils.py）
#
# 用法：
#   bash scripts/gptaq_dp.sh [模型路径] [可见 GPU 列表] [每节点进程数]
# 例：4 卡
#   bash scripts/gptaq_dp.sh ./modelzoo/Qwen3/Qwen3-1.7B "0,1,2,3" 4
#
# 超参与 scripts/gptaq.sh 对齐（N_SAMPLES / alpha）；多任务并行可设 MASTER_PORT。
# 省显存：--dp_fasterquant_master_only 或 export DP_FASTERQUANT_MASTER_ONLY=1
# 省显存：export DP_SHARD_INPS=1

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=gptaq_dp
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
ALPHA=${ALPHA:-0.25}
ROTATE_TAG=""
if [[ "${ROTATE:-1}" == "1" ]]; then
  ROTATE_TAG="_rot"
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    ROTATE_TAG="${ROTATE_TAG}_optrot"
  fi
fi
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/gptaq_dp_w4_ns${N_SAMPLES}_a${ALPHA}_dp${NPROC}${ROTATE_TAG}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29610}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gptq/bin/python}"

echo "[Info] DP ranks=${NPROC} GPUs=${GPU_LIST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs 在 logs/ 子目录)"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"

EXTRA_ARGS=()
if [[ "${DP_FASTERQUANT_MASTER_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_fasterquant_master_only)
  echo "[Info] dp_fasterquant_master_only=1 (H reduce→rank0; fasterquant+broadcast weights)"
fi
if [[ "${DP_SHARD_INPS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_shard_inps)
  echo "[Info] dp_shard_inps=1 (per-rank activation shard; rank0 CPU capture + scatter)"
fi
if [[ "${ROTATE:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--rotate)
  echo "[Info] rotate=1 (SpinQuant-style hadamard rotation)"
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    EXTRA_ARGS+=(--optimized_rotation_path "${OPTIMIZED_ROTATION_PATH}")
    echo "[Info] optimized_rotation_path=${OPTIMIZED_ROTATION_PATH}"
  fi
fi

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_gptaq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method gptaq --w_bits 4 --w_clip --act_order --w_asym \
    --alpha "${ALPHA}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    --lm_eval --lm_eval_batch_size 32
