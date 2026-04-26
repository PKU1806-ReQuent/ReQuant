#!/bin/bash
# AWQ launcher with data-parallel activation statistics and layer-wise scale search.
#
# Usage:
#   bash scripts/awq_dp.sh [model path] [visible GPU list] [nproc]
# Example:
#   bash scripts/awq_dp.sh ./modelzoo/Qwen3/Qwen3-1.7B "0,1,2,3" 4

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=awq_dp
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
AWQ_GRID=${AWQ_GRID:-20}
AWQ_MIN_ALPHA=${AWQ_MIN_ALPHA:-0.0}
AWQ_MAX_ALPHA=${AWQ_MAX_ALPHA:-1.0}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-1}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.003}
REQUANT_TAG=""
if [[ "${REQUANT}" == "1" ]]; then
  REQUANT_TAG="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
fi
ROTATE_TAG=""
if [[ "${ROTATE:-1}" == "1" ]]; then
  ROTATE_TAG="_rot"
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    ROTATE_TAG="${ROTATE_TAG}_optrot"
  fi
fi
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/awq_dp_w4_ns${N_SAMPLES}_g${AWQ_GRID}_a${AWQ_MIN_ALPHA}-${AWQ_MAX_ALPHA}_dp${NPROC}${ROTATE_TAG}${REQUANT_TAG}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29630}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "[Info] DP ranks=${NPROC} GPUs=${GPU_LIST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs in logs/)"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"
echo "[Info] requant=${REQUANT}"

EXTRA_ARGS=()
if [[ "${DP_SHARD_INPS:-1}" == "1" ]]; then
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
if [[ "${REQUANT}" == "1" ]]; then
  EXTRA_ARGS+=(
    --requant_sweeps "${REQUANT_SWEEPS}"
    --requant_candidates "${REQUANT_CANDIDATES}"
    --requant_min_gain_eps "${REQUANT_MIN_GAIN_EPS}"
  )
  echo "[Info] requant sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
fi

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method awq --w_bits 4 --w_clip --w_asym \
    --awq_grid "${AWQ_GRID}" \
    --awq_min_alpha "${AWQ_MIN_ALPHA}" \
    --awq_max_alpha "${AWQ_MAX_ALPHA}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps
