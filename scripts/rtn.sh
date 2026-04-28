#!/bin/bash

# Input arguments
MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-0}
NPROC=${3:-1}

MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
EXP=${EXP:-rtn_dp}
N_SAMPLES=${N_SAMPLES:-1024}
SEQ_LEN=${SEQ_LEN:-2048}
A_BITS=${A_BITS:-16}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-1}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.003}
ROTATE=${ROTATE:-0}
OFFLOAD_INPS=${OFFLOAD_INPS:-1}
OFFLOAD_INPS_FLAG=""
if [[ "${OFFLOAD_INPS}" == "1" ]]; then OFFLOAD_INPS_FLAG="--offload_inps"; fi
ACT_TAG=""
if [[ "${A_BITS}" != "16" ]]; then
  ACT_TAG="_a${A_BITS}"
  [[ "${A_ASYM}" == "1" ]] && ACT_TAG="${ACT_TAG}_aasym"
fi
REQUANT_TAG=""
if [[ "${REQUANT}" == "1" ]]; then
  REQUANT_TAG="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
fi
ROTATE_TAG=""
if [[ "${ROTATE}" == "1" ]]; then
  ROTATE_TAG="_rot"
  [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]] && ROTATE_TAG="${ROTATE_TAG}_optrot"
fi
SAVE_QMODEL_PATH="${OUTPUT_ROOT}/${MODEL_NAME}/${EXP}/rtn_dp_w4${ACT_TAG}_ns${N_SAMPLES}_dp${NPROC}${ROTATE_TAG}${REQUANT_TAG}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29640}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

EXTRA_ARGS=()
if [[ "${DP_SHARD_INPS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_shard_inps)
  echo "[Info] dp_shard_inps=1 (per-rank activation shard; rank0 CPU capture + scatter)"
fi
if [[ "${ROTATE}" == "1" ]]; then
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
if [[ "${A_ASYM}" == "1" ]]; then
  EXTRA_ARGS+=(--a_asym)
fi

echo "[Info] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC} rotate=${ROTATE} requant=${REQUANT}"
echo "[Info] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] activation quant: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM}"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"

LM_EVAL_ARGS=()
if [[ "${LM_EVAL_ENABLE:-0}" == "1" ]]; then
  LM_EVAL_ARGS=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE:-32}")
fi

# Execute the distributed run
${PYTHON_BIN} -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${NPROC}" --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    ./ptq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method rtn --w_bits 4 --w_clip \
    --output_dir "${OUTPUT_ROOT}" \
    --a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    ${OFFLOAD_INPS_FLAG} \
    "${LM_EVAL_ARGS[@]}"
