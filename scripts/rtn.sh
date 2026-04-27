#!/bin/bash

# Input arguments
MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=${EXP:-rtn}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
A_BITS=${A_BITS:-16}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-1}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_BETA=${REQUANT_BETA:-0.3}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}
ROTATE=${ROTATE:-0}
REQUANT_TAG=""
if [[ "${REQUANT}" == "1" ]]; then
  REQUANT_TAG="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
fi
ROTATE_TAG=""
if [[ "${ROTATE}" == "1" ]]; then
  ROTATE_TAG="_rot"
  [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]] && ROTATE_TAG="${ROTATE_TAG}_optrot"
fi
ACT_TAG=""
if [[ "${A_BITS}" != "16" ]]; then
  ACT_TAG="_a${A_BITS}"
  [[ "${A_ASYM}" == "1" ]] && ACT_TAG="${ACT_TAG}_aasym"
fi
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/rtn_w4${ACT_TAG}_ns${N_SAMPLES}_n${NPROC}${ROTATE_TAG}${REQUANT_TAG}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29640}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "[Info] ranks=${NPROC} GPUs=${GPU_LIST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

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
    --requant_beta "${REQUANT_BETA}"
    --requant_min_gain_eps "${REQUANT_MIN_GAIN_EPS}"
  )
  echo "[Info] requant sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} beta=${REQUANT_BETA} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
fi
if [[ "${ENABLE_AQ_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_aq_calibration)
else
  EXTRA_ARGS+=(--no-enable_aq_calibration)
fi
if [[ "${A_ASYM}" == "1" ]]; then
  EXTRA_ARGS+=(--a_asym)
fi

echo "[Info] model=${MODEL_PATH} rotate=${ROTATE} requant=${REQUANT}"
echo "[Info] activation quant: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM}"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"

# Execute the distributed run
${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method rtn --w_bits 4 --w_clip \
    --a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps
