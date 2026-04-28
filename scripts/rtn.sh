#!/bin/bash

# Input arguments
MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
DEVICE=${2:-0}
NPROC=${3:-1}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=${EXP:-rtn}
N_SAMPLES=${N_SAMPLES:-1024}
SEQ_LEN=${SEQ_LEN:-2048}
A_BITS=${A_BITS:-16}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}
ROTATE=${ROTATE:-0}
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
DP_TAG=""
ENTRYPOINT="./ptq.py"
if [[ "${NPROC}" -gt 1 ]]; then
  DP_TAG="_dp${NPROC}"
  ENTRYPOINT="./ptq_rtn_dp.py"
fi
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/rtn_w4${ACT_TAG}_ns${N_SAMPLES}${ROTATE_TAG}${REQUANT_TAG}${DP_TAG}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MASTER_PORT="${MASTER_PORT:-29640}"

EXTRA_ARGS=()
if [[ "${ROTATE}" == "1" ]]; then
  EXTRA_ARGS+=(--rotate)
  echo "[Info] rotate=1 (SpinQuant-style hadamard rotation)"
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    EXTRA_ARGS+=(--optimized_rotation_path "${OPTIMIZED_ROTATION_PATH}")
    echo "[Info] optimized_rotation_path=${OPTIMIZED_ROTATION_PATH}"
  fi
fi
if [[ "${A_BITS}" != "16" ]]; then
  EXTRA_ARGS+=(--a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}")
  [[ "${A_ASYM}" == "1" ]] && EXTRA_ARGS+=(--a_asym)
  echo "[Info] activation quant: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM}"
fi
if [[ "${ENABLE_AQ_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_aq_calibration)
else
  EXTRA_ARGS+=(--no-enable_aq_calibration)
fi
echo "[Info] enable_aq_calibration=${ENABLE_AQ_CALIBRATION}"
if [[ "${REQUANT}" == "1" ]]; then
  EXTRA_ARGS+=(
    --requant_sweeps "${REQUANT_SWEEPS}"
    --requant_candidates "${REQUANT_CANDIDATES}"
    --requant_min_gain_eps "${REQUANT_MIN_GAIN_EPS}"
  )
  echo "[Info] requant sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
fi

echo "[Info] model=${MODEL_PATH} device=${DEVICE} nproc=${NPROC} rotate=${ROTATE} requant=${REQUANT} a_bits=${A_BITS}"
echo "[Info] entrypoint=${ENTRYPOINT} MASTER_PORT=${MASTER_PORT}"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"

# Execute the distributed run
${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" "${ENTRYPOINT}" \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method rtn --w_bits 4 --w_clip \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    --lm_eval --lm_eval_batch_size 32
