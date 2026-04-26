#!/bin/bash

# Input arguments
MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
DEVICE=${2:-0}

MODEL_NAME=$(basename "${MODEL_PATH}")
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs}"
EXP=${EXP:-rtn}
N_SAMPLES=${N_SAMPLES:-1024}
SEQ_LEN=${SEQ_LEN:-2048}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-1}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.003}
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
SAVE_QMODEL_PATH="${OUTPUT_ROOT}/${MODEL_NAME}/${EXP}/rtn_w4_ns${N_SAMPLES}${ROTATE_TAG}${REQUANT_TAG}.pt"

# Set environment variables
export CUDA_VISIBLE_DEVICES=${DEVICE}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

EXTRA_ARGS=()
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

echo "[Info] model=${MODEL_PATH} device=${DEVICE} rotate=${ROTATE} requant=${REQUANT}"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"

LM_EVAL_ARGS=()
if [[ "${LM_EVAL_ENABLE:-0}" == "1" ]]; then
  LM_EVAL_ARGS=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE:-32}")
fi

# Execute the distributed run
${PYTHON_BIN} -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:2940${DEVICE} ./ptq.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method rtn --w_bits 4 --w_clip \
    --output_dir "${OUTPUT_ROOT}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    "${LM_EVAL_ARGS[@]}"
