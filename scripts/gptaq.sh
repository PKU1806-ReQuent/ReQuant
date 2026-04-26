#!/bin/bash
# GPTAQ / GPTAQ+ReQuant launcher. It supports single-GPU and multi-GPU
# data-parallel calibration without exposing "dp" in the script name.
#
# Usage:
#   bash scripts/gptaq.sh [model path] [visible GPU list] [nproc]
# Example:
#   REQUANT=1 ROTATE=1 A_BITS=4 bash scripts/gptaq.sh ./modelzoo/Meta-Llama-3-8B "0,1,2,3" 4

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=${EXP:-gptaq}
DATASET=${DATASET:-wikitext2}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
ALPHA=${ALPHA:-0.25}

W_ASYM=${W_ASYM:-1}
A_BITS=${A_BITS:-4}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}

ROTATE=${ROTATE:-1}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-${REFINE_SWEEPS:-1}}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-${REFINE_CANDIDATES:-2}}
REQUANT_BETA=${REQUANT_BETA:-${REFINE_BETA:-0.3}}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-${REFINE_MIN_GAIN_EPS:-0.003}}
REQUANT_ANCHOR_LAMBDA=${REQUANT_ANCHOR_LAMBDA:-${REFINE_ANCHOR_LAMBDA:-}}

ACT_TAG=""
if [[ "${A_BITS}" != "16" ]]; then
  ACT_TAG="_a${A_BITS}"
  [[ "${A_ASYM}" == "1" ]] && ACT_TAG="${ACT_TAG}_aasym"
fi
ROTATE_TAG=""
if [[ "${ROTATE}" == "1" ]]; then
  ROTATE_TAG="_rot"
  [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]] && ROTATE_TAG="${ROTATE_TAG}_optrot"
fi
REQUANT_TAG=""
if [[ "${REQUANT}" == "1" ]]; then
  REQUANT_TAG="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
fi

SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/gptaq_w4${ACT_TAG}_ns${N_SAMPLES}_a${ALPHA}_n${NPROC}${ROTATE_TAG}${REQUANT_TAG}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29620}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

echo "[Info] ranks=${NPROC} GPUs=${GPU_LIST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs in logs/)"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"
echo "[Info] dataset=${DATASET} nsamples=${N_SAMPLES} seq_len=${SEQ_LEN}"
echo "[Info] weight method=gptaq w_asym=${W_ASYM} requant=${REQUANT} rotate=${ROTATE}"
echo "[Info] activation quant: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM}"
echo "[Info] python=${PYTHON_BIN}"

EXTRA_ARGS=()
if [[ "${DP_FASTERQUANT_MASTER_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_fasterquant_master_only)
  echo "[Info] dp_fasterquant_master_only=1 (H reduce→rank0; fasterquant+broadcast weights)"
fi
if [[ "${DP_SHARD_INPS:-0}" == "1" ]]; then
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
if [[ "${ENABLE_AQ_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_aq_calibration)
else
  EXTRA_ARGS+=(--no-enable_aq_calibration)
fi
if [[ "${W_ASYM}" == "1" ]]; then
  EXTRA_ARGS+=(--w_asym)
fi
if [[ "${A_ASYM}" == "1" ]]; then
  EXTRA_ARGS+=(--a_asym)
fi
if [[ "${REQUANT}" == "1" ]]; then
  EXTRA_ARGS+=(
    --enable_requant
    --requant_phase2_sweeps "${REQUANT_SWEEPS}"
    --requant_phase2_candidates "${REQUANT_CANDIDATES}"
    --requant_beta "${REQUANT_BETA}"
    --requant_min_gain_eps "${REQUANT_MIN_GAIN_EPS}"
  )
  if [[ -n "${REQUANT_ANCHOR_LAMBDA:-}" ]]; then
    EXTRA_ARGS+=(--requant_anchor_lambda "${REQUANT_ANCHOR_LAMBDA}")
  fi
  echo "[Info] requant sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} beta=${REQUANT_BETA} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
fi

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset "${DATASET}" --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method gptaq --w_bits 4 --w_clip --act_order \
    --a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}" \
    --alpha "${ALPHA}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps
