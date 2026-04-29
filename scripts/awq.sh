#!/bin/bash
# AWQ launcher with data-parallel activation statistics and layer-wise scale search.
#
# Usage:
#   bash scripts/awq.sh [model path] [visible GPU list] [nproc]
# Example:
#   REQUANT=1 ROTATE=1 A_BITS=4 bash scripts/awq.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4

MODEL_PATH=${1:-./modelzoo/Llama3/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=${EXP:-awq_dp}
DATASET=${DATASET:-wikitext2}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}

AWQ_NSAMPLES=${AWQ_NSAMPLES:-128}
AWQ_GRID=${AWQ_GRID:-20}
AWQ_MIN_ALPHA=${AWQ_MIN_ALPHA:-0.0}
# Use the semi-open formulation ratio = min + i/grid * (max - min), i in [0, grid),
# so MAX_ALPHA=1.0, MIN_ALPHA=0.0, GRID=20 -> {0, 0.05, ..., 0.95} (matches MIT llm-awq).
AWQ_MAX_ALPHA=${AWQ_MAX_ALPHA:-1.0}
AWQ_AUTO_CLIP=${AWQ_AUTO_CLIP:-1}
AWQ_CLIP_N_GRID=${AWQ_CLIP_N_GRID:-20}
AWQ_CLIP_MAX_SHRINK=${AWQ_CLIP_MAX_SHRINK:-0.5}
AWQ_CLIP_N_SAMPLE_TOKEN=${AWQ_CLIP_N_SAMPLE_TOKEN:-512}
AWQ_FC_FC_SCALE=${AWQ_FC_FC_SCALE:-0}

W_ASYM=${W_ASYM:-1}
A_BITS=${A_BITS:-4}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}

ROTATE=${ROTATE:-1}
REQUANT=${REQUANT:-0}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}

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
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    ROTATE_TAG="${ROTATE_TAG}_optrot"
  fi
fi
SAVE_QMODEL_PATH="./outputs/${MODEL_NAME}/${EXP}/awq_w4${ACT_TAG}_ns${N_SAMPLES}_g${AWQ_GRID}_a${AWQ_MIN_ALPHA}-${AWQ_MAX_ALPHA}${ROTATE_TAG}${REQUANT_TAG}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29630}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
LM_EVAL=${LM_EVAL:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}

echo "[Info] DP ranks=${NPROC} GPUs=${GPU_LIST} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs in logs/)"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"
echo "[Info] dataset=${DATASET} nsamples=${N_SAMPLES} seq_len=${SEQ_LEN}"
echo "[Info] weight method=awq w_asym=${W_ASYM} requant=${REQUANT} rotate=${ROTATE}"
echo "[Info] activation quant: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM}"
echo "[Info] awq: grid=${AWQ_GRID} alpha=[${AWQ_MIN_ALPHA}, ${AWQ_MAX_ALPHA}) auto_clip=${AWQ_AUTO_CLIP} fc_fc_scale=${AWQ_FC_FC_SCALE}"
echo "[Info] awq calibration: awq_nsamples=${AWQ_NSAMPLES} (independent from N_SAMPLES=${N_SAMPLES})"
echo "[Info] lm_eval=${LM_EVAL} lm_eval_batch_size=${LM_EVAL_BATCH_SIZE}"
echo "[Info] python=${PYTHON_BIN}"

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
    --requant_sweeps "${REQUANT_SWEEPS}"
    --requant_candidates "${REQUANT_CANDIDATES}"
    --requant_min_gain_eps "${REQUANT_MIN_GAIN_EPS}"
  )
  echo "[Info] requant sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
fi
if [[ "${LM_EVAL}" == "1" ]]; then
  EXTRA_ARGS+=(--lm_eval --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}")
fi

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset "${DATASET}" --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method awq --w_bits 4 \
    --a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}" \
    --awq_nsamples "${AWQ_NSAMPLES}" \
    --awq_grid "${AWQ_GRID}" \
    --awq_min_alpha "${AWQ_MIN_ALPHA}" \
    --awq_max_alpha "${AWQ_MAX_ALPHA}" \
    --awq_auto_clip "${AWQ_AUTO_CLIP}" \
    --awq_clip_n_grid "${AWQ_CLIP_N_GRID}" \
    --awq_clip_max_shrink "${AWQ_CLIP_MAX_SHRINK}" \
    --awq_clip_n_sample_token "${AWQ_CLIP_N_SAMPLE_TOKEN}" \
    --awq_fc_fc_scale "${AWQ_FC_FC_SCALE}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps
