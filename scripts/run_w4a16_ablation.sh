#!/bin/bash
# Run W4A16 / W4A4 ablations.
#
# Method support:
#   - RTN: supported via scripts/rtn.sh.
#   - GPTQ: supported via scripts/gptq.sh.
#   - GPTAQ: supported via scripts/gptaq.sh.
#   - AWQ: supported via scripts/awq.sh.
#
# Precision:
#   - Default: W4A4 (A_BITS=4, A_ASYM=1, A_CLIP_RATIO=0.9).
#   - W4A16: set A_BITS=16.
#
# Select runs:
#   Add / uncomment run_* lines at the bottom of this file. The helper function
#   identifies the method from the function name / method argument and builds
#   the corresponding checkpoint path automatically.
#
# Included runs:
#   - RTN: RTN, RTN + ReQuant, RTN + Rotate, RTN + Rotate + ReQuant.
#   - GPTQ: GPTQ + ReQuant, GPTQ + Rotate + ReQuant.
#   - GPTAQ: GPTAQ + ReQuant, GPTAQ + Rotate + ReQuant.
#   - AWQ: AWQ, AWQ + ReQuant, AWQ + Rotate, AWQ + Rotate + ReQuant.
#
# Usage:
#   bash scripts/run_w4a16_ablation.sh [model path] [GPU list] [nproc]
#
# Example:
#   bash scripts/run_w4a16_ablation.sh ./modelzoo/Llama3/Meta-Llama-3-8B "3,4,5,6" 4

set -euo pipefail

MODEL_PATH=${1:-./modelzoo/Llama3/Meta-Llama-3-8B}
GPU_LIST=${2:-"3,4,5,6"}
NPROC=${3:-4}

N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
A_BITS=${A_BITS:-4}
A_ASYM=${A_ASYM:-1}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
ALPHA=${ALPHA:-0.25}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_BETA=${REQUANT_BETA:-0.25}
GPTQ_REQUANT_BETA=${GPTQ_REQUANT_BETA:-0.25}
GPTAQ_REQUANT_BETA=${GPTAQ_REQUANT_BETA:-0.25}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}
AWQ_NSAMPLES=${AWQ_NSAMPLES:-128}
AWQ_GRID=${AWQ_GRID:-20}
AWQ_MIN_ALPHA=${AWQ_MIN_ALPHA:-0.0}
AWQ_MAX_ALPHA=${AWQ_MAX_ALPHA:-1.0}
LM_EVAL=${LM_EVAL:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}
EVAL_AFTER_RUN=${EVAL_AFTER_RUN:-0}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

ACT_TAG=""
RUN_PRECISION_TAG="W4A16"
if [[ "${A_BITS}" != "16" ]]; then
  ACT_TAG="_a${A_BITS}"
  [[ "${A_ASYM}" == "1" ]] && ACT_TAG="${ACT_TAG}_aasym"
  RUN_PRECISION_TAG="W4A${A_BITS}"
fi

run_one() {
  local name="$1"
  local rotate="$2"
  local requant="$3"
  local rotate_tag=""
  local requant_tag=""
  local qmodel_path=""

  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  if [[ "${requant}" == "1" ]]; then
    requant_tag="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
  fi
  qmodel_path="./outputs/$(basename "${MODEL_PATH}")/rtn/rtn_w4${ACT_TAG}_ns${N_SAMPLES}${rotate_tag}${requant_tag}.pt"

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] N_SAMPLES=${N_SAMPLES} SEQ_LEN=${SEQ_LEN} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=${requant} REQUANT_BETA=${REQUANT_BETA}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="rtn" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ROTATE="${rotate}" \
  REQUANT="${requant}" \
  REQUANT_SWEEPS="${REQUANT_SWEEPS}" \
  REQUANT_CANDIDATES="${REQUANT_CANDIDATES}" \
  REQUANT_BETA="${REQUANT_BETA}" \
  REQUANT_MIN_GAIN_EPS="${REQUANT_MIN_GAIN_EPS}" \
  LM_EVAL="${LM_EVAL}" \
  LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
  bash scripts/rtn.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"

  if [[ "${EVAL_AFTER_RUN}" == "1" ]]; then
    echo
    echo "------------------------------------------------------------"
    echo "[Eval] ${name}"
    echo "[Eval] qmodel=${qmodel_path}"
    echo "------------------------------------------------------------"
    HF_ENDPOINT="${HF_ENDPOINT}" \
    LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
    bash scripts/eval_lm_tasks.sh "${MODEL_PATH}" "${qmodel_path}" "${GPU_LIST}"
  fi
}

run_gpt_one() {
  local method="$1"
  local name="$2"
  local rotate="$3"
  local rotate_tag=""
  local qmodel_path=""
  local launcher=""
  local method_beta=""

  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  launcher="scripts/${method}.sh"
  if [[ "${method}" == "gptaq" ]]; then
    method_beta="${GPTAQ_REQUANT_BETA}"
  else
    method_beta="${GPTQ_REQUANT_BETA}"
  fi
  if [[ "${method}" == "gptaq" ]]; then
    qmodel_path="./outputs/$(basename "${MODEL_PATH}")/${method}/${method}_w4${ACT_TAG}_ns${N_SAMPLES}_a${ALPHA}${rotate_tag}_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}.pt"
  else
    qmodel_path="./outputs/$(basename "${MODEL_PATH}")/${method}/${method}_w4${ACT_TAG}_ns${N_SAMPLES}${rotate_tag}_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}.pt"
  fi

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] method=${method} N_SAMPLES=${N_SAMPLES} SEQ_LEN=${SEQ_LEN} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=1 REQUANT_BETA=${method_beta} REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="${method}" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  ALPHA="${ALPHA}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ROTATE="${rotate}" \
  REQUANT=1 \
  REQUANT_SWEEPS="${REQUANT_SWEEPS}" \
  REQUANT_CANDIDATES="${REQUANT_CANDIDATES}" \
  REQUANT_BETA="${method_beta}" \
  REQUANT_MIN_GAIN_EPS="${REQUANT_MIN_GAIN_EPS}" \
  LM_EVAL="${LM_EVAL}" \
  LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
  bash "${launcher}" "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"

  if [[ "${EVAL_AFTER_RUN}" == "1" ]]; then
    echo
    echo "------------------------------------------------------------"
    echo "[Eval] ${name}"
    echo "[Eval] qmodel=${qmodel_path}"
    echo "------------------------------------------------------------"
    HF_ENDPOINT="${HF_ENDPOINT}" \
    LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
    bash scripts/eval_lm_tasks.sh "${MODEL_PATH}" "${qmodel_path}" "${GPU_LIST}"
  fi
}

run_awq_one() {
  local name="$1"
  local rotate="$2"
  local requant="$3"
  local rotate_tag=""
  local requant_tag=""
  local qmodel_path=""

  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  if [[ "${requant}" == "1" ]]; then
    requant_tag="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
  fi
  qmodel_path="./outputs/$(basename "${MODEL_PATH}")/awq_dp/awq_w4${ACT_TAG}_ns${N_SAMPLES}_g${AWQ_GRID}_a${AWQ_MIN_ALPHA}-${AWQ_MAX_ALPHA}${rotate_tag}${requant_tag}.pt"

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] method=awq N_SAMPLES=${N_SAMPLES} AWQ_NSAMPLES=${AWQ_NSAMPLES} SEQ_LEN=${SEQ_LEN} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=${requant} REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="awq_dp" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ROTATE="${rotate}" \
  REQUANT="${requant}" \
  REQUANT_SWEEPS="${REQUANT_SWEEPS}" \
  REQUANT_CANDIDATES="${REQUANT_CANDIDATES}" \
  REQUANT_MIN_GAIN_EPS="${REQUANT_MIN_GAIN_EPS}" \
  AWQ_NSAMPLES="${AWQ_NSAMPLES}" \
  AWQ_GRID="${AWQ_GRID}" \
  AWQ_MIN_ALPHA="${AWQ_MIN_ALPHA}" \
  AWQ_MAX_ALPHA="${AWQ_MAX_ALPHA}" \
  LM_EVAL="${LM_EVAL}" \
  LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
  bash scripts/awq.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"

  if [[ "${EVAL_AFTER_RUN}" == "1" ]]; then
    echo
    echo "------------------------------------------------------------"
    echo "[Eval] ${name}"
    echo "[Eval] qmodel=${qmodel_path}"
    echo "------------------------------------------------------------"
    HF_ENDPOINT="${HF_ENDPOINT}" \
    LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE}" \
    bash scripts/eval_lm_tasks.sh "${MODEL_PATH}" "${qmodel_path}" "${GPU_LIST}"
  fi
}

# ---------------------------------------------------------------------------
# Select experiments here.
#
# Signature:
#   run_one     "<name>" <rotate:0|1> <requant:0|1>   # RTN
#   run_gpt_one <gptq|gptaq> "<name>" <rotate:0|1>
#   run_awq_one "<name>" <rotate:0|1> <requant:0|1>
#
# The script automatically uses A_BITS to label/save W4A16 or W4A4 outputs.
# Inputs remain simple: model path, GPU list, and nproc.
# ---------------------------------------------------------------------------

# RTN examples:
# run_one "RTN ${RUN_PRECISION_TAG}" 0 0
# run_one "RTN + ReQuant ${RUN_PRECISION_TAG}" 0 1
# run_one "RTN + Rotate ${RUN_PRECISION_TAG}" 1 0
# run_one "RTN + Rotate + ReQuant ${RUN_PRECISION_TAG}" 1 1

# Default active runs:
run_gpt_one "gptq" "GPTQ + ReQuant ${RUN_PRECISION_TAG}" 0
run_gpt_one "gptaq" "GPTAQ + ReQuant ${RUN_PRECISION_TAG}" 0
run_gpt_one "gptq" "GPTQ + Rotate + ReQuant ${RUN_PRECISION_TAG}" 1
run_gpt_one "gptaq" "GPTAQ + Rotate + ReQuant ${RUN_PRECISION_TAG}" 1

# AWQ examples:
# run_awq_one "AWQ ${RUN_PRECISION_TAG}" 0 0
# run_awq_one "AWQ + ReQuant ${RUN_PRECISION_TAG}" 0 1
# run_awq_one "AWQ + Rotate ${RUN_PRECISION_TAG}" 1 0
# run_awq_one "AWQ + Rotate + ReQuant ${RUN_PRECISION_TAG}" 1 1

# Example runs:
# bash scripts/run_w4a16_ablation.sh \
#   ./modelzoo/Llama3/Meta-Llama-3-8B \
#   "0,1,2,3" \
#   4