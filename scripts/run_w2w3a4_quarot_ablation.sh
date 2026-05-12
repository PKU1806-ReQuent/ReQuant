#!/bin/bash
# Run W2A4 / W3A4 QuaRot ablations.
#
# Method support:
#   - RTN: supported via scripts/rtn.sh.
#   - GPTQ: supported via scripts/gptq.sh.
#   - GPTAQ: supported via scripts/gptaq.sh.
#   - AWQ: supported via scripts/awq.sh.
#
# Included runs:
#   - GPTAQ/GPTQ/RTN/AWQ with QuaRot and QuaRot + ReQuant.
#   - Default precision set: W2A4 and W3A4 on Llama3 8B.
#
# Usage:
#   bash scripts/run_w2w3a4_quarot_ablation.sh [model path] [GPU list] [nproc]
#
# Example:
#   bash scripts/run_w2w3a4_quarot_ablation.sh ./modelzoo/Meta-Llama-3-8B "0,1,2,3" 4

set -euo pipefail

MODEL_PATH=${1:-./modelzoo/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

W_BITS_LIST=${W_BITS_LIST:-"2 3"}
A_BITS=${A_BITS:-4}
A_ASYM=${A_ASYM:-1}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
W_ASYM=${W_ASYM:-1}
W_CLIP=${W_CLIP:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}
DATASET=${DATASET:-wikitext2}
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
ALPHA=${ALPHA:-0.25}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_BETA=${REQUANT_BETA:-0.25}
GPTQ_REQUANT_BETA=${GPTQ_REQUANT_BETA:-0.25}
GPTAQ_REQUANT_BETA=${GPTAQ_REQUANT_BETA:-0.25}
AWQ_REQUANT_BETA=${AWQ_REQUANT_BETA:-0.25}
RTN_REQUANT_BETA=${RTN_REQUANT_BETA:-0.25}
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

act_tag_for() {
  local a_bits="$1"
  local act_tag=""
  if [[ "${a_bits}" != "16" ]]; then
    act_tag="_a${a_bits}"
    [[ "${A_ASYM}" == "1" ]] && act_tag="${act_tag}_aasym"
  fi
  echo "${act_tag}"
}

run_one() {
  local name="$1"
  local w_bits="$2"
  local rotate="$3"
  local requant="$4"
  local act_tag=""
  local rotate_tag=""
  local requant_tag=""
  local qmodel_path=""

  act_tag=$(act_tag_for "${A_BITS}")
  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  if [[ "${requant}" == "1" ]]; then
    requant_tag="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
  fi
  qmodel_path="./outputs/$(basename "${MODEL_PATH}")/rtn/rtn_w${w_bits}${act_tag}_ns${N_SAMPLES}${rotate_tag}${requant_tag}.pt"

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] W_BITS=${w_bits} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=${requant} REQUANT_BETA=${RTN_REQUANT_BETA}"
  echo "[Config] N_SAMPLES=${N_SAMPLES} SEQ_LEN=${SEQ_LEN} DATASET=${DATASET}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="rtn" \
  DATASET="${DATASET}" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  W_BITS="${w_bits}" \
  W_ASYM="${W_ASYM}" \
  W_CLIP="${W_CLIP}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ENABLE_AQ_CALIBRATION="${ENABLE_AQ_CALIBRATION}" \
  ROTATE="${rotate}" \
  REQUANT="${requant}" \
  REQUANT_SWEEPS="${REQUANT_SWEEPS}" \
  REQUANT_CANDIDATES="${REQUANT_CANDIDATES}" \
  REQUANT_BETA="${RTN_REQUANT_BETA}" \
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
  local w_bits="$3"
  local rotate="$4"
  local requant="$5"
  local act_tag=""
  local rotate_tag=""
  local requant_tag=""
  local qmodel_path=""
  local launcher=""
  local method_beta=""

  act_tag=$(act_tag_for "${A_BITS}")
  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  if [[ "${requant}" == "1" ]]; then
    requant_tag="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
  fi
  launcher="scripts/${method}.sh"
  if [[ "${method}" == "gptaq" ]]; then
    method_beta="${GPTAQ_REQUANT_BETA}"
    qmodel_path="./outputs/$(basename "${MODEL_PATH}")/${method}/${method}_w${w_bits}${act_tag}_ns${N_SAMPLES}_a${ALPHA}${rotate_tag}${requant_tag}.pt"
  else
    method_beta="${GPTQ_REQUANT_BETA}"
    qmodel_path="./outputs/$(basename "${MODEL_PATH}")/${method}/${method}_w${w_bits}${act_tag}_ns${N_SAMPLES}${rotate_tag}${requant_tag}.pt"
  fi

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] method=${method} W_BITS=${w_bits} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=${requant} REQUANT_BETA=${method_beta}"
  echo "[Config] N_SAMPLES=${N_SAMPLES} SEQ_LEN=${SEQ_LEN} DATASET=${DATASET} REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="${method}" \
  DATASET="${DATASET}" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  ALPHA="${ALPHA}" \
  W_BITS="${w_bits}" \
  W_ASYM="${W_ASYM}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ENABLE_AQ_CALIBRATION="${ENABLE_AQ_CALIBRATION}" \
  ROTATE="${rotate}" \
  REQUANT="${requant}" \
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
  local w_bits="$2"
  local rotate="$3"
  local requant="$4"
  local act_tag=""
  local rotate_tag=""
  local requant_tag=""
  local qmodel_path=""

  act_tag=$(act_tag_for "${A_BITS}")
  if [[ "${rotate}" == "1" ]]; then
    rotate_tag="_rot"
  fi
  if [[ "${requant}" == "1" ]]; then
    requant_tag="_requant_s${REQUANT_SWEEPS}_c${REQUANT_CANDIDATES}"
  fi
  qmodel_path="./outputs/$(basename "${MODEL_PATH}")/awq_dp/awq_w${w_bits}${act_tag}_ns${N_SAMPLES}_g${AWQ_GRID}_a${AWQ_MIN_ALPHA}-${AWQ_MAX_ALPHA}${rotate_tag}${requant_tag}.pt"

  echo
  echo "============================================================"
  echo "[Run] ${name}"
  echo "[Config] model=${MODEL_PATH} GPUs=${GPU_LIST} nproc=${NPROC}"
  echo "[Config] method=awq W_BITS=${w_bits} A_BITS=${A_BITS} ROTATE=${rotate} REQUANT=${requant} REQUANT_BETA=${AWQ_REQUANT_BETA}"
  echo "[Config] N_SAMPLES=${N_SAMPLES} AWQ_NSAMPLES=${AWQ_NSAMPLES} SEQ_LEN=${SEQ_LEN} DATASET=${DATASET}"
  echo "[Config] qmodel=${qmodel_path}"
  echo "============================================================"

  EXP="awq_dp" \
  DATASET="${DATASET}" \
  N_SAMPLES="${N_SAMPLES}" \
  SEQ_LEN="${SEQ_LEN}" \
  W_BITS="${w_bits}" \
  W_ASYM="${W_ASYM}" \
  A_BITS="${A_BITS}" \
  A_ASYM="${A_ASYM}" \
  A_CLIP_RATIO="${A_CLIP_RATIO}" \
  ENABLE_AQ_CALIBRATION="${ENABLE_AQ_CALIBRATION}" \
  ROTATE="${rotate}" \
  REQUANT="${requant}" \
  REQUANT_SWEEPS="${REQUANT_SWEEPS}" \
  REQUANT_CANDIDATES="${REQUANT_CANDIDATES}" \
  REQUANT_BETA="${AWQ_REQUANT_BETA}" \
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
#   run_one     "<name>" <w_bits> <rotate:0|1> <requant:0|1>   # RTN
#   run_gpt_one <gptq|gptaq> "<name>" <w_bits> <rotate:0|1> <requant:0|1>
#   run_awq_one "<name>" <w_bits> <rotate:0|1> <requant:0|1>
#
# Defaults run W2A4 and W3A4 for QuaRot / QuaRot + ReQuant only.
# ---------------------------------------------------------------------------

for W_BITS in ${W_BITS_LIST}; do
  RUN_PRECISION_TAG="W${W_BITS}A${A_BITS}"

  run_gpt_one "gptaq" "GPTAQ + QuaRot ${RUN_PRECISION_TAG}" "${W_BITS}" 1 0
  run_gpt_one "gptaq" "GPTAQ + QuaRot + ReQuant ${RUN_PRECISION_TAG}" "${W_BITS}" 1 1

  run_gpt_one "gptq" "GPTQ + QuaRot ${RUN_PRECISION_TAG}" "${W_BITS}" 1 0
  run_gpt_one "gptq" "GPTQ + QuaRot + ReQuant ${RUN_PRECISION_TAG}" "${W_BITS}" 1 1

  run_one "RTN + QuaRot ${RUN_PRECISION_TAG}" "${W_BITS}" 1 0
  run_one "RTN + QuaRot + ReQuant ${RUN_PRECISION_TAG}" "${W_BITS}" 1 1

  run_awq_one "AWQ + QuaRot ${RUN_PRECISION_TAG}" "${W_BITS}" 1 0
  run_awq_one "AWQ + QuaRot + ReQuant ${RUN_PRECISION_TAG}" "${W_BITS}" 1 1
done

# Example runs:
# bash scripts/run_w2w3a4_quarot_ablation.sh \
#   ./modelzoo/Meta-Llama-3-8B \
#   "0,1,2,3" \
#   4
