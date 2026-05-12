#!/bin/bash
# Sweep RTN ReQuant sweeps and compare PPL / optional lm-eval accuracy.
#
# Usage:
#   bash scripts/sweep_rtn_requant_sweeps.sh [model path] [visible GPU list] [nproc]
#
# Examples:
#   SWEEP_LIST="5 6 7 8" BASE_EXP=rtn_rot_sweep_5_8 \
#     bash scripts/sweep_rtn_requant_sweeps.sh ./modelzoo/Meta-Llama-3-8B "0,1,2,3" 4
#
#   SWEEP_LIST="0 1 2 3 4" LM_EVAL=1 LM_EVAL_BATCH_SIZE=8 \
#     bash scripts/sweep_rtn_requant_sweeps.sh ./modelzoo/Llama-2-7b-hf "0,1,2,3" 4

set -u

MODEL_PATH=${1:-./modelzoo/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

# 0 is the RTN baseline without ReQuant. Positive values enable ReQuant and
# set REQUANT_SWEEPS to that value.
SWEEP_LIST=${SWEEP_LIST:-"0 1 2 3 4"}

N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
W_ASYM=${W_ASYM:-1}
W_CLIP=${W_CLIP:-1}
A_BITS=${A_BITS:-16}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}
ROTATE=${ROTATE:-1}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_BETA=${REQUANT_BETA:-0.25}
REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}
LM_EVAL=${LM_EVAL:-0}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-32}

# Each sweep uses a separate EXP so checkpoints/logs are not overwritten.
BASE_EXP=${BASE_EXP:-rtn_sweep}
RUN_TAG=${RUN_TAG:-"$(date +%y%m%d_%H%M%S)"}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export N_SAMPLES SEQ_LEN W_ASYM W_CLIP
export A_BITS A_CLIP_RATIO A_ASYM ENABLE_AQ_CALIBRATION
export ROTATE REQUANT_CANDIDATES REQUANT_BETA REQUANT_MIN_GAIN_EPS
export LM_EVAL LM_EVAL_BATCH_SIZE

echo "[Info] model=${MODEL_PATH} gpus=${GPU_LIST} nproc=${NPROC}"
echo "[Info] RTN sweep_list=${SWEEP_LIST}"
echo "[Info] N_SAMPLES=${N_SAMPLES} SEQ_LEN=${SEQ_LEN} W_CLIP=${W_CLIP} W_ASYM=${W_ASYM}"
echo "[Info] activation: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM} enable_aq_calibration=${ENABLE_AQ_CALIBRATION}"
echo "[Info] rotate=${ROTATE} candidates=${REQUANT_CANDIDATES} beta=${REQUANT_BETA} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"
echo "[Info] lm_eval=${LM_EVAL} lm_eval_batch_size=${LM_EVAL_BATCH_SIZE}"
echo "[Info] base_exp=${BASE_EXP} run_tag=${RUN_TAG}"

FAILED_SWEEPS=()

for sweeps in ${SWEEP_LIST}; do
  sweep_tag=${sweeps//./p}
  sweep_tag=${sweep_tag//-/m}
  exp="${BASE_EXP}_${RUN_TAG}_s${sweep_tag}"
  requant=1
  if [[ "${sweeps}" == "0" ]]; then
    requant=0
  fi

  echo
  echo "=============================="
  echo "[Run] RTN REQUANT_SWEEPS=${sweeps} REQUANT=${requant} EXP=${exp}"
  echo "=============================="

  if ! EXP="${exp}" \
       REQUANT="${requant}" \
       REQUANT_SWEEPS="${sweeps}" \
       bash scripts/rtn.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"; then
    echo "[Warn] sweeps=${sweeps} failed"
    FAILED_SWEEPS+=("${sweeps}")
  fi
done

echo
echo "========== RTN Sweep Done =========="
if [ ${#FAILED_SWEEPS[@]} -eq 0 ]; then
  echo "[Info] All sweep runs finished successfully."
else
  echo "[Warn] Failed sweep values: ${FAILED_SWEEPS[*]}"
fi
echo "[Info] Logs/checkpoints are under outputs/$(basename "${MODEL_PATH}")/${BASE_EXP}_${RUN_TAG}_s*/"
