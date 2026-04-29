#!/bin/bash
# Sweep REQUANT_MIN_GAIN_EPS for GPTAQ W4A4 + Rotate + ReQuant.
#
# Usage:
#   bash scripts/sweep_gptaq_requant_eps.sh [model path] [visible GPU list] [nproc]
# Example:
#   EPS_LIST="0.2 0.5 1 2 4" \
#   bash scripts/sweep_gptaq_requant_eps.sh ./modelzoo/Llama3/Meta-Llama-3-8B "0,1,2,3" 4

set -u

MODEL_PATH=${1:-./modelzoo/Llama3/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

# Coarse default sweep around the current threshold formula:
# accept if best_gain < -eps * L_row / d.
EPS_LIST=${EPS_LIST:-"0.1 0.2 0.5 1 2"}

# Fixed GPTAQ W4A4 + Rotate + ReQuant settings.
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
REQUANT=${REQUANT:-1}
REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
REQUANT_BETA=${REQUANT_BETA:-0.3}

# Put each eps in a separate experiment directory to avoid overwriting the
# checkpoint, because gptaq.sh filenames do not include REQUANT_MIN_GAIN_EPS.
BASE_EXP=${BASE_EXP:-gptaq_eps_sweep}
RUN_TAG=${RUN_TAG:-"$(date +%y%m%d_%H%M%S)"}

export DATASET N_SAMPLES SEQ_LEN ALPHA W_ASYM
export A_BITS A_CLIP_RATIO A_ASYM ENABLE_AQ_CALIBRATION
export ROTATE REQUANT REQUANT_SWEEPS REQUANT_CANDIDATES REQUANT_BETA

echo "[Info] model=${MODEL_PATH} gpus=${GPU_LIST} nproc=${NPROC}"
echo "[Info] GPTAQ W4A4 rotate=${ROTATE} requant=${REQUANT} sweeps=${REQUANT_SWEEPS} candidates=${REQUANT_CANDIDATES} beta=${REQUANT_BETA}"
echo "[Info] activation: A_BITS=${A_BITS} A_CLIP_RATIO=${A_CLIP_RATIO} A_ASYM=${A_ASYM} enable_aq_calibration=${ENABLE_AQ_CALIBRATION}"
echo "[Info] eps_list=${EPS_LIST}"
echo "[Info] base_exp=${BASE_EXP} run_tag=${RUN_TAG}"

FAILED_EPS=()

for eps in ${EPS_LIST}; do
  eps_tag=${eps//./p}
  eps_tag=${eps_tag//-/m}
  exp="${BASE_EXP}_${RUN_TAG}_eps${eps_tag}"

  echo
  echo "=============================="
  echo "[Run] REQUANT_MIN_GAIN_EPS=${eps} EXP=${exp}"
  echo "=============================="

  if ! EXP="${exp}" \
       REQUANT_MIN_GAIN_EPS="${eps}" \
       bash scripts/gptaq.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"; then
    echo "[Warn] eps=${eps} failed"
    FAILED_EPS+=("${eps}")
  fi
done

echo
echo "========== Sweep Done =========="
if [ ${#FAILED_EPS[@]} -eq 0 ]; then
  echo "[Info] All eps runs finished successfully."
else
  echo "[Warn] Failed eps values: ${FAILED_EPS[*]}"
fi
