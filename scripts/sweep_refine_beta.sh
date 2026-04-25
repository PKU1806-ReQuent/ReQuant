#!/bin/bash
# Sweep GPTAQ+ReQuant Phase-2 beta with fixed requant sweeps=2.
#
# Usage:
#   bash scripts/sweep_refine_beta.sh [model path] [visible GPU list] [nproc]
# Example:
#   bash scripts/sweep_refine_beta.sh ./modelzoo/Qwen3-4B "0,1,2,3" 4

set -u

MODEL_PATH=${1:-./modelzoo/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

# Defaults to W4A16. Set A_BITS=4 and optional A_* vars before running for activation quantization.
A_BITS=${A_BITS:-16}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-0}
export A_BITS ENABLE_AQ_CALIBRATION

# Defaults to beta=0.3. Set BETA_LIST="0.2 0.3 0.4" to sweep more values.
BETA_LIST=${BETA_LIST:-"0.3"}

# Keep sweeps fixed as requested.
REQUANT_SWEEPS_FIXED=2

RUN_TAG=${RUN_TAG:-"_rbeta_sweep_$(date +%y%m%d_%H%M%S)"}

echo "[Info] model=${MODEL_PATH} gpus=${GPU_LIST} nproc=${NPROC}"
echo "[Info] activation_quant=off (A_BITS=${A_BITS}) enable_aq_calibration=${ENABLE_AQ_CALIBRATION}"
echo "[Info] requant_sweeps=${REQUANT_SWEEPS_FIXED} beta_list=${BETA_LIST}"
echo "[Info] run_tag=${RUN_TAG}"

FAILED_BETAS=()

for beta in ${BETA_LIST}; do
  beta_tag=${beta//./p}
  ckpt_tag="${RUN_TAG}_b${beta_tag}"
  echo
  echo "=============================="
  echo "[Run] REQUANT_BETA=${beta} GPTAQ_DP_CKPT_TAG=${ckpt_tag}"
  echo "=============================="

  if ! REQUANT=1 \
       REQUANT_SWEEPS=${REQUANT_SWEEPS_FIXED} \
       REQUANT_BETA=${beta} \
       GPTAQ_DP_CKPT_TAG=${ckpt_tag} \
       bash scripts/gptq.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"; then
    echo "[Warn] beta=${beta} failed"
    FAILED_BETAS+=("${beta}")
  fi
done

echo
echo "========== Sweep Done =========="
if [ ${#FAILED_BETAS[@]} -eq 0 ]; then
  echo "[Info] All beta runs finished successfully."
else
  echo "[Warn] Failed betas: ${FAILED_BETAS[*]}"
fi

