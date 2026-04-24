#!/bin/bash
# Sweep GPTAQ-Refined Phase-2 beta with fixed refine_sweeps=2.
#
# Usage:
#   bash scripts/sweep_refine_beta.sh [模型路径] [可见 GPU 列表] [每节点进程数]
# Example:
#   bash scripts/sweep_refine_beta.sh ./modelzoo/Qwen3-4B "0,1,2,3" 4

set -u

MODEL_PATH=${1:-./modelzoo/Meta-Llama-3-8B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

# Default beta list; can be overridden by env var, e.g.:
#   BETA_LIST="0 0.1 0.2" bash scripts/sweep_refine_beta.sh ...
BETA_LIST=${BETA_LIST:-"0.35 0.4 0.450.5"}

# Keep sweeps fixed as requested.
REFINE_SWEEPS_FIXED=2

RUN_TAG=${RUN_TAG:-"_rbeta_sweep_$(date +%y%m%d_%H%M%S)"}

echo "[Info] model=${MODEL_PATH} gpus=${GPU_LIST} nproc=${NPROC}"
echo "[Info] refine_sweeps=${REFINE_SWEEPS_FIXED} beta_list=${BETA_LIST}"
echo "[Info] run_tag=${RUN_TAG}"

FAILED_BETAS=()

for beta in ${BETA_LIST}; do
  beta_tag=${beta//./p}
  ckpt_tag="${RUN_TAG}_b${beta_tag}"
  echo
  echo "=============================="
  echo "[Run] REFINE_BETA=${beta} GPTAQ_DP_CKPT_TAG=${ckpt_tag}"
  echo "=============================="

  if ! REFINE_SWEEPS=${REFINE_SWEEPS_FIXED} \
       REFINE_BETA=${beta} \
       GPTAQ_DP_CKPT_TAG=${ckpt_tag} \
       bash scripts/gptq_refined_dp.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}"; then
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

