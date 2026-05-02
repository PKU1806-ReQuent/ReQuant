#!/bin/bash
# Driver: Qwen3-14B W4A4 + QuaRot + ReQuant for 4 methods (GPTAQ, GPTQ, RTN, AWQ).
#
# Runs methods sequentially (each releases all 8 GPUs before the next starts).
# Each method's full stdout+stderr goes to logs_14b_w4a4/0X_<method>_w4a4_rot_requant.log.
# A short summary log aggregates START/END timestamps and return codes.
#
# Recommended usage (long-running, survives SSH drops):
#   tmux new -s requant_w4a4 -d "bash scripts/run_14b_requant_w4a4.sh"
#   tmux attach -t requant_w4a4

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

# --- Environment (shared / offline HF) -------------------------------------
export HF_HOME=${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# --- Experiment config (match existing baselines in logs_14b_w4a4/) --------
MODEL_PATH=${MODEL_PATH:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B}
GPU_LIST=${GPU_LIST:-"0,1,2,3,4,5,6,7"}
NPROC=${NPROC:-8}

export N_SAMPLES=${N_SAMPLES:-1024}
export SEQ_LEN=${SEQ_LEN:-2048}
export A_BITS=${A_BITS:-4}
export A_ASYM=${A_ASYM:-1}
export A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
export W_CLIP=${W_CLIP:-1}
export ALPHA=${ALPHA:-0.25}

# ReQuant hyper-params (as agreed for W4A4 ablation).
export REQUANT_SWEEPS=${REQUANT_SWEEPS:-4}
export REQUANT_CANDIDATES=${REQUANT_CANDIDATES:-2}
export REQUANT_BETA=${REQUANT_BETA:-0.25}
export GPTQ_REQUANT_BETA=${GPTQ_REQUANT_BETA:-0.25}
export GPTAQ_REQUANT_BETA=${GPTAQ_REQUANT_BETA:-0.25}
export REQUANT_MIN_GAIN_EPS=${REQUANT_MIN_GAIN_EPS:-0.2}

# AWQ defaults (match baseline).
export AWQ_NSAMPLES=${AWQ_NSAMPLES:-128}
export AWQ_GRID=${AWQ_GRID:-20}
export AWQ_MIN_ALPHA=${AWQ_MIN_ALPHA:-0.0}
export AWQ_MAX_ALPHA=${AWQ_MAX_ALPHA:-1.0}

# Downstream evaluation inside each launcher (wikitext2 + ultrachat_2k + numinamath).
export LM_EVAL=${LM_EVAL:-1}
export LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-64}

# Keep CPU as default device for loading (avoids GPU0 OOM on 14B);
# each rank will move layers to its own cuda:LOCAL_RANK during calibration.
export MODEL_DEVICE_MAP=${MODEL_DEVICE_MAP:-cpu}

LOG_DIR=${LOG_DIR:-logs_14b_w4a4}
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/run_14b_requant_w4a4.log"

run_method() {
  local tag="$1"   # e.g. 05_gptaq_w4a4_rot_requant
  local method="$2" # gptaq|gptq|rtn|awq
  local log="${LOG_DIR}/${tag}.log"

  {
    echo "============================================================"
    echo "[$(date '+%F %T')] START ${tag}"
    echo "log -> ${log}"
    echo "============================================================"
  } | tee -a "${SUMMARY_LOG}"

  RUN_ONLY="${method}" \
  bash scripts/run_w4a4_ablation.sh "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}" \
    >"${log}" 2>&1
  local rc=$?

  {
    echo "[$(date '+%F %T')] END   ${tag} rc=${rc}"
  } | tee -a "${SUMMARY_LOG}"

  return ${rc}
}

echo "[$(date '+%F %T')] driver start; HF_HOME=${HF_HOME} MODEL=${MODEL_PATH}" \
  | tee -a "${SUMMARY_LOG}"

# Order: match baseline numbering (01..04 were baselines). 05..08 = +ReQuant.
run_method "05_gptaq_w4a4_rot_requant" "gptaq" || true
run_method "06_gptq_w4a4_rot_requant"  "gptq"  || true
run_method "07_rtn_w4a4_rot_requant"   "rtn"   || true
run_method "08_awq_w4a4_rot_requant"   "awq"   || true

echo "[$(date '+%F %T')] ALL DONE" | tee -a "${SUMMARY_LOG}"
