#!/bin/bash
# -----------------------------------------------------------------------------
# Add ReQuant on top of the existing W4A4 + Rotate baselines for Qwen3-14B.
#
# Existing baselines (already rc=0 in logs_14b_w4a4/):
#   01 GPTAQ + Rotate       (ns=512)
#   02 GPTQ  + Rotate       (ns=512)
#   03 RTN   + Rotate       (ns=1024)
#   04 AWQ   + Rotate       (ns=512)
#
# This driver reruns each with ReQuant enabled, all other knobs held constant:
#   A_BITS=4, A_ASYM=1, A_CLIP_RATIO=0.9, ALPHA=0.25
#   REQUANT=1, SWEEPS=4, CANDIDATES=2, MIN_GAIN_EPS=0.2
#   REQUANT_BETA=0.25  (user-specified, differs from W4A16 run)
#
# Outputs per run  -> ./outputs/Qwen3-14B/<exp>/...
# Per-run log      -> logs_14b_w4a4/0{5..8}_*.log
# Master log       -> logs_14b_w4a4/run_14b_requant_w4a4.master.log
#
# Usage:
#   bash scripts/run_w4a4_requant_on_baselines.sh [MODEL_PATH] [GPU_LIST] [NPROC]
#
# Intended to be launched inside a tmux session.
# -----------------------------------------------------------------------------

set -u -o pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

# Shared HF cache / endpoints.
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh" >/dev/null

MODEL_PATH=${1:-${QWEN3_14B_PATH:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B}}
GPU_LIST=${2:-"0,1,2,3,4,5,6,7"}
NPROC=${3:-8}

LOG_DIR="${REPO_ROOT}/logs_14b_w4a4"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/run_14b_requant_w4a4.master.log"

# ReQuant hyper-params (user-confirmed).
export ALPHA=0.25
export A_BITS=4
export A_ASYM=1
export A_CLIP_RATIO=0.9
export REQUANT_SWEEPS=4
export REQUANT_CANDIDATES=2
export REQUANT_BETA=0.25
export GPTQ_REQUANT_BETA=0.25
export GPTAQ_REQUANT_BETA=0.25
export REQUANT_MIN_GAIN_EPS=0.2
export SEQ_LEN=2048

# Evaluation inline (matches baselines: lm_eval=1, bs=64, 3 PPL datasets).
export LM_EVAL=1
export LM_EVAL_BATCH_SIZE=64

banner() {
  {
    echo "============================================================"
    echo "[$(date '+%F %T')] $1"
    echo "============================================================"
  } | tee -a "${MASTER_LOG}"
}

note() {
  echo "[$(date '+%F %T')] $*" | tee -a "${MASTER_LOG}"
}

run_step() {
  local tag="$1"     # e.g. 05_gptaq_w4a4_rot_requant
  local launcher="$2"
  shift 2
  local -a extra_env=("$@")

  local log_file="${LOG_DIR}/${tag}.log"
  banner "START ${tag}"
  note   "log -> ${log_file}"
  note   "launcher -> ${launcher}"
  note   "extra env -> ${extra_env[*]}"

  # Use `env` so the exported REQUANT_* above are still visible but per-run
  # overrides (EXP name, MASTER_PORT, nsamples) take priority.
  env "${extra_env[@]}" bash "${launcher}" "${MODEL_PATH}" "${GPU_LIST}" "${NPROC}" \
       >"${log_file}" 2>&1
  local rc=$?

  banner "END   ${tag} rc=${rc}"
  if [[ ${rc} -ne 0 ]]; then
    note "[WARN] ${tag} failed with rc=${rc}; continuing with next run."
  fi
  return 0
}

banner "BEGIN W4A4 + ReQuant ablations on Qwen3-14B"
note "model=${MODEL_PATH}  GPUs=${GPU_LIST}  nproc=${NPROC}"
note "A_BITS=${A_BITS} A_ASYM=${A_ASYM} A_CLIP_RATIO=${A_CLIP_RATIO}"
note "REQUANT_BETA=${REQUANT_BETA} sweeps=${REQUANT_SWEEPS} cands=${REQUANT_CANDIDATES} min_gain_eps=${REQUANT_MIN_GAIN_EPS}"

# 05 GPTAQ + Rotate + ReQuant  (baseline used ns=512)
# NOTE: DP_FASTERQUANT_MASTER_ONLY=1 + DP_SHARD_INPS=1 mirror the W4A4
# baseline's memory layout (only rank0 holds the fasterquant state; activations
# are sharded across ranks). Without these two the 14B weights + activations
# replicate on every rank and blow a 140 GiB card even with cpu offload.
run_step "05_gptaq_w4a4_rot_requant" "scripts/gptaq.sh" \
  EXP=gptaq \
  N_SAMPLES=512 \
  ROTATE=1 \
  REQUANT=1 \
  DP_FASTERQUANT_MASTER_ONLY=1 \
  DP_SHARD_INPS=1 \
  MASTER_PORT=29721

# 06 GPTQ  + Rotate + ReQuant  (baseline used ns=512)
run_step "06_gptq_w4a4_rot_requant" "scripts/gptq.sh" \
  EXP=gptq \
  N_SAMPLES=512 \
  ROTATE=1 \
  REQUANT=1 \
  DP_FASTERQUANT_MASTER_ONLY=1 \
  DP_SHARD_INPS=1 \
  MASTER_PORT=29711

# 07 RTN   + Rotate + ReQuant  (baseline used ns=1024, exp=rtn_dp)
run_step "07_rtn_w4a4_rot_requant" "scripts/rtn.sh" \
  EXP=rtn_dp \
  N_SAMPLES=1024 \
  ROTATE=1 \
  REQUANT=1 \
  MASTER_PORT=29741

# 08 AWQ   + Rotate + ReQuant  (baseline used ns=512, exp=awq_dp)
run_step "08_awq_w4a4_rot_requant" "scripts/awq.sh" \
  EXP=awq_dp \
  N_SAMPLES=512 \
  ROTATE=1 \
  REQUANT=1 \
  AWQ_GRID=20 AWQ_MIN_ALPHA=0.0 AWQ_MAX_ALPHA=1.0 \
  MASTER_PORT=29731

banner "ALL DONE"
