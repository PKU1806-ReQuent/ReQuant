#!/bin/bash
# Sequential runner for Qwen3-14B +ReQuant experiments (Quarot + W4A16 + Phase-2 ReQuant).
# Launched inside a tmux session. Each experiment reuses up to 8 GPUs.
# ReQuant hyper-parameters use the smoke-validated defaults (sweeps=1, cand=2, beta=0.3, eps=0.003).
set -u

MODEL=${MODEL:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/outputs}
LOG_DIR=${LOG_DIR:-logs_14b}
GPUS_8="0,1,2,3,4,5,6,7"

mkdir -p "${LOG_DIR}"

# --- Conda env bypass (avoid 'conda activate' in non-login shells) ---
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/data/miniconda3/envs/requant}"
export PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}"
export PATH="${CONDA_ENV_DIR}/bin:${PATH}"

export OUTPUT_ROOT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"
export LM_EVAL_ENABLE=1
export LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-64}"
export ROTATE=1
export REQUANT=1

# ReQuant Phase-2 hyper-parameters
export REQUANT_SWEEPS="${REQUANT_SWEEPS:-4}"
export REQUANT_CANDIDATES="${REQUANT_CANDIDATES:-2}"
export REQUANT_BETA="${REQUANT_BETA:-0.3}"
export REQUANT_MIN_GAIN_EPS="${REQUANT_MIN_GAIN_EPS:-0.2}"
# Anchor left unset (disabled). Set REQUANT_ANCHOR_LAMBDA=0.05 to enable if needed.

# Real 8-GPU DP for GPTQ/GPTAQ/AWQ (RTN script is single-GPU by design).
export DP_SHARD_INPS="${DP_SHARD_INPS:-1}"
export DP_FASTERQUANT_MASTER_ONLY="${DP_FASTERQUANT_MASTER_ONLY:-1}"
export OFFLOAD_INPS="${OFFLOAD_INPS:-0}"

run_step() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}.log"
    echo "============================================================"
    echo "[$(date '+%F %T')] START ${name}"
    echo "log -> ${log}"
    echo "============================================================"
    ( "$@" ) > "${log}" 2>&1
    local rc=$?
    echo "[$(date '+%F %T')] END   ${name} rc=${rc}"
    if [[ ${rc} -ne 0 ]]; then
        echo "[FATAL] ${name} failed with rc=${rc}. Aborting the pipeline."
        exit ${rc}
    fi
}

# Order: fastest Phase-1 first to de-risk.
# 1) GPTAQ + Quarot + ReQuant (8-GPU DP)
run_step 06_gptaq_w4a16_rot_requant \
    env A_BITS=16 MASTER_PORT=29721 \
    bash scripts/gptaq.sh "${MODEL}" "${GPUS_8}" 8

# 2) GPTQ + Quarot + ReQuant (8-GPU DP)
run_step 07_gptq_w4a16_rot_requant \
    env A_BITS=16 MASTER_PORT=29711 \
    bash scripts/gptq.sh "${MODEL}" "${GPUS_8}" 8

# 3) RTN + Quarot + ReQuant (single-GPU, fastest)
run_step 08_rtn_w4a16_rot_requant \
    bash scripts/rtn.sh "${MODEL}" 0

# 4) AWQ + Quarot + ReQuant (8-GPU DP)
run_step 09_awq_w4a16_rot_requant \
    env MASTER_PORT=29731 \
    bash scripts/awq.sh "${MODEL}" "${GPUS_8}" 8

echo "[$(date '+%F %T')] ALL DONE"
