#!/bin/bash
# Sequential runner for Qwen3-14B baselines (Quarot + W4A16, with in-process lm_eval).
# Launched inside a tmux session. Each experiment reuses up to 8 GPUs.
set -u

MODEL=${MODEL:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/Qwen3-14B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/outputs}
LOG_DIR=${LOG_DIR:-logs_14b}
GPUS_8="0,1,2,3,4,5,6,7"

mkdir -p "${LOG_DIR}"

export OUTPUT_ROOT
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/apdcephfs_fsgm/share_304739527/peterfywang/model_zoo/datasets/_hf_cache}"
export LM_EVAL_ENABLE=1
# H20-3e has 140GB/card -> we can push batch much higher
export LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-64}"
export ROTATE=1
export REQUANT=0
# Real 8-GPU DP: each rank only processes nsamples/8 calibration samples.
export DP_SHARD_INPS="${DP_SHARD_INPS:-1}"
# Reduce Hessian to rank0, fasterquant+broadcast weights: avoids 7x redundant compute.
export DP_FASTERQUANT_MASTER_ONLY="${DP_FASTERQUANT_MASTER_ONLY:-1}"
# 140GB/card is plenty; keep inps resident on GPU (avoid CPU<->GPU copies).
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

# 1) GPTAQ + Quarot (W4A16), 8-GPU DP
run_step 01_gptaq_w4a16_rot \
    env A_BITS=16 MASTER_PORT=29621 \
    bash scripts/gptaq.sh "${MODEL}" "${GPUS_8}" 8

# 2) GPTQ + Quarot (W4A16), 8-GPU DP
run_step 02_gptq_w4a16_rot \
    env A_BITS=16 MASTER_PORT=29611 \
    bash scripts/gptq.sh "${MODEL}" "${GPUS_8}" 8

# 3) RTN + Quarot (W4A16), single-GPU (script limitation, nproc=1)
run_step 03_rtn_w4a16_rot \
    bash scripts/rtn.sh "${MODEL}" 0

# 4) AWQ + Quarot (W4A16), 8-GPU DP
run_step 04_awq_w4a16_rot \
    env MASTER_PORT=29631 \
    bash scripts/awq.sh "${MODEL}" "${GPUS_8}" 8

echo "[$(date '+%F %T')] ALL DONE"
