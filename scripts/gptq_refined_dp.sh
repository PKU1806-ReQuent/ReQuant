#!/bin/bash
# GPTAQ-Refined：多卡数据并行累计 H / dXXT（每 rank 一份 layer 副本，校准样本按 rank 划分）
#
# 用法：
#   bash scripts/gptq_refined_dp.sh [模型路径] [可见 GPU 列表] [每节点进程数]
# 例：4 卡
#   bash scripts/gptq_refined_dp.sh ./modelzoo/Qwen3/Qwen3-1.7B "0,1,2,3" 4
#
# 注意：与单卡脚本不同，不要只绑一张卡；CUDA_VISIBLE_DEVICES 与 nproc_per_node 一致。
# 超参与 scripts/gptq_refined.sh 对齐（N_SAMPLES / alpha / refine 等）；多任务并行可设 MASTER_PORT。
#
# 省显存（Hessian 仍多卡分片累计，Cholesky/refine 仅在 rank0）：在命令中加 --dp_fasterquant_master_only
# 或 export DP_FASTERQUANT_MASTER_ONLY=1（见下方 EXTRA_ARGS）。
# 省显存（每卡只存 1/world_size 的 inps，rank0 CPU 捕获后 scatter）：export DP_SHARD_INPS=1

MODEL_PATH=${1:-./modelzoo/Qwen3/Qwen3-0.6B}
GPU_LIST=${2:-"0,1,2,3"}
NPROC=${3:-4}

MODEL_NAME=$(basename "${MODEL_PATH}")
EXP=gptaq_refined_dp
# 与 scripts/gptq_refined.sh 一致，便于单卡 / DP 数值对比（改一方时请同步另一方）
N_SAMPLES=${N_SAMPLES:-512}
SEQ_LEN=${SEQ_LEN:-2048}
ALPHA=${ALPHA:-0.25}
REFINE_SWEEPS=${REFINE_SWEEPS:-10}
REFINE_CANDIDATES=${REFINE_CANDIDATES:-2}
REFINE_MIN_GAIN_EPS=${REFINE_MIN_GAIN_EPS:-0}
ROTATE=${ROTATE:-1}

# 可选：扫 anchor 时 export GPTAQ_DP_CKPT_TAG=_lambda1e-3，避免覆盖同名 .pt
SAVE_BASE="./outputs/${MODEL_NAME}/${EXP}/gptaq_refined_dp_w4_ns${N_SAMPLES}_a${ALPHA}_s${REFINE_SWEEPS}_c${REFINE_CANDIDATES}_dp${NPROC}"
SAVE_QMODEL_PATH="${SAVE_BASE}${GPTAQ_DP_CKPT_TAG:-}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/gptq/bin/python}"

echo "[Info] DP ranks=${NPROC} GPUs=${GPU_LIST} alpha=${ALPHA} refine_min_gain_eps=${REFINE_MIN_GAIN_EPS} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[Info] output_dir=./outputs/${MODEL_NAME}/${EXP}/ (logs 在 logs/ 子目录)"
echo "[Info] Save path: ${SAVE_QMODEL_PATH}"


EXTRA_ANCHOR=()
if [[ "${REFINE_ANCHOR_LAMBDA+isset}" == "isset" ]]; then
  EXTRA_ANCHOR=(--refine_anchor_lambda "${REFINE_ANCHOR_LAMBDA}")
fi

EXTRA_ARGS=()
if [[ "${DP_FASTERQUANT_MASTER_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_fasterquant_master_only)
  echo "[Info] dp_fasterquant_master_only=1 (H reduce→rank0; fasterquant+broadcast weights)"
fi
if [[ "${DP_SHARD_INPS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--dp_shard_inps)
  echo "[Info] dp_shard_inps=1 (per-rank activation shard; rank0 CPU capture + scatter)"
fi
if [[ "${ROTATE}" == "1" ]]; then
  EXTRA_ARGS+=(--rotate)
  echo "[Info] rotate=1 (SpinQuant-style hadamard rotation)"
  if [[ -n "${OPTIMIZED_ROTATION_PATH:-}" ]]; then
    EXTRA_ARGS+=(--optimized_rotation_path "${OPTIMIZED_ROTATION_PATH}")
    echo "[Info] optimized_rotation_path=${OPTIMIZED_ROTATION_PATH}"
  fi
fi

${PYTHON_BIN} -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
    ./ptq_gptaq_refined_dp.py \
    --model "${MODEL_PATH}" \
    --exp "${EXP}" \
    --dataset wikitext2 --nsamples "${N_SAMPLES}" --seq_len "${SEQ_LEN}" \
    --w_method gptaq_refined --w_bits 4 --w_clip --act_order --w_asym \
    --alpha "${ALPHA}" --refine_sweeps "${REFINE_SWEEPS}" --refine_candidates "${REFINE_CANDIDATES}" \
    --refine_min_gain_eps "${REFINE_MIN_GAIN_EPS}" \
    "${EXTRA_ANCHOR[@]}" \
    "${EXTRA_ARGS[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    --lm_eval --lm_eval_batch_size 32
