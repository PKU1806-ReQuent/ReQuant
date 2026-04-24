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
# 激活量化：结构为 utils/quant_utils.py 的 ActQuantWrapper + ActQuantizer；ptq_gptaq_refined_dp.py 在
#   add_actquant 之后若 a_bits<16 会调用 quantizer.configure。默认开启（A_BITS=4），关闭：export A_BITS=16。
#   可选：A_GROUPSIZE / A_ASYM / A_CLIP_RATIO（见下方 EXTRA_ACT）。
#   与 GPTAQ/fake_quant 一致：默认启用 --enable_aq_calibration（可通过 ENABLE_AQ_CALIBRATION=0 关闭），
#   在校准前向中先启用激活伪量化再跑权重量化。
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
REFINE_SWEEPS=${REFINE_SWEEPS:-0}
REFINE_CANDIDATES=${REFINE_CANDIDATES:-2}
REFINE_BETA=${REFINE_BETA:-1.0}
REFINE_MIN_GAIN_EPS=${REFINE_MIN_GAIN_EPS:-0.003}
ROTATE=${ROTATE:-1}

# 激活量化：默认 A4，可通过 A_BITS 调整（16=关闭）
A_BITS=${A_BITS:-4}
A_GROUPSIZE=${A_GROUPSIZE:--1}
A_CLIP_RATIO=${A_CLIP_RATIO:-0.9}
A_ASYM=${A_ASYM:-1}
ENABLE_AQ_CALIBRATION=${ENABLE_AQ_CALIBRATION:-1}

# 可选：扫 anchor 时 export GPTAQ_DP_CKPT_TAG=_lambda1e-3，避免覆盖同名 .pt
ACT_TAG=""
if [[ "${A_BITS}" != "16" ]]; then
  ACT_TAG="_a${A_BITS}"
  [[ "${A_GROUPSIZE}" != "-1" ]] && ACT_TAG="${ACT_TAG}_ags${A_GROUPSIZE}"
  [[ "${A_ASYM}" == "1" ]] && ACT_TAG="${ACT_TAG}_aasym"
fi
if [[ "${ENABLE_AQ_CALIBRATION}" == "1" ]]; then
  ACT_TAG="${ACT_TAG}_aqcal"
fi
SAVE_BASE="./outputs/${MODEL_NAME}/${EXP}/gptaq_refined_dp_w4${ACT_TAG}_ns${N_SAMPLES}_a${ALPHA}_s${REFINE_SWEEPS}_c${REFINE_CANDIDATES}_dp${NPROC}"
SAVE_QMODEL_PATH="${SAVE_BASE}${GPTAQ_DP_CKPT_TAG:-}.pt"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29600}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  PYTHON_BIN="/root/miniconda3/envs/gptq/bin/python"
fi

echo "[Info] DP ranks=${NPROC} GPUs=${GPU_LIST} alpha=${ALPHA} refine_beta=${REFINE_BETA} refine_min_gain_eps=${REFINE_MIN_GAIN_EPS} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
if [[ "${A_BITS}" == "16" ]]; then
  echo "[Info] activation quant: off (A_BITS=16)"
else
  echo "[Info] activation quant: A_BITS=${A_BITS} a_groupsize=${A_GROUPSIZE} a_clip_ratio=${A_CLIP_RATIO} a_asym=${A_ASYM}"
fi
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
if [[ "${ENABLE_AQ_CALIBRATION}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_aq_calibration)
  echo "[Info] enable_aq_calibration=1 (activation configure before weight PTQ, GPTAQ/fake_quant style)"
else
  EXTRA_ARGS+=(--no-enable_aq_calibration)
  echo "[Info] enable_aq_calibration=0 (use original ReQuant order: weight PTQ then activation configure)"
fi

EXTRA_ACT=()
if [[ "${A_BITS}" != "16" ]]; then
  EXTRA_ACT+=(--a_bits "${A_BITS}" --a_groupsize "${A_GROUPSIZE}" --a_clip_ratio "${A_CLIP_RATIO}")
  if [[ "${A_ASYM}" == "1" ]]; then
    EXTRA_ACT+=(--a_asym)
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
    --refine_beta "${REFINE_BETA}" \
    --refine_min_gain_eps "${REFINE_MIN_GAIN_EPS}" \
    "${EXTRA_ANCHOR[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${EXTRA_ACT[@]}" \
    --save_qmodel_path "${SAVE_QMODEL_PATH}" \
    --offload_inps \
    --lm_eval --lm_eval_batch_size 32
