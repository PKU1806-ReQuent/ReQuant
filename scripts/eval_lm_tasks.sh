#!/bin/bash
# Run lm-eval for an existing quantized checkpoint.
#
# Usage:
#   bash scripts/eval_lm_tasks.sh <HF model path> <quantized .pt path> [GPU or GPU list]
# Example:
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Llama3/Meta-Llama-3-8B \
#     ./outputs/Meta-Llama-3-8B/gptaq/gptaq_w4_ns512_a0.25_rot_requant_s4_c2.pt 0
# Multi-GPU:
#   bash scripts/eval_lm_tasks.sh ./modelzoo/Llama3/Meta-Llama-3-8B \
#     ./outputs/Meta-Llama-3-8B/gptaq/gptaq_w4_ns512_a0.25_rot.pt 0,1,2,3
# Lower this when lm-eval runs out of memory:
#   LM_EVAL_BATCH_SIZE=8 bash scripts/eval_lm_tasks.sh <model> <ckpt.pt> <GPU>
# Checkpoints with "_rot" in the filename automatically enable --rotate; set ROTATE=1 to force it.
# Activation quantization is auto-detected from the filename pattern
#   "_w<bits>_a<abits>[_aasym]_ns..." produced by scripts/{rtn,gptq,gptaq}.sh. Override via:
#   A_BITS=<n> A_ASYM=<0|1> A_CLIP_RATIO=<r> bash scripts/eval_lm_tasks.sh ...
# Set HTTP_PROXY_URL=<url> to route outbound HTTP(S) through a proxy (useful when
# lm-eval needs to download new datasets from behind a restricted network):
#   HTTP_PROXY_URL=http://proxy.example.com:7890 \
#       bash scripts/eval_lm_tasks.sh <model> <ckpt.pt> <GPU>

MODEL_PATH=${1:?Usage: $0 <model path> <checkpoint.pt> [GPU or GPU list]}
QMODEL=${2:?Usage: $0 <model path> <checkpoint.pt> [GPU or GPU list]}
DEVICE=${3:-0}

if [[ "${MODEL_PATH}" != /* ]]; then
    MODEL_PATH="$(pwd)/${MODEL_PATH}"
fi
if [[ "${QMODEL}" != /* ]]; then
    QMODEL="$(pwd)/${QMODEL}"
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}" || exit 1

export CUDA_VISIBLE_DEVICES=${DEVICE}
PYTHON_BIN="${PYTHON_BIN:-python}"
SEQ_LEN="${SEQ_LEN:-2048}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-32}"
if [[ "${DEVICE}" == *,* ]]; then
    LM_EVAL_PLACEMENT="${LM_EVAL_PLACEMENT:-dispatch}"
else
    LM_EVAL_PLACEMENT="${LM_EVAL_PLACEMENT:-auto}"
fi
if [[ -n "${HTTP_PROXY_URL:-}" ]]; then
    export http_proxy="${HTTP_PROXY_URL}"
    export https_proxy="${HTTP_PROXY_URL}"
    export HTTP_PROXY="${HTTP_PROXY_URL}"
    export HTTPS_PROXY="${HTTP_PROXY_URL}"
fi
QMODEL_BASENAME=$(basename "${QMODEL}")

EXTRA_ARGS=()
if [[ "${ROTATE:-0}" == "1" || "${QMODEL_BASENAME}" == *_rot* ]]; then
    EXTRA_ARGS+=(--rotate)
fi

# Auto-detect activation quantization from the checkpoint filename.
# Match `_w<wbits>_a<abits>[_aasym]_ns` (ACT_TAG slot in rtn/gptq/gptaq scripts).
A_BITS_DEFAULT=16
A_ASYM_DEFAULT=0
if [[ "${QMODEL_BASENAME}" =~ _w[0-9]+_a([0-9]+)(_aasym)?_ns ]]; then
    A_BITS_DEFAULT="${BASH_REMATCH[1]}"
    [[ -n "${BASH_REMATCH[2]}" ]] && A_ASYM_DEFAULT=1
fi
A_BITS="${A_BITS:-${A_BITS_DEFAULT}}"
A_ASYM="${A_ASYM:-${A_ASYM_DEFAULT}}"
A_CLIP_RATIO="${A_CLIP_RATIO:-0.9}"
if [[ "${A_BITS}" != "16" ]]; then
    EXTRA_ARGS+=(--a_bits "${A_BITS}" --a_clip_ratio "${A_CLIP_RATIO}")
    [[ "${A_ASYM}" == "1" ]] && EXTRA_ARGS+=(--a_asym)
    echo "[Info] activation quant: A_BITS=${A_BITS} A_ASYM=${A_ASYM} A_CLIP_RATIO=${A_CLIP_RATIO}"
else
    echo "[Info] activation quant: disabled (A_BITS=16)"
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/eval_lm_tasks.py" \
    --model "${MODEL_PATH}" \
    --load_qmodel_path "${QMODEL}" \
    --seq_len "${SEQ_LEN}" \
    --placement "${LM_EVAL_PLACEMENT}" \
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}" \
    "${EXTRA_ARGS[@]}"
