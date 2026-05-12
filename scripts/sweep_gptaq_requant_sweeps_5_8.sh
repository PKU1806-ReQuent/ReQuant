#!/bin/bash
# Sweep GPTAQ ReQuant sweeps 5-8. This reuses the main GPTAQ sweep script.
#
# Usage:
#   bash scripts/sweep_gptaq_requant_sweeps_5_8.sh [model path] [visible GPU list] [nproc]
#
# Example:
#   bash scripts/sweep_gptaq_requant_sweeps_5_8.sh ./modelzoo/Meta-Llama-3-8B "4,5,6,7" 4

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

SWEEP_LIST=${SWEEP_LIST:-"5 6 7 8"} \
BASE_EXP=${BASE_EXP:-gptaq_rot_sweep_5_8} \
bash "${SCRIPT_DIR}/sweep_gptaq_requant_sweeps.sh" "$@"
