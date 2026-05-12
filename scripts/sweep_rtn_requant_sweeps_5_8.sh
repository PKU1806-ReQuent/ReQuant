#!/bin/bash
# Sweep RTN ReQuant sweeps 5-8. This reuses the main RTN sweep script.
#
# Usage:
#   bash scripts/sweep_rtn_requant_sweeps_5_8.sh [model path] [visible GPU list] [nproc]
#
# Example:
#   bash scripts/sweep_rtn_requant_sweeps_5_8.sh ./modelzoo/Meta-Llama-3-8B "0,1,2,3" 4

set -u

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

SWEEP_LIST=${SWEEP_LIST:-"5 6 7 8"} \
BASE_EXP=${BASE_EXP:-rtn_sweep_5_8} \
bash "${SCRIPT_DIR}/sweep_rtn_requant_sweeps.sh" "$@"
