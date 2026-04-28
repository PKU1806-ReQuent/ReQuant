# coding=utf-8
"""
PTQ driver for base GPTQ with data-parallel Hessian accumulation (multi-GPU).

Launch example::

    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:29500 \
        ptq_gptq_dp.py --model ... --w_method gptq ...

``--w_method`` must stay ``gptq`` (same checkpoints as single-GPU path).

Internally delegates to the shared :func:`run_dp_ptq` driver so the activation
quantizer configuration timing (``--enable_aq_calibration``) stays consistent
with ``ptq_dp.py`` / ``ptq_rtn_dp.py``.
"""

from process_args import parse_gen
from gptq_utils.quantize_gptq_dp import quantize_weights_gptq_dp
from gptq_utils.ptq_dp_runner import run_dp_ptq


def main(args):
    run_dp_ptq(
        args,
        quantize_weights_gptq_dp,
        expected_methods=("gptq",),
        entry_name="ptq_gptq_dp",
    )


if __name__ == "__main__":
    args = parse_gen()
    main(args)
