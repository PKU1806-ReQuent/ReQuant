# coding=utf-8
"""Unified torchrun entry point for data-parallel PTQ methods.

Launch with::

    torchrun --nnodes=1 --nproc_per_node=N ptq_dp.py \
        --model ... --w_method {gptq,gptaq,awq} ...

``--enable_requant`` enables Phase-2 ReQuant for GPTQ/GPTAQ. AWQ uses its
own ``--requant_sweeps`` post-processing path.
"""

from process_args import parse_gen
from gptq_utils.quantize_awq_dp import quantize_weights_awq_dp
from gptq_utils.quantize_gptaq_dp import quantize_weights_gptaq_dp
from gptq_utils.ptq_dp_runner import run_dp_ptq


DP_QUANTIZERS = {
    "gptq": quantize_weights_gptaq_dp,
    "gptaq": quantize_weights_gptaq_dp,
    "awq": quantize_weights_awq_dp,
}


def main(args):
    if args.w_method not in DP_QUANTIZERS:
        supported = ", ".join(sorted(DP_QUANTIZERS))
        raise ValueError(
            f"ptq_dp.py supports DP methods {{{supported}}}; got {args.w_method}"
        )
    run_dp_ptq(
        args,
        DP_QUANTIZERS[args.w_method],
        expected_methods=DP_QUANTIZERS.keys(),
        entry_name="ptq_dp",
    )


if __name__ == "__main__":
    main(parse_gen())
