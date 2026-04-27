import argparse
import os
import logging

import transformers

from utils import dist_utils
from utils.log_utils import init_logging


def parse_gen():
    parser = argparse.ArgumentParser(description="Quantize a model to any precision")
    parser.add_argument("--model", type=str, required=True, help="The model to quantize")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument("--seed", type=int, default=42,
                        help="The random state to use for reproducibility\n"
                             "[WARNING] May not be reproducible across different machines")
    # Paths
    parser.add_argument("--output_dir", type=str, default="./outputs", help="The directory to save results in")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="The directory to cache results in")
    # Datasets
    parser.add_argument("--dataset", type=str, default="neuralmagic",
                        choices=["wikitext2", "neuralmagic", "ultrachat_2k", "numinamath"], help="The dataset to use")
    parser.add_argument("--nsamples", type=int, default=1024, help="The number of examples to use")
    parser.add_argument("--seq_len", type=int, default=2048, help="The sequence length to use in calibration")
    parser.add_argument("--eval_seq_len", type=int, default=2048, help="The sequence length to use in PPL&KL evaluation")
    # Gradient profiling
    parser.add_argument("--mode", type=str, default="gradients", choices=["tokens", "gradients"],
                        help="The mode to run in")
    # Quantization configs
    parser.add_argument("--w_bits", type=int, default=16, help="Weight bits")
    parser.add_argument("--a_bits", type=int, default=16, help="Activation bits")
    parser.add_argument("--k_bits", type=int, default=16, help="K cache bits")
    parser.add_argument("--v_bits", type=int, default=16, help="V cache bits")
    parser.add_argument("--w_groupsize", type=int, default=-1, help="Weight group size")
    parser.add_argument("--a_groupsize", type=int, default=-1, help="Activation group size")
    parser.add_argument("--k_groupsize", type=int, default=-1, help="K cache group size")
    parser.add_argument("--v_groupsize", type=int, default=-1, help="V cache group size")
    parser.add_argument("--w_asym", action="store_true", help="Weight asymmetric quantization")
    parser.add_argument("--a_asym", action="store_true", help="Activation asymmetric quantization")
    parser.add_argument("--k_asym", action="store_true", help="K cache asymmetric quantization")
    parser.add_argument("--v_asym", action="store_true", help="V cache asymmetric quantization")
    parser.add_argument("--w_clip", action="store_true", help="Enable weight clipping")
    parser.add_argument("--a_clip_ratio", type=float, default=1.0, help="Activation clipping ratio")
    parser.add_argument(
        "--enable_requant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable ReQuant Phase-2 as an independent component after GPTAQ/GPTQ Phase-1.",
    )
    parser.add_argument(
        "--enable_aq_calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Configure activation quantizers before weight PTQ so calibration "
            "forwards include fake activation quantization."
        ),
    )
    parser.add_argument("--k_clip_ratio", type=float, default=1.0, help="K cache clipping ratio")
    parser.add_argument("--v_clip_ratio", type=float, default=1.0, help="V cache clipping ratio")
    parser.add_argument("--export_to_et", action="store_true", help="Export quantized model (TODO)")
    # Rotate
    parser.add_argument("--optimized_rotation_path", type=str, default=None, help="The path to rotation ckpt")
    parser.add_argument("--rotate", action="store_true", help="Rotate model (SpinQuant)")
    # GPTQ
    parser.add_argument("--w_method", type=str, default="gptq",
                        choices=["rtn", "gptq", "gptaq", "gptq_guided", "gptq_plus", "awq"],
                        help="Weight quantization method")
    parser.add_argument("--act_order", action="store_true", help="Activation reorder (with static groups)")
    parser.add_argument("--num_groups", type=int, default=4,
                        help="Number of groups $g$ to use for block-diagonal Hessian")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="GPTAQ dXXT correction strength for P-matrix (Phase-1); scripts often use 0.25",
    )
    parser.add_argument(
        "--requant_phase2_sweeps",
        dest="requant_phase2_sweeps",
        type=int,
        default=4,
        help="Phase-2 requant (GPTQReQuant): coordinate-descent sweeps on the int grid (0 disables)",
    )
    parser.add_argument(
        "--requant_phase2_candidates",
        dest="requant_phase2_candidates",
        type=int,
        default=2,
        help="Phase-2 requant: ±k integer steps around current W_int per column (grid half-width)",
    )
    parser.add_argument("--requant_sweeps", type=int, default=0, help="requant_layer: coordinate descent sweeps after base GPTAQ/AWQ (0 to disable)")
    parser.add_argument("--requant_candidates", type=int, default=2, help="requant_layer: grid neighbours per direction")
    parser.add_argument(
        "--requant_anchor_lambda",
        dest="requant_anchor_lambda",
        type=float,
        default=0.2,
        help="Phase-2 requant: anchor λ to keep Q near Phase-1 init (GPTQReQuant)",
    )
    parser.add_argument(
        "--requant_beta",
        dest="requant_beta",
        type=float,
        default=0.3,
        help="Phase-2 requant: scales dXXT term in gradient G.",
    )
    parser.add_argument(
        "--requant_min_gain_eps",
        type=float,
        default=0.2,
        help="requant: accept move only if best_gain < -eps * row_loss / columns. "
        "0 => any strict improvement.",
    )
    parser.add_argument(
        "--awq_grid",
        type=int,
        default=20,
        help="awq only: number of alpha candidates in the scale search grid",
    )
    parser.add_argument(
        "--awq_min_alpha",
        type=float,
        default=0.0,
        help="awq only: minimum alpha in the scale search grid",
    )
    parser.add_argument(
        "--awq_max_alpha",
        type=float,
        default=1.0,
        help="awq only: maximum alpha in the scale search grid",
    )
    parser.add_argument(
        "--awq_scale_eps",
        type=float,
        default=1e-4,
        help="awq only: minimum scale clamp value",
    )
    parser.add_argument(
        "--awq_search_samples",
        type=int,
        default=128,
        help="awq only: number of global calibration samples used for output-MSE scale search",
    )
    parser.add_argument(
        "--awq_clip_tokens",
        type=int,
        default=512,
        help="awq only: maximum sampled tokens per layer used for AWQ clipping",
    )
    parser.add_argument(
        "--awq_clip_grid",
        type=int,
        default=20,
        help="awq only: grid size for AWQ clipping search",
    )
    parser.add_argument(
        "--awq_clip_max_shrink",
        type=float,
        default=0.5,
        help="awq only: maximum shrink ratio for AWQ clipping search",
    )
    parser.add_argument("--kl_topk", type=int, default=-1, help="Top-k KL loss")
    parser.add_argument("--bsz", type=int, default=1, help="Batch size for computing hessians and gradients")
    parser.add_argument("--load_qmodel_path", type=str, default=None, help="The path to load quantized model ckpt")
    parser.add_argument("--save_qmodel_path", type=str, default=None, help="The path to save quantized model ckpt")
    parser.add_argument("--offload_inps", action="store_true", help="Offload inputs to CPU")
    parser.add_argument(
        "--dp_fasterquant_master_only",
        action="store_true",
        help="DP with optional ReQuant (ptq_dp.py): reduce Hessian to rank 0 "
        "only, run fasterquant only on rank 0, then broadcast each submodule's weight. Frees full H/dXXT + "
        "Cholesky (and ReQuant on ReQuant path) VRAM on worker GPUs. Not supported with --export_to_et.",
    )
    parser.add_argument(
        "--dp_shard_inps",
        action="store_true",
        help="gptaq DP with optional ReQuant: rank0 captures full calibration activations on CPU, scatters per-rank "
        "shards so each GPU holds only ~1/world_size of inps/fp_inps. Layer-wise updates run locally per "
        "shard (weights are already synced). Reduces per-GPU VRAM vs a full [nsamples,seqlen,hidden] buffer.",
    )
    # Eval
    parser.add_argument("--lm_eval", action="store_true", help="Enable QA eval")
    parser.add_argument("--lm_eval_batch_size", type=int, default=32, help="Batch size for QA tasks")
    parser.add_argument("--eval_datasets", type=list[str], default=["wikitext2", "ultrachat_2k", "numinamath"],
                        help="Datasets for PPL & KL eval")
    # Exp
    parser.add_argument("--enable_debug", action="store_true", help="Enable debugging")

    args = parser.parse_args()

    # set paths & others
    args.model_name = args.model.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, args.model_name, args.exp)
    args.log_dir = os.path.join(args.output_dir, "logs")
    args.tokens_cache_path = (f"{args.cache_dir}/tokens/"
                              f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}.pt")
    if args.num_groups is not None:
        args.saliency_cache_path = (f"{args.cache_dir}/saliency/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}")
        args.gradients_cache_path = (f"{args.cache_dir}/gradients/"
                                    f"{args.model_name}-{args.dataset}_s{args.nsamples}_blk{args.seq_len}_g{args.num_groups}.pt")
    else:
        args.saliency_cache_path = None
        args.gradients_cache_path = None

    transformers.set_seed(args.seed)

    init_logging(args.log_dir)
    logging.info(args)

    # Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.enable_debug:
        import debugpy
        debugpy.listen(5678 + dist_utils.get_rank())
        logging.info("Waiting for debugger attach")
        debugpy.wait_for_client()
        # debugpy.breakpoint()

    return args
