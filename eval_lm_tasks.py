# coding=utf-8
"""
Run lm-eval tasks for a saved quantized checkpoint.

The task set matches the repo's qa_eval/lm_eval defaults. This script loads an
existing ``save_qmodel_path`` checkpoint and does not run PTQ again.

Example::

    python eval_lm_tasks.py \\
        --model ./modelzoo/Qwen3/Qwen3-0.6B \\
        --load_qmodel_path ./outputs/Qwen3-0.6B/gptaq_refined_dp_w4_ns512_a0.5_s3_c2_dp2.pt \\
        --lm_eval_batch_size 32

Multi-GPU example::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_lm_tasks.py \\
        --model ./modelzoo/Qwen3/Qwen3-14B \\
        --load_qmodel_path ./outputs/Qwen3-14B/gptaq_dp/xxx.pt \\
        --placement dispatch \\
        --lm_eval_batch_size 4

``--placement`` selects automatic placement, forced dispatch, or single-GPU
execution.

``--rotate`` must match checkpoints produced with SpinQuant rotation.

``--a_bits`` / ``--a_asym`` / ``--a_clip_ratio`` must match W4A4 checkpoints
so that activation quantization is enabled during evaluation.
"""

import argparse
import logging
import os
import sys

import torch
from accelerate.hooks import remove_hook_from_module

from utils import dist_utils, eval_utils, hadamard_utils, model_utils, quant_utils
from utils.log_utils import init_logging


def _configure_rotate_down_proj(model) -> None:
    """Configure down_proj Hadamard buffers to match rotated PTQ checkpoints."""
    qlayers = quant_utils.find_qlayers(model)
    for name in qlayers:
        if "down_proj" not in name:
            continue
        had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
        m = qlayers[name]
        m.online_full_had = True
        m.had_K = had_k
        m.K = k
        m.fp32_had = False


def parse_args():
    p = argparse.ArgumentParser(description="Load quantized ckpt and run lm-eval harness tasks")
    p.add_argument("--model", type=str, required=True, help="Original HF model path used during PTQ")
    p.add_argument(
        "--load_qmodel_path",
        type=str,
        required=True,
        help="Saved PTQ .pt checkpoint containing a model state_dict",
    )
    p.add_argument("--seq_len", type=int, default=2048, help="Sequence length for ModelAnalyzer")
    p.add_argument("--lm_eval_batch_size", type=int, default=32)
    p.add_argument("--a_bits", type=int, default=16, help="Activation bits used during PTQ")
    p.add_argument("--a_groupsize", type=int, default=-1, help="Activation group size used during PTQ")
    p.add_argument("--a_clip_ratio", type=float, default=1.0, help="Activation clipping ratio used during PTQ")
    p.add_argument("--a_asym", action="store_true", help="Use asymmetric activation quantization")
    p.add_argument("--v_bits", type=int, default=16, help="V-cache/output activation bits")
    p.add_argument("--v_groupsize", type=int, default=-1, help="V-cache/output activation group size")
    p.add_argument("--v_clip_ratio", type=float, default=1.0, help="V-cache/output activation clipping ratio")
    p.add_argument("--v_asym", action="store_true", help="Use asymmetric V-cache/output activation quantization")
    p.add_argument(
        "--placement",
        type=str,
        default="auto",
        choices=("auto", "single", "dispatch"),
        help="Model placement: auto, forced dispatch, or single cuda:0",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Log directory; defaults to lm_eval_only/logs next to the checkpoint",
    )
    p.add_argument(
        "--rotate",
        action="store_true",
        help="Enable when evaluating checkpoints quantized with --rotate",
    )
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = args.log_dir
    if not log_dir:
        base = os.path.dirname(os.path.abspath(args.load_qmodel_path))
        log_dir = os.path.join(base, "lm_eval_only", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_dir)

    if not os.path.isfile(args.load_qmodel_path):
        logging.error("Checkpoint not found: %s", args.load_qmodel_path)
        sys.exit(1)

    logging.info("Loading base model from %s", args.model)
    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    model.cpu()
    remove_hook_from_module(model, recurse=True)

    logging.info("add_actquant (structure must match training)")
    quant_utils.add_actquant(analyzer)
    quant_utils.configure_actquant_from_args(args, model)
    if args.rotate:
        logging.info("rotate=1: configure down_proj online Hadamard (match ptq_dp.py)")
        _configure_rotate_down_proj(model)

    logging.info("Loading weights from %s", args.load_qmodel_path)
    # These checkpoints may include custom objects, so disable weights_only when available.
    try:
        save_dict = torch.load(
            args.load_qmodel_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        save_dict = torch.load(args.load_qmodel_path, map_location="cpu")
    if "model" not in save_dict:
        logging.error("Checkpoint is missing key 'model'")
        sys.exit(1)
    missing, unexpected = model.load_state_dict(save_dict["model"], strict=False)
    if missing:
        logging.warning("load_state_dict missing keys (first 20): %s", list(missing)[:20])
    if unexpected:
        logging.warning("load_state_dict unexpected keys (first 20): %s", list(unexpected)[:20])

    model.eval()
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        use_dispatch = args.placement == "dispatch" or (
            args.placement == "auto" and n_gpu > 1
        )
        if use_dispatch:
            logging.info(
                "dispatch_model across %d visible GPU(s) (same as ptq.py lm_eval)", n_gpu
            )
            dist_utils.distribute_model(model)
        else:
            model.cuda()
            logging.info("Model moved to single CUDA device for lm_eval")

    logging.info("Running qa_eval (same tasks as ptq.py --lm_eval)")
    eval_utils.qa_eval(model, analyzer.tokenizer, args.lm_eval_batch_size)


if __name__ == "__main__":
    main()