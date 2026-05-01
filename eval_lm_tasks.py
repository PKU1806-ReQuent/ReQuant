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
"""

import argparse
import gc
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
    p.add_argument(
        "--model_device_map",
        type=str,
        default=os.environ.get("MODEL_DEVICE_MAP", "auto"),
        help=(
            "Device map used when loading the base model. Use 'auto' for "
            "multi-GPU loading, 'cuda:0' for single large GPU, or 'cpu' for "
            "the old CPU-first path."
        ),
    )
    p.add_argument(
        "--checkpoint_load_device",
        type=str,
        default="cpu",
        help=(
            "Device used by torch.load for the .pt checkpoint. Keep this as "
            "'cpu' for large models; CPU mmap avoids holding a second full "
            "copy on GPU."
        ),
    )
    p.add_argument(
        "--checkpoint_mmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use torch.load(..., mmap=True) when supported to reduce CPU memory peak.",
    )
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
    p.add_argument("--a_bits", type=int, default=16)
    p.add_argument("--a_groupsize", type=int, default=-1)
    p.add_argument("--a_asym", action="store_true")
    p.add_argument("--a_clip_ratio", type=float, default=1.0)
    p.add_argument("--v_bits", type=int, default=16)
    p.add_argument("--v_groupsize", type=int, default=-1)
    p.add_argument("--v_asym", action="store_true")
    p.add_argument("--v_clip_ratio", type=float, default=1.0)
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

    os.environ["MODEL_DEVICE_MAP"] = args.model_device_map
    logging.info(
        "Loading base model from %s with device_map=%s",
        args.model,
        args.model_device_map,
    )
    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    if args.model_device_map == "cpu":
        model.cpu()
    else:
        logging.info("Keeping base model on load device_map=%s", args.model_device_map)
    remove_hook_from_module(model, recurse=True)

    logging.info("add_actquant (structure must match training)")
    quant_utils.add_actquant(analyzer)
    if args.rotate:
        logging.info("rotate=1: configure down_proj online Hadamard (match ptq_dp.py)")
        _configure_rotate_down_proj(model)
    quant_utils.configure_actquant_from_args(args, model)

    logging.info(
        "Loading weights from %s (map_location=%s, mmap=%s)",
        args.load_qmodel_path,
        args.checkpoint_load_device,
        args.checkpoint_mmap,
    )
    # These checkpoints may include custom objects, so disable weights_only when available.
    try:
        load_kwargs = {
            "map_location": args.checkpoint_load_device,
            "weights_only": False,
        }
        if args.checkpoint_mmap:
            load_kwargs["mmap"] = True
        save_dict = torch.load(args.load_qmodel_path, **load_kwargs)
    except TypeError:
        save_dict = torch.load(
            args.load_qmodel_path, map_location=args.checkpoint_load_device
        )
    if "model" not in save_dict:
        logging.error("Checkpoint is missing key 'model'")
        sys.exit(1)
    state_dict = save_dict["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict, save_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if missing:
        logging.warning("load_state_dict missing keys (first 20): %s", list(missing)[:20])
    if unexpected:
        logging.warning("load_state_dict unexpected keys (first 20): %s", list(unexpected)[:20])

    model.eval()
    if torch.cuda.is_available():
        if args.model_device_map == "cpu":
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
        else:
            logging.info(
                "Model already loaded with device_map=%s; skipping post-load placement",
                args.model_device_map,
            )

    logging.info("Running qa_eval (same tasks as ptq.py --lm_eval)")
    eval_utils.qa_eval(model, analyzer.tokenizer, args.lm_eval_batch_size)


if __name__ == "__main__":
    main()