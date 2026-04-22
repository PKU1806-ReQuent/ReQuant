# coding=utf-8
"""
仅跑 lm-eval 任务集（与 gptaq / ptq 里 qa_eval 一致：piqa、hellaswag、arc、lambada、ceval-valid 等）。

用于已对模型做完量化并保存了 ``save_qmodel_path`` 的 checkpoint，无需再跑 PTQ。

用法示例::

    python eval_lm_tasks.py \\
        --model ./modelzoo/Qwen3/Qwen3-0.6B \\
        --load_qmodel_path ./outputs/Qwen3-0.6B/gptaq_refined_dp_w4_ns512_a0.5_s3_c2_dp2.pt \\
        --lm_eval_batch_size 32

多卡（与 ``ptq.py`` 的 ``lm_eval`` 一致，Accelerate 按层切分；先 ``export CUDA_VISIBLE_DEVICES=0,1,2,3``）::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_lm_tasks.py \\
        --model ./modelzoo/Qwen3/Qwen3-14B \\
        --load_qmodel_path ./outputs/Qwen3-14B/gptaq_dp/xxx.pt \\
        --placement dispatch \\
        --lm_eval_batch_size 4

``--placement``：``auto``（可见 GPU>1 时自动多卡切分，否则整模上 cuda:0）、``dispatch``（强制多卡切分）、
``single``（整模单卡，易 OOM 时勿用于大模型）。

``--rotate``：与 ``ptq_gptaq_dp.py`` / ``ptq.py`` 中带 ``--rotate`` 训练出的 ckpt 一致，在
``add_actquant`` 之后、``load_state_dict`` 之前为各层 ``mlp.down_proj`` 配置
``online_full_had`` / ``had_K`` / ``K``，否则 ckpt 里这些键会变成 unexpected 且推理错误。
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
    """与 ptq_gptaq_dp.py 中 rotate 分支一致，保证 ActQuantWrapper 能加载 had_K 等 buffer。"""
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
    p.add_argument("--model", type=str, required=True, help="原始 HF 模型目录（与量化时 --model 一致）")
    p.add_argument(
        "--load_qmodel_path",
        type=str,
        required=True,
        help="ptq 保存的 .pt（含 model state_dict）",
    )
    p.add_argument("--seq_len", type=int, default=2048, help="ModelAnalyzer 用 seqlen，与训练时一致即可")
    p.add_argument("--lm_eval_batch_size", type=int, default=32)
    p.add_argument(
        "--placement",
        type=str,
        default="auto",
        choices=("auto", "single", "dispatch"),
        help="模型上 GPU 方式：auto=多可见卡时 accelerate 切分否则单卡；dispatch=始终切分；single=整模 cuda:0",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="日志目录；默认放在 checkpoint 同目录下 lm_eval_only/logs",
    )
    p.add_argument(
        "--rotate",
        action="store_true",
        help="量化时若使用了 --rotate（SpinQuant 风格），评估此 ckpt 时必须打开",
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
        logging.error("找不到 checkpoint: %s", args.load_qmodel_path)
        sys.exit(1)

    logging.info("Loading base model from %s", args.model)
    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    model.cpu()
    remove_hook_from_module(model, recurse=True)

    logging.info("add_actquant (structure must match training)")
    quant_utils.add_actquant(analyzer)
    if args.rotate:
        logging.info("rotate=1: configure down_proj online Hadamard (match ptq_gptaq_dp.py)")
        _configure_rotate_down_proj(model)

    logging.info("Loading weights from %s", args.load_qmodel_path)
    # PyTorch>=2.6 默认 weights_only=True，本仓库 ckpt 含 WeightQuantizer 等需完整反序列化
    try:
        save_dict = torch.load(
            args.load_qmodel_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        save_dict = torch.load(args.load_qmodel_path, map_location="cpu")
    if "model" not in save_dict:
        logging.error("checkpoint 中缺少 key 'model'")
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