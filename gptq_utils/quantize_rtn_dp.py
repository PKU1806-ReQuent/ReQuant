# coding=utf-8
"""
Weight quantization entry for RTN with distributed layer-parallel execution.

Use with ``torchrun --nproc_per_node=R`` and ``ptq_rtn_dp.py``.
"""

import logging
import os

import torch
import transformers
from accelerate.hooks import remove_hook_from_module

from gptq_utils.rtn_dp_utils import resolve_local_cuda_device, rtn_fwrd_data_parallel
from utils import dist_utils, model_utils


def quantize_weights_rtn_dp(args, analyzer: model_utils.ModelAnalyzer, dataloader=None):
    transformers.set_seed(args.seed)

    model = analyzer.model
    if not model_utils.keep_model_on_load_device():
        model.cpu()
    remove_hook_from_module(model, recurse=True)

    if args.w_bits >= 16:
        return model

    save_dict = {}
    if args.load_qmodel_path:
        logging.info("Load quantized model from %s", args.load_qmodel_path)
        save_dict = torch.load(args.load_qmodel_path)
        model.load_state_dict(save_dict["model"])
    else:
        dev = resolve_local_cuda_device()
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        quantizers = rtn_fwrd_data_parallel(args, analyzer, dev, dataloader=dataloader)
        save_dict["w_quantizers"] = quantizers

    if args.save_qmodel_path and dist_utils.is_main():
        os.makedirs(os.path.dirname(args.save_qmodel_path) or ".", exist_ok=True)
        save_dict["model"] = model.state_dict()
        torch.save(save_dict, args.save_qmodel_path)

    return model
