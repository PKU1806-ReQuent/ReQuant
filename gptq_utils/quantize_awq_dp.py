# coding=utf-8
"""
Weight quantization entry for AWQ with data-parallel activation-stat collection.

Use with ``torchrun --nproc_per_node=R`` and ``ptq_awq_dp.py``.
"""

import logging
import os

import torch
import transformers
from accelerate.hooks import remove_hook_from_module

from gptq_utils.awq_dp_utils import awq_fwrd_data_parallel, resolve_local_cuda_device
from utils import data_utils, dist_utils, model_utils


def quantize_weights_awq_dp(args, analyzer: model_utils.ModelAnalyzer):
    transformers.set_seed(args.seed)

    model = analyzer.model
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
        trainloader = data_utils.get_tokens(
            args.dataset,
            "train",
            analyzer.tokenizer,
            args.seq_len,
            args.nsamples,
            args.tokens_cache_path,
            args.seed,
        )

        if isinstance(trainloader[0], torch.Tensor):
            assert trainloader[0].dim() == 1
            logging.info("Reformatting input tokens to tuple + Unsqueeze")
            trainloader = [(x.unsqueeze(0), None) for x in trainloader]

        dev = resolve_local_cuda_device()
        if dev.type == "cuda":
            torch.cuda.set_device(dev)

        quantizers = awq_fwrd_data_parallel(args, analyzer, trainloader, dev)
        save_dict["w_quantizers"] = quantizers

    if args.save_qmodel_path and dist_utils.is_main():
        os.makedirs(os.path.dirname(args.save_qmodel_path) or ".", exist_ok=True)
        save_dict["model"] = model.state_dict()
        torch.save(save_dict, args.save_qmodel_path)

    return model
