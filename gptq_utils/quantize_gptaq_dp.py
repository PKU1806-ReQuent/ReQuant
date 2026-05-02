# coding=utf-8
"""
Weight quantization entry for GPTQ/GPTAQ (+ optional ReQuant) with data-parallel
Hessian accumulation.

Use with ``torchrun --nproc_per_node=R`` and ``ptq_dp.py``.

``use_requant=True`` (auto when ``--enable_requant`` is set) instantiates
``GPTQReQuant`` (Phase-1 P-matrix + optional Phase-2 coordinate descent);
otherwise ``GPTAQ`` (Phase-1 only).
"""

import logging
import os

import torch
import transformers
from accelerate.hooks import remove_hook_from_module

from gptq_utils.dp_common import resolve_local_cuda_device
from gptq_utils.gptq_dp_utils import gptq_fwrd_data_parallel
from utils import data_utils, dist_utils, model_utils


def quantize_weights_gptaq_dp(args, analyzer: model_utils.ModelAnalyzer):
    """
    Same contract as ``gptq_utils.main.quantize_weights`` for
    ``w_method in {gptq,gptaq}`` with optional ``--enable_requant``, but calibration uses
    ``gptq_fwrd_data_parallel``. Checkpoint is written only on global rank 0.
    """
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

        use_requant = bool(getattr(args, "enable_requant", False))
        quantizers = gptq_fwrd_data_parallel(
            args, analyzer, trainloader, dev, use_requant=use_requant
        )
        save_dict["w_quantizers"] = quantizers

    if args.save_qmodel_path and dist_utils.is_main():
        os.makedirs(os.path.dirname(args.save_qmodel_path) or ".", exist_ok=True)
        save_dict["model"] = model.state_dict()
        torch.save(save_dict, args.save_qmodel_path)

    return model
