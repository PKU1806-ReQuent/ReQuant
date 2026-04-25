# coding=utf-8
"""Shared outer driver for torchrun-based PTQ entry points."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import torch.distributed as dist

from utils import (
    data_utils,
    dist_utils,
    eval_utils,
    hadamard_utils,
    memory_utils,
    model_utils,
    quant_utils,
    rotation_utils,
)


def _prepare_rotation_and_actquant(args, analyzer, model) -> None:
    if args.rotate:
        rotation_utils.fuse_layer_norms(analyzer)
        rotation_utils.rotate_model(args, analyzer)
        memory_utils.cleanup_memory()

    quant_utils.add_actquant(analyzer)
    if not args.rotate:
        return

    for name, qlayer in quant_utils.find_qlayers(model).items():
        if "down_proj" not in name:
            continue
        had_k, k = hadamard_utils.get_hadK(model.config.intermediate_size)
        qlayer.online_full_had = True
        qlayer.had_K = had_k
        qlayer.K = k
        qlayer.fp32_had = False


def _load_eval_refs(args, analyzer, tokenizer):
    test_loader_dict, ref_logits_dict = {}, {}
    orig_lm_head = None
    for eval_dataset in args.eval_datasets:
        test_loader = data_utils.get_loaders(
            eval_dataset,
            split="test",
            tokenizer=tokenizer,
            seq_len=args.eval_seq_len,
            num_samples=args.nsamples,
        )
        ref_logits, orig_lm_head = eval_utils.get_ref_logits(
            args, analyzer, eval_dataset, test_loader
        )
        test_loader_dict[eval_dataset] = test_loader
        ref_logits_dict[eval_dataset] = ref_logits
    return test_loader_dict, ref_logits_dict, orig_lm_head


def _add_kv_cache_quant(args, analyzer) -> None:
    if args.k_bits >= 16:
        return
    k_quant_config = {
        "k_bits": args.k_bits,
        "k_groupsize": args.k_groupsize,
        "k_sym": not args.k_asym,
        "k_clip_ratio": args.k_clip_ratio,
    }
    for layer in analyzer.get_layers():
        rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            "apply_rotary_pos_emb",
            head_dim=analyzer.head_dim,
            **k_quant_config,
        )


def run_dp_ptq(
    args,
    quantize_fn: Callable,
    expected_methods: Iterable[str],
    entry_name: str,
):
    dist_utils.init_process_group()

    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    expected_methods = tuple(expected_methods)
    if args.w_method not in expected_methods:
        logging.warning(
            "%s expects w_method in %s; got %s",
            entry_name,
            expected_methods,
            args.w_method,
        )

    test_loader_dict, ref_logits_dict, orig_lm_head = _load_eval_refs(
        args, analyzer, tokenizer
    )

    _prepare_rotation_and_actquant(args, analyzer, model)
    if args.enable_aq_calibration:
        quant_utils.configure_actquant_from_args(args, model)

    model = quantize_fn(args, analyzer)

    if not args.enable_aq_calibration:
        quant_utils.configure_actquant_from_args(args, model)

    _add_kv_cache_quant(args, analyzer)
    eval_utils.kl_ppl_eval(
        args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict
    )
    del orig_lm_head, ref_logits_dict

    if args.lm_eval and dist_utils.is_main():
        dist_utils.distribute_model(model)
        eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)

    dist.barrier()
