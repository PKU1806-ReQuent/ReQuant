# coding=utf-8
"""
PTQ driver for GPTAQ-Refined with data-parallel Hessian / dXXT (multi-GPU).

Launch example::

    torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=localhost:29500 \\
        ptq_gptaq_refined_dp.py --model ... --w_method gptaq_refined ...

``--w_method`` must stay ``gptaq_refined`` (same checkpoints as single-GPU path).
"""

import logging

import torch.distributed as dist

from process_args import parse_gen
from gptq_utils.quantize_gptaq_refined_dp import quantize_weights_gptaq_refined_dp
from utils import data_utils, dist_utils, eval_utils, model_utils, rotation_utils, \
                  memory_utils, quant_utils, hadamard_utils


def main(args):
    dist_utils.init_process_group()

    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model
    tokenizer = analyzer.tokenizer

    if args.w_method != "gptaq_refined":
        logging.warning(
            "ptq_gptaq_refined_dp is intended for w_method=gptaq_refined; got %s",
            args.w_method,
        )

    test_loader_dict, ref_logits_dict = {}, {}
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

    if args.rotate:
        rotation_utils.fuse_layer_norms(analyzer)
        rotation_utils.rotate_model(args, analyzer)
        memory_utils.cleanup_memory()

        quant_utils.add_actquant(analyzer)
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = False
    else:
        quant_utils.add_actquant(analyzer)

    model = quantize_weights_gptaq_refined_dp(args, analyzer)

    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            if "v_proj" in name and args.v_bits < 16:
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "lm_head" in name:
                layer_input_bits = 16

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        rope_function_name = "apply_rotary_pos_emb"
        layers = analyzer.get_layers()
        k_quant_config = {
            "k_bits": args.k_bits,
            "k_groupsize": args.k_groupsize,
            "k_sym": not (args.k_asym),
            "k_clip_ratio": args.k_clip_ratio,
        }
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                head_dim=analyzer.head_dim,
                **k_quant_config,
            )

    eval_utils.kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict)
    del orig_lm_head, ref_logits_dict

    if args.lm_eval:
        # 仅 rank0 跑 harness，避免 torchrun 多进程各跑一遍 lm_eval
        if dist_utils.is_main():
            dist_utils.distribute_model(model)
            eval_utils.qa_eval(model, tokenizer, args.lm_eval_batch_size)

    dist.barrier()


if __name__ == "__main__":
    args = parse_gen()
    main(args)
