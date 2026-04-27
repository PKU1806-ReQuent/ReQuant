# coding=utf-8
"""
Run lm-eval tasks on full-precision (non-quantized) model.

Example:
    python eval_fp_lm_tasks.py \
        --model ./modelzoo/Qwen3/Qwen3-4B \
        --lm_eval_batch_size 16
"""

import argparse
import logging
import os

import torch
import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.models.huggingface import eval_logger

from utils import data_utils, dist_utils, eval_utils, model_utils
from utils.log_utils import init_logging

eval_logger.level = logging.ERROR


DEFAULT_TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "ceval-valid",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "social_iqa",
    "winogrande",
]


def parse_args():
    p = argparse.ArgumentParser(description="Run lm-eval on full-precision model")
    p.add_argument("--model", type=str, required=True, help="HF model path")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--lm_eval_batch_size", type=int, default=32)
    p.add_argument(
        "--placement",
        type=str,
        default="auto",
        choices=("auto", "single", "dispatch"),
        help="GPU placement: auto/ single/ dispatch",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task names",
    )
    p.add_argument(
        "--eval_mode",
        type=str,
        default="ppl",
        choices=("qa", "ppl", "both"),
        help="qa: lm-eval acc tasks; ppl: KL/PPL on eval_datasets; both: run both",
    )
    p.add_argument(
        "--eval_datasets",
        type=str,
        default="wikitext2,ultrachat_2k,numinamath",
        help="Comma-separated datasets for KL/PPL eval",
    )
    p.add_argument(
        "--eval_seq_len",
        type=int,
        default=2048,
        help="Sequence length for KL/PPL eval",
    )
    p.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Cache directory used by KL/PPL eval",
    )
    p.add_argument(
        "--ppl_chunk_size",
        type=int,
        default=16,
        help="Token chunk size for GPU PPL CE computation",
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Default: outputs/<model_name>/fp_lm_eval/logs",
    )
    return p.parse_args()


def pretty_print_results(data):
    headers = list(data.keys())
    values = [str(v) for v in data.values()]
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_row = "| " + " | ".join(values) + " |"
    logging.info("\n%s\n%s\n%s", header_row, separator_row, data_row)


def run_lm_eval(model, tokenizer, tasks, batch_size):
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

    custom_task_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "datasets", "lm_eval_configs", "tasks")
    )
    tm_kwargs = {"include_defaults": True}
    if os.path.isdir(custom_task_dir):
        tm_kwargs["include_path"] = custom_task_dir
    task_manager = lm_eval.tasks.TaskManager(**tm_kwargs)

    task_names = lm_eval_utils.pattern_match(tasks, task_manager.all_tasks)
    if not task_names:
        logging.warning("No task matched: %s", tasks)
        return

    logging.info("Resolved tasks: %s", task_names)

    results = {}
    metric_kind = {}
    acc_values = []
    ppl_values = []
    for task_name in task_names:
        logging.info("Evaluating %s ...", task_name)
        current_bs = max(1, int(batch_size))
        while True:
            hflm.batch_size_per_gpu = current_bs
            try:
                out = lm_eval.simple_evaluate(
                    hflm,
                    tasks=[task_name],
                    task_manager=task_manager,
                )["results"][task_name]
                break
            except (torch.OutOfMemoryError, RuntimeError) as e:
                is_oom = isinstance(e, torch.OutOfMemoryError) or (
                    "out of memory" in str(e).lower()
                )
                if not is_oom:
                    raise
                if current_bs <= 1:
                    logging.exception(
                        "OOM on task %s even with batch_size=1, aborting.",
                        task_name,
                    )
                    raise
                next_bs = max(1, current_bs // 2)
                logging.warning(
                    "OOM on task %s with batch_size=%d, retry with batch_size=%d",
                    task_name,
                    current_bs,
                    next_bs,
                )
                current_bs = next_bs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if "acc_norm,none" in out or "acc,none" in out:
            acc = round(out.get("acc_norm,none", out["acc,none"]) * 100, 2)
            results[task_name] = acc
            metric_kind[task_name] = "acc"
            acc_values.append(acc)
            logging.info("%s acc=%.2f%%", task_name, acc)
            continue

        ppl_key = None
        for candidate in ("word_perplexity,none", "perplexity,none"):
            if candidate in out:
                ppl_key = candidate
                break
        if ppl_key is None:
            for k in out:
                if "perplexity" in k.lower() or "ppl" in k.lower():
                    ppl_key = k
                    break
        if ppl_key is not None:
            ppl = float(out[ppl_key])
            results[task_name] = ppl
            metric_kind[task_name] = "ppl"
            ppl_values.append(ppl)
            logging.info("%s %s=%.4f", task_name, ppl_key, ppl)
            continue

        raise KeyError(
            f"Unsupported metric keys for task {task_name}: {list(out.keys())}"
        )

    results_str = {}
    for task in task_names:
        if task not in results:
            continue
        if metric_kind.get(task) == "acc":
            results_str[task] = f"{results[task]:.2f}"
        else:
            results_str[task] = f"{results[task]:.4f}"
    if acc_values:
        results_str["acc_avg"] = f"{sum(acc_values) / len(acc_values):.2f}"
    if ppl_values:
        results_str["ppl_avg"] = f"{sum(ppl_values) / len(ppl_values):.4f}"
    pretty_print_results(results_str)


@torch.no_grad()
def run_kl_ppl_eval(args, analyzer):
    eval_datasets = [x.strip() for x in args.eval_datasets.split(",") if x.strip()]
    if not eval_datasets:
        logging.warning("No eval datasets configured for PPL.")
        return
    args.eval_datasets = eval_datasets
    args.model_name = os.path.basename(os.path.abspath(args.model.rstrip("/")))

    metric_vals = {}
    model = analyzer.model
    orig_device = next(model.parameters()).device
    dev = torch.device("cuda") if torch.cuda.is_available() else orig_device
    ppl_dev = dev

    for eval_dataset in args.eval_datasets:
        logging.info("Evaluating PPL on %s", eval_dataset)
        test_loader = data_utils.get_loaders(
            eval_dataset,
            split="test",
            tokenizer=analyzer.tokenizer,
            seq_len=args.eval_seq_len,
            num_samples=1,
        )
        logits_list, input_ids = eval_utils._get_logits(args, analyzer, test_loader, dev)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.lm_head.to(ppl_dev)
        nlls = []
        for logits, ids in zip(logits_list, input_ids):
            # Compute NLL in sequence chunks to avoid allocating full [seq_len, vocab] logits at once.
            hidden = logits.to(ppl_dev)
            labels = ids.to(ppl_dev)
            chunk_nlls = []
            chunk_size = max(1, int(args.ppl_chunk_size))
            for start in range(0, max(0, hidden.shape[0] - 1), chunk_size):
                end = min(hidden.shape[0] - 1, start + chunk_size)
                if end <= start:
                    continue
                chunk_logits = model.lm_head(hidden[start:end])
                chunk_labels = labels[start + 1 : end + 1]
                chunk_loss = torch.nn.functional.cross_entropy(
                    chunk_logits,
                    chunk_labels,
                    reduction="none",
                )
                chunk_nlls.append(chunk_loss.float())
                del chunk_logits, chunk_labels, chunk_loss
            if not chunk_nlls:
                continue
            nlls.append(torch.cat(chunk_nlls).mean().unsqueeze(0))
            del hidden, labels, chunk_nlls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ppl = torch.exp(torch.cat(nlls).mean()).item()
        metric_vals[f"PPL-{eval_dataset}"] = f"{ppl:.2f}"
        logging.info("PPL on %s: %.2f", eval_dataset, ppl)

    model.lm_head.to(orig_device)
    pretty_print_results(metric_vals)


def main():
    args = parse_args()
    model_name = os.path.basename(os.path.abspath(args.model.rstrip("/")))
    log_dir = args.log_dir or os.path.join("outputs", model_name, "fp_lm_eval", "logs")
    os.makedirs(log_dir, exist_ok=True)
    init_logging(log_dir)

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    logging.info("Loading full-precision model from %s", args.model)
    analyzer = model_utils.ModelAnalyzer(args.model, args.seq_len)
    model = analyzer.model.eval()

    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        use_dispatch = args.placement == "dispatch" or (
            args.placement == "auto" and n_gpu > 1
        )
        if use_dispatch:
            logging.info("dispatch_model across %d visible GPU(s)", n_gpu)
            dist_utils.distribute_model(model)
        else:
            model.cuda()
            logging.info("Model moved to single CUDA device")

    if args.eval_mode in ("ppl", "both"):
        run_kl_ppl_eval(args, analyzer)
    if args.eval_mode in ("qa", "both"):
        run_lm_eval(model, analyzer.tokenizer, tasks, args.lm_eval_batch_size)


if __name__ == "__main__":
    main()
