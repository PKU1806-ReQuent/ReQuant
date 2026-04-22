import logging
import os
import copy
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import lm_eval
from lm_eval import utils as lm_eval_utils
from lm_eval.models.huggingface import HFLM
from lm_eval.models.huggingface import eval_logger
eval_logger.level = logging.ERROR
from datasets.exceptions import DatasetNotFoundError

from utils import memory_utils, dist_utils, model_utils


@torch.no_grad()
def _get_logits(args, analyzer: model_utils.ModelAnalyzer, testenc, dev):
    model = analyzer.model
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = analyzer.get_layers()
    orig_device = next(model.parameters()).device

    input_ids = testenc.input_ids  
    nsamples = input_ids.numel() // args.eval_seq_len
    input_ids = input_ids[:, :nsamples * args.eval_seq_len].view(nsamples, args.eval_seq_len).to(dev)

    for module in analyzer.get_pre_block_modules():
        module.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, args.eval_seq_len, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        try:
            model(input_ids[i: i+1])
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].to(orig_device)
    memory_utils.cleanup_memory()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    position_embeddings = cache["position_embeddings"]

    for i in tqdm(range(len(layers)), ncols=80, desc="Forwarding Layers"):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        layers[i] = layer.to(orig_device)
        del layer
        memory_utils.cleanup_memory()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    memory_utils.cleanup_memory()

    # Get model logits
    model.model.norm.to(dev)
    lm_logits = []
    for i in range(nsamples):
        hidden_states = inps[i: i + 1].to(dev)
        hidden_states = model.model.norm(hidden_states)
        lm_logits.append(hidden_states.cpu())
    lm_logits = torch.cat(lm_logits, dim=0)
    model.model.norm.to(orig_device)

    for module in analyzer.get_pre_block_modules():
        module.to(orig_device)
    memory_utils.cleanup_memory(verbos=True)

    return lm_logits, input_ids


@torch.no_grad()
def get_ref_logits(args, analyzer, dataset, dataloader):
    cache_dir = os.path.join(args.cache_dir, "ref_logits")
    os.makedirs(cache_dir, exist_ok=True)
    ref_logits_path = f'{cache_dir}/{args.model_name}_{dataset}_test_{args.eval_seq_len}.cache'
    if not os.path.exists(ref_logits_path):
        logging.info(f"Generating reference logits for {dataset}...")
        ref_logits, _ = _get_logits(args, analyzer, dataloader, torch.device("cuda"))
        if dist_utils.is_main():
            torch.save(ref_logits, ref_logits_path)
    else:
        logging.info(f"Loading reference logits for {dataset}...")
        ref_logits = torch.load(ref_logits_path).cpu()
    orig_lm_head = copy.deepcopy(analyzer.model.lm_head)
    memory_utils.cleanup_memory()
    return ref_logits, orig_lm_head


@torch.no_grad()
def _kl_ppl_eval(args, analyzer: model_utils.ModelAnalyzer, orig_lm_head, dataloader, ref_logits_list):
    model = analyzer.model
    orig_device = next(model.parameters()).device

    dev = torch.device("cuda")

    logits_list, input_ids_list = _get_logits(args, analyzer, dataloader, dev)
    model.lm_head.to(dev)
    orig_lm_head.to(dev)

    kl_loss = 0
    nlls = []
    for logits, ref_logits, input_ids in tqdm(zip(logits_list, ref_logits_list, input_ids_list), ncols=80,
                                              total=len(ref_logits_list), desc="Computing PPL & KL"):
        logits = model.lm_head(logits.to(dev))
        ref_logits = orig_lm_head(ref_logits.to(dev))

        # NLL loss
        shift_labels = input_ids[None, 1:].to(dev)
        shift_logits = logits[None, :-1, :].to(dev)
        loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels,
                               reduction="none")
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)

        # topk kl loss
        k = 20
        ref_logits, indices = ref_logits.topk(k, dim=-1, sorted=False)
        logits = logits.gather(-1, indices)
        loss = F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.softmax(ref_logits, dim=-1),
            reduction="none",
        )
        kl_loss += loss.float().sum(-1).mean()
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    kl_loss /= len(ref_logits_list)

    model.lm_head.to(orig_device)
    orig_lm_head.to(orig_device)
    memory_utils.cleanup_memory()

    return ppl.item(), kl_loss.item()


def kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict, ref_logits_dict):
    metric_vals = OrderedDict()
    for eval_dataset in args.eval_datasets:
        logging.info(f"Evaluating KL&PPL on {eval_dataset}")
        ppl, kl_loss = _kl_ppl_eval(args, analyzer, orig_lm_head, test_loader_dict[eval_dataset], ref_logits_dict[eval_dataset])
        metric_vals[f"KL-{eval_dataset}"] = f"{kl_loss:.2e}"
        metric_vals[f"PPL-{eval_dataset}"] = f"{ppl:.2f}"
        logging.info(f"KL&PPL on {eval_dataset}: {kl_loss:.2e}, {ppl:.2f}")
    pretty_print_results(metric_vals)


def qa_eval(model, tokenizer, lm_eval_batch_size=32):
    # Keep evaluation output concise: hide noisy progress bars and metadata warnings.
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for logger_name in (
        "lm_eval",
        "lm_eval.evaluator",
        "lm_eval.models.huggingface",
        "datasets",
        "huggingface_hub",
        "transformers",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from datasets.utils.logging import disable_progress_bar
        disable_progress_bar()
    except Exception:
        pass

    class _QuietEvalFilter(logging.Filter):
        _blocked = (
            "Failed to get model SHA",
            "fatal: not a git repository",
            "Stopping at filesystem boundary",
        )

        def filter(self, record):
            msg = record.getMessage()
            return not any(x in msg for x in self._blocked)

    _quiet_filter = _QuietEvalFilter()
    root_logger = logging.getLogger()
    root_logger.addFilter(_quiet_filter)
    for _h in root_logger.handlers:
        _h.addFilter(_quiet_filter)

    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=lm_eval_batch_size)
    # Prevent lm_eval from stringifying full model structure when collecting model SHA metadata.
    hflm.pretrained = getattr(model.config, "_name_or_path", "local-preinitialized-model")

    tasks = [
        "piqa",
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "openbookqa",
        "social_iqa",
        "boolq",
        "lambada_openai",
        "ceval-valid",
    ]
    # include_defaults=True: built-in harness tasks (piqa, hellaswag, ...). Optional ./datasets/lm_eval_configs/tasks
    # for project-specific YAMLs when that directory exists (include_defaults=False alone yields empty all_tasks).
    _custom_task_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets", "lm_eval_configs", "tasks"))
    _tm_kwargs = {"include_defaults": True}
    if os.path.isdir(_custom_task_dir):
        _tm_kwargs["include_path"] = _custom_task_dir
    task_manager = lm_eval.tasks.TaskManager(**_tm_kwargs)
    task_names = lm_eval_utils.pattern_match(tasks, task_manager.all_tasks)
    results, results_str = {}, {}
    for task_name in task_names:
        logging.info(f"Evaluating {task_name}...")
        hflm.batch_size_per_gpu = lm_eval_batch_size
        try:
            result = lm_eval.simple_evaluate(
                hflm, tasks=[task_name], task_manager=task_manager
            )["results"]
        except (
            ConnectionError,
            RequestsConnectionError,
            DatasetNotFoundError,
            FileNotFoundError,
            OSError,
        ) as e:
            logging.warning(
                "Skipping lm_eval task '%s' due to dataset/load issue: %s",
                task_name,
                e,
            )
            continue
        result = result[task_name]
        acc = round(result.get('acc_norm,none', result['acc,none']) * 100, 2)
        results[task_name] = acc
        logging.info(f"acc: {acc}%")
    results_str.update({task: f"{result:.2f}" for task, result in results.items()})
    if results:
        results_str["acc_avg"] = f"{sum(results.values()) / len(results):.2f}"
    else:
        logging.warning(
            "lm_eval: no successful task results; skip acc_avg. "
            "Check network/cache/task names or add YAMLs under datasets/lm_eval_configs/tasks.",
            tasks,
        )
    if results_str:
        pretty_print_results(results_str)


def pretty_print_results(data):
    headers = list(data.keys())
    values = [str(v) for v in data.values()]

    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_row = "| " + " | ".join(values) + " |"

    logging.info("\n" + header_row + "\n" + separator_row + "\n" + data_row)