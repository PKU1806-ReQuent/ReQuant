from typing import Optional, Any
import datetime
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from utils import memory_utils


def init_process_group():
    # Bind each rank to its local GPU before NCCL initialization.
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    dist.barrier()


def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_available_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_available_and_initialized():
        return dist.get_rank()
    return 0


def is_main():
    return get_rank() == 0


def broadcast_parameters(module: nn.Module, src: Any = 0, group: Optional[Any] = None):
    for param in module.parameters():
        # NCCL broadcast requires contiguous tensors.
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
        dist.broadcast(param.data, src=src, group=group)


def gather_into_tensor(tensor, dim: int = 0):
    world_size = get_world_size()
    if is_main():
        gathered_shape = (*tensor.shape[:dim], world_size * tensor.shape[dim], *tensor.shape[dim + 1 :])
        gathered_tensor = torch.empty(gathered_shape, device=tensor.device, dtype=tensor.dtype)
        gathered_tensor_chunks = list(gathered_tensor.chunk(world_size, dim=dim))
    else:
        gathered_tensor = None
        gathered_tensor_chunks = None
    dist.gather(tensor, gathered_tensor_chunks)
    return gathered_tensor


def print_on_main(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def distribute_model(model) -> None:
    # Qwen3MoeForCausalLM is the top-level model class, not a decoder layer;
    # no_split should only list per-layer classes to prevent splitting a layer across devices.
    no_split_module_classes = ["Qwen3DecoderLayer", "Qwen3MoeDecoderLayer", "LlamaDecoderLayer"]
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    # Do NOT pass state_dict here: the model is already on CPU with correct weights.
    # Passing state_dict=model.state_dict() creates a redundant full-weight copy (~16 GB
    # for 8B models) before dispatch_model moves tensors to their target devices.
    # offload_buffers=False: keep all buffers (ActQuantWrapper scale/zero/maxq etc.) on GPU;
    # offloading them to disk causes device mismatches and numerical differences vs single-GPU eval.
    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=False,
    )
    memory_utils.cleanup_memory()
