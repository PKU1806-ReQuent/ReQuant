# coding=utf-8
"""Shared helpers for data-parallel PTQ flows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.distributed as dist

from utils import memory_utils


@dataclass
class DPCalibrationState:
    local_indices: List[int]
    shard_inputs: bool
    attention_mask: Any
    position_ids: Any
    position_embeddings: Any
    inps: Optional[torch.Tensor] = None
    fp_inps: Optional[torch.Tensor] = None
    inps_shard: Optional[torch.Tensor] = None
    fp_inps_shard: Optional[torch.Tensor] = None


def resolve_local_cuda_device() -> torch.device:
    """Return the torchrun-local CUDA device."""
    return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")


def move_to_device(x: Any, device: torch.device) -> Any:
    """Move tensor leaves of a nested tuple/list to ``device``."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(move_to_device(t, device) for t in x)
    if isinstance(x, list):
        return [move_to_device(t, device) for t in x]
    return x


def local_sample_indices(rank: int, world_size: int, nsamples: int) -> List[int]:
    base, rem = divmod(nsamples, world_size)
    start = rank * base + min(rank, rem)
    count = base + (1 if rank < rem else 0)
    return list(range(start, start + count))


def capture_first_layer_inputs(model, layers, storage, dataloader, dev):
    """Run calibration samples until the first decoder block and capture inputs."""
    cache = {"i": 0}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "attention_type"):
                self.attention_type = module.attention_type

        def forward(self, inp, **kwargs):
            storage[cache["i"]] = (
                inp.detach().cpu() if storage.device.type == "cpu" else inp
            )
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    return cache


def capture_calibration_state(
    args,
    model,
    layers,
    dataloader,
    dev: torch.device,
    rank: int,
    world_size: int,
    group: Optional[Any] = None,
    clone_fp: bool = False,
) -> DPCalibrationState:
    """Capture first-block inputs and attention metadata for DP layerwise PTQ."""
    local_indices = local_sample_indices(rank, world_size, args.nsamples)
    shard_inputs = bool(getattr(args, "dp_shard_inps", False) and world_size > 1)
    dtype = next(iter(model.parameters())).dtype
    hidden = model.config.hidden_size
    seqlen = model.seqlen

    if shard_inputs:
        inps_full_cpu = (
            torch.zeros((args.nsamples, seqlen, hidden), dtype=dtype, device="cpu")
            if rank == 0
            else None
        )
        cache: dict = {"i": 0}
        if rank == 0:
            assert inps_full_cpu is not None
            cache = capture_first_layer_inputs(model, layers, inps_full_cpu, dataloader, dev)
        dist.barrier(group=group)

        meta = (
            [
                cache.get("attention_mask"),
                cache.get("position_ids"),
                cache.get("position_embeddings"),
            ]
            if rank == 0
            else [None, None, None]
        )
        meta = [x.detach().cpu() if torch.is_tensor(x) else x for x in meta]
        dist.broadcast_object_list(meta, src=0, group=group)
        attention_mask = move_to_device(meta[0], dev)
        position_ids = move_to_device(meta[1], dev)
        position_embeddings = move_to_device(meta[2], dev)

        scatter_objs = None
        if rank == 0:
            assert inps_full_cpu is not None
            scatter_objs = [
                inps_full_cpu[local_sample_indices(r, world_size, args.nsamples)]
                .contiguous()
                for r in range(world_size)
            ]
        recv_list: List[Any] = [None]
        dist.scatter_object_list(recv_list, scatter_objs, src=0, group=group)
        storage_device = torch.device("cpu") if args.offload_inps else dev
        inps_shard = recv_list[0].to(storage_device)
        fp_inps_shard = inps_shard.clone() if clone_fp else None
        del recv_list, scatter_objs, inps_full_cpu
        memory_utils.cleanup_memory(False)
        return DPCalibrationState(
            local_indices=local_indices,
            shard_inputs=True,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            inps_shard=inps_shard,
            fp_inps_shard=fp_inps_shard,
        )

    inps = torch.zeros((args.nsamples, seqlen, hidden), dtype=dtype, device=dev)
    cache = capture_first_layer_inputs(model, layers, inps, dataloader, dev)
    if world_size > 1:
        dist.broadcast(inps, src=0, group=group)
    fp_inps = inps.clone() if clone_fp else None
    if args.offload_inps:
        inps = inps.cpu()
        if fp_inps is not None:
            fp_inps = fp_inps.cpu()

    return DPCalibrationState(
        local_indices=local_indices,
        shard_inputs=False,
        attention_mask=cache["attention_mask"],
        position_ids=cache["position_ids"],
        position_embeddings=cache["position_embeddings"],
        inps=inps,
        fp_inps=fp_inps,
    )


def run_layer_sample(
    layer,
    src: torch.Tensor,
    idx: int,
    dev: torch.device,
    attention_mask: Any,
    position_ids: Any,
    position_embeddings: Any,
    store: bool = True,
):
    out = layer(
        src[idx].unsqueeze(0).to(dev),
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )[0]
    if store:
        src[idx] = out.to(src.device)
    return out


def propagate_layer_inputs(
    args,
    layer,
    state: DPCalibrationState,
    dev: torch.device,
    rank: int,
    world_size: int,
    group: Optional[Any] = None,
) -> None:
    if state.shard_inputs:
        assert state.inps_shard is not None
        for k in range(len(state.local_indices)):
            run_layer_sample(
                layer,
                state.inps_shard,
                k,
                dev,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
            )
        return

    assert state.inps is not None
    for j in range(args.nsamples):
        if rank == 0:
            run_layer_sample(
                layer,
                state.inps,
                j,
                dev,
                state.attention_mask,
                state.position_ids,
                state.position_embeddings,
            )
        if world_size > 1:
            buf = (
                state.inps[j].to(dev)
                if rank == 0
                else torch.empty(state.inps[j].shape, dtype=state.inps.dtype, device=dev)
            )
            dist.broadcast(buf, src=0, group=group)
            state.inps[j] = buf.to(state.inps.device)
