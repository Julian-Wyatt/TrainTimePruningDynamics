import os

import torch
import torch.distributed as dist

from utils.device import get_device


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if dist.is_initialized():
        return dist.get_rank() == 0
    # Before init_process_group, fall back to torchrun's RANK so all N processes
    # don't each initialise W&B or write checkpoints.
    return int(os.environ.get("RANK", 0)) == 0


def reduce_dict(input_dict: dict) -> dict:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return input_dict
    world_size = dist.get_world_size()
    reduced = {}
    for k, v in input_dict.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            reduced[k] = v / world_size
        else:
            t = torch.tensor(v, dtype=torch.float32, device=get_device())
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            reduced[k] = (t / world_size).item()
    return reduced
