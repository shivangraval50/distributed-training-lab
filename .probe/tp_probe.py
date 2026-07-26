"""Probe: does real torch.distributed.tensor.parallel (DTensor-based TP) work
on CPU/gloo? Tries ColwiseParallel+RowwiseParallel on a tiny 2-layer MLP with
a DeviceMesh("cpu", ...) across 2 processes spawned locally."""
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29701"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    torch.manual_seed(0)

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    d = 8
    model = MLP(d)

    try:
        mesh = DeviceMesh("cpu", list(range(world_size)))
        plan = {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
        tp_model = parallelize_module(model, mesh, plan)
        x = torch.randn(2, d)
        out = tp_model(x)
        print(f"[rank {rank}] TP forward OK, out shape {out.shape}")
    except Exception as e:
        print(f"[rank {rank}] TP FAILED: {type(e).__name__}: {e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
