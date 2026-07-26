"""Follow-up probe: inspect raw profiler events (thread id, time range,
self_cpu) for the allreduce-related event names found by profiling_probe.py,
to understand why 'gloo:all_reduce' shows self_cpu_time_total=0 in
key_averages (nested/cross-thread event, need self time from the raw event
directly rather than the aggregated key_averages view)."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    model = nn.Linear(16, 16)
    ddp_model = DDP(model)
    opt = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    x = torch.randn(4, 16)
    y = torch.randn(4, 16)
    for _ in range(2):
        opt.zero_grad()
        loss = ((ddp_model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            opt.zero_grad()
            loss = ((ddp_model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
    if rank == 0:
        for e in prof.events():
            name_l = e.name.lower()
            if "allreduce" in name_l or "all_reduce" in name_l:
                print(
                    f"name={e.name!r} thread={e.thread} "
                    f"time_range={e.time_range} self_cpu_time_total={e.self_cpu_time_total}"
                )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29513), nprocs=2, join=True)
