"""Probe FSDP's real op names (all_gather / reduce_scatter) on CPU/gloo, same
method as profiling_probe.py, so src/profiling.py's comm-keyword list is
built from what this torch version actually emits, not assumption."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import ProfilerActivity, profile


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32))
    fsdp_model = FSDP(model, device_id=torch.device("cpu"))
    opt = torch.optim.SGD(fsdp_model.parameters(), lr=0.01)
    x = torch.randn(4, 32)
    y = torch.randn(4, 32)
    for _ in range(2):
        opt.zero_grad()
        loss = ((fsdp_model(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            opt.zero_grad()
            loss = ((fsdp_model(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
    if rank == 0:
        events = prof.key_averages()
        print(f"\n=== rank {rank}: {len(events)} distinct FSDP event names ===")
        for e in sorted(events, key=lambda e: -e.cpu_time_total):
            print(f"{e.key!r:55s} count={e.count:4d} self_us={e.self_cpu_time_total:9.1f} total_us={e.cpu_time_total:9.1f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29514), nprocs=2, join=True)
