"""Probe raw dist.send/recv op names on CPU/gloo (used by train_pp.py's real
PipelineStage under the hood) so src/profiling.py's comm-keyword list covers
PP's send/recv pattern too, not just DDP's allreduce / FSDP's
all_gather+reduce_scatter."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            t = torch.randn(8, 8)
            if rank == 0:
                dist.send(t, dst=1)
            else:
                dist.recv(t, src=0)

    if rank == 0:
        events = prof.key_averages()
        print(f"\n=== rank {rank} (sender) send-related events ===")
        for e in sorted(events, key=lambda e: -e.cpu_time_total):
            print(f"{e.key!r:40s} count={e.count:4d} self_us={e.self_cpu_time_total:9.1f} total_us={e.cpu_time_total:9.1f}")
    dist.barrier()
    if rank == 1:
        events = prof.key_averages()
        print(f"\n=== rank {rank} (receiver) recv-related events ===")
        for e in sorted(events, key=lambda e: -e.cpu_time_total):
            print(f"{e.key!r:40s} count={e.count:4d} self_us={e.self_cpu_time_total:9.1f} total_us={e.cpu_time_total:9.1f}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29515), nprocs=2, join=True)
