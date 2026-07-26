"""Phase 5 Step 1 probe: what do torch.profiler event names ACTUALLY look
like for real gloo collectives on this machine (torch 2.13.0, macOS, CPU)?

Not guessing/hardcoding op names -- this script prints every captured event
name from a real 2-process DDP step (which triggers gloo allreduce under the
hood) so src/profiling.py's comm-vs-compute classifier can be built from
observed strings, not assumption.
"""
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

    # warmup (first DDP step does extra bucket-rebuild bookkeeping)
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
        events = prof.key_averages()
        print(f"\n=== rank {rank}: ALL {len(events)} distinct event names (name, count, self_cpu_us, total_cpu_us) ===")
        for e in sorted(events, key=lambda e: -e.cpu_time_total):
            print(
                f"{e.key!r:60s} count={e.count:4d} "
                f"self_cpu_time_total_us={e.self_cpu_time_total:10.1f} "
                f"cpu_time_total_us={e.cpu_time_total:10.1f}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29511), nprocs=2, join=True)
