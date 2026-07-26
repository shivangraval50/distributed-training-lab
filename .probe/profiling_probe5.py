"""Probe real DTensor/TP collective op names on CPU/gloo (train_tp.py's
RowwiseParallel triggers an implicit all_reduce during backward) -- confirms
whether TP shows up under the same c10d::allreduce_/gloo:all_reduce names as
DDP, or something DTensor-specific."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from src.tensor_parallel import TPTinyGPT, build_tp_plan


def worker(rank: int, world_size: int, port: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    torch.manual_seed(0)
    model = TPTinyGPT(vocab_size=32, block_size=16, d_model=16, n_layer=1, n_head=2)
    mesh = DeviceMesh("cpu", list(range(world_size)))
    parallelize_module(model, mesh, build_tp_plan(1))
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randint(0, 32, (2, 16))
    y = torch.randint(0, 32, (2, 16))
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(2):
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(3):
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            opt.step()

    if rank == 0:
        events = prof.key_averages()
        print(f"\n=== rank {rank}: TP event names containing comm-ish substrings ===")
        for e in sorted(events, key=lambda e: -e.cpu_time_total):
            name_l = e.key.lower()
            if any(k in name_l for k in ["reduce", "gather", "scatter", "broadcast", "send", "recv", "c10d", "gloo"]):
                print(f"{e.key!r:45s} count={e.count:4d} self_us={e.self_cpu_time_total:9.1f} total_us={e.cpu_time_total:9.1f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2, 29516), nprocs=2, join=True)
