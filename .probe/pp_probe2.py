"""Probe #2: is torch.distributed.pipelining's real 2-stage GPipe schedule
NUMERICALLY correct on CPU/gloo vs a non-pipelined single-process reference,
for a tiny 2-block "transformer-like" stack (mirrors splitting TinyGPT's
blocks across 2 ranks)?
"""
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return x + torch.relu(self.fc2(torch.relu(self.fc1(x))))


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29704"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    d = 8
    torch.manual_seed(42)
    blocks = [Block(d) for _ in range(world_size)]  # one block per stage, same seed => same weights every rank

    x_full = torch.randn(4, d)  # replicated "input batch" -- same on every rank (same seed)

    # Reference: plain sequential forward through ALL blocks, single process.
    with torch.no_grad():
        ref = x_full
        for b in blocks:
            ref = b(ref)

    my_stage_mod = blocks[rank]
    stage = PipelineStage(my_stage_mod, stage_index=rank, num_stages=world_size, device=torch.device("cpu"))
    schedule = ScheduleGPipe(stage, n_microbatches=2)

    with torch.no_grad():
        if rank == 0:
            schedule.step(x_full)
            out = None
        else:
            out = schedule.step()

    if rank == world_size - 1:
        match = torch.allclose(out, ref, atol=1e-6, rtol=1e-5)
        print(f"[rank {rank}] PP forward matches reference: {match} "
              f"(max abs diff={(out-ref).abs().max().item():.3e})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
