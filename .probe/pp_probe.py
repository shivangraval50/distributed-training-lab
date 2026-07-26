"""Probe: does torch.distributed.pipelining (PipelineStage + ScheduleGPipe)
work on CPU/gloo (no CUDA)? Tries a tiny 2-stage split of a 2-layer MLP
across 2 processes."""
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29703"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    d = 8
    torch.manual_seed(0)

    class Stage(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc = nn.Linear(d, d)

        def forward(self, x):
            return torch.relu(self.fc(x))

    stage_mod = Stage(d)
    x = torch.randn(4, d)

    try:
        stage = PipelineStage(
            stage_mod,
            stage_index=rank,
            num_stages=world_size,
            device=torch.device("cpu"),
        )
        schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=nn.MSELoss())
        if rank == 0:
            schedule.step(x)
        else:
            target = torch.randn(4, d)
            out = schedule.step(target=target)
            print(f"[rank {rank}] pipelining step OK, out shape {None if out is None else out.shape}")
        print(f"[rank {rank}] torch.distributed.pipelining ran without error")
    except Exception as e:
        print(f"[rank {rank}] torch.distributed.pipelining FAILED: {type(e).__name__}: {e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
