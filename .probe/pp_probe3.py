"""Probe #3: does torch.distributed.pipelining support a full train step
(forward + loss + backward) on CPU/gloo, with per-rank local .grad on the
pipelined stage's parameters matching a single-process reference's
gradients for the corresponding sub-module?
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
    os.environ["MASTER_PORT"] = "29705"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    d = 8
    torch.manual_seed(42)
    blocks = [Block(d) for _ in range(world_size)]

    torch.manual_seed(7)
    x_full = torch.randn(4, d)
    target_full = torch.randn(4, d)
    loss_fn = nn.MSELoss()

    # ---- reference: single-process forward+backward through ALL blocks ----
    ref_blocks = [Block(d) for _ in range(world_size)]
    for rb, b in zip(ref_blocks, blocks):
        rb.load_state_dict(b.state_dict())
    ref_x = x_full
    for b in ref_blocks:
        ref_x = b(ref_x)
    ref_loss = loss_fn(ref_x, target_full)
    ref_loss.backward()
    ref_grad_this_stage = {n: p.grad.clone() for n, p in ref_blocks[rank].named_parameters()}

    # ---- pipelined: this rank owns blocks[rank] only ----
    my_stage_mod = blocks[rank]
    stage = PipelineStage(my_stage_mod, stage_index=rank, num_stages=world_size, device=torch.device("cpu"))
    schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=loss_fn)

    if rank == 0:
        schedule.step(x_full)
    else:
        losses = []
        schedule.step(target=target_full, losses=losses)
        total_loss = torch.stack(losses).mean() if losses else None
        if total_loss is not None:
            print(f"[rank {rank}] pipelined mean loss={total_loss.item():.6f} vs ref loss={ref_loss.item():.6f}")

    my_grad = {n: p.grad for n, p in my_stage_mod.named_parameters()}
    ok = all(g is not None for g in my_grad.values())
    print(f"[rank {rank}] all local params got grads: {ok}")
    if ok:
        max_diff = max((my_grad[n] - ref_grad_this_stage[n]).abs().max().item() for n in my_grad)
        print(f"[rank {rank}] grad max abs diff vs reference (this stage's params): {max_diff:.3e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
