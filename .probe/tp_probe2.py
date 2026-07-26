"""Probe #2: is the real DTensor TP API not just runnable but NUMERICALLY
correct on CPU/gloo (vs a non-parallel reference with identical weights)?
And can it target nn.MultiheadAttention directly (TinyGPT's actual attention
module), or only plain nn.Linear submodules?"""
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29702"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.fc1 = nn.Linear(d, 4 * d)
            self.fc2 = nn.Linear(4 * d, d)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    d = 8
    torch.manual_seed(123)  # SAME seed on every rank -> identical full weights before sharding
    model = MLP(d)
    torch.manual_seed(999)  # different seed for input, but same across ranks too (replicated input)
    x = torch.randn(4, d)

    # Reference: compute on this rank's UN-parallelized copy before we mutate `model` in place.
    ref_model = MLP(d)
    ref_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        ref_out = ref_model(x)

    mesh = DeviceMesh("cpu", list(range(world_size)))
    plan = {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()}
    tp_model = parallelize_module(model, mesh, plan)
    with torch.no_grad():
        tp_out = tp_model(x)
        if hasattr(tp_out, "full_tensor"):
            tp_out = tp_out.full_tensor()

    match = torch.allclose(tp_out, ref_out, atol=1e-6, rtol=1e-5)
    print(f"[rank {rank}] DTensor-TP numerical match vs reference: {match} "
          f"(max abs diff={ (tp_out-ref_out).abs().max().item():.3e})")

    # Now: can the plan target nn.MultiheadAttention directly (TinyGPT's real
    # attention module)? It combines Q/K/V into ONE raw Parameter
    # (in_proj_weight), not separate nn.Linear submodules.
    mha = nn.MultiheadAttention(embed_dim=d, num_heads=2, batch_first=True)
    has_linear_qkv = any(isinstance(m, nn.Linear) for name, m in mha.named_modules() if "in_proj" in name)
    print(f"[rank {rank}] MultiheadAttention exposes in_proj as nn.Linear submodule? {has_linear_qkv} "
          f"(in_proj_weight is a raw Parameter: {type(mha.in_proj_weight).__name__}, "
          f"shape={tuple(mha.in_proj_weight.shape)})")
    try:
        plan2 = {"in_proj_weight": ColwiseParallel()}
        parallelize_module(mha, mesh, plan2)
        print(f"[rank {rank}] parallelize_module on 'in_proj_weight' unexpectedly SUCCEEDED")
    except Exception as e:
        print(f"[rank {rank}] parallelize_module on raw Parameter name FAILED as expected: "
              f"{type(e).__name__}: {e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
