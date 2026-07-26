"""Probe #3: full TPTinyGPT sharded across 2 real gloo ranks via the real
DeviceMesh + parallelize_module API, compared against the single-process
TinyGPT reference over the SAME input (forward, then one training step)."""
import os
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

sys.path.insert(0, "/Users/shivangraval/Downloads/distributed-training-lab")
from src.model import TinyGPT
from src.tensor_parallel import TPTinyGPT, load_from_tinygpt, build_tp_plan


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29707"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    torch.manual_seed(0)
    vocab_size, block_size, d_model, n_layer, n_head = 17, 16, 32, 2, 4
    ref = TinyGPT(vocab_size, block_size, d_model, n_layer, n_head, dropout=0.0)

    tp_model = TPTinyGPT(vocab_size, block_size, d_model, n_layer, n_head)
    load_from_tinygpt(tp_model, ref)

    mesh = DeviceMesh("cpu", list(range(world_size)))
    parallelize_module(tp_model, mesh, build_tp_plan(n_layer))

    wq_weight = tp_model.blocks[0].attn.wq.weight
    is_dtensor = hasattr(wq_weight, "to_local")
    local_wq_out = wq_weight.to_local().shape[0] if is_dtensor else wq_weight.shape[0]
    print(f"[rank {rank}] wq.weight is DTensor={is_dtensor} global_shape={tuple(wq_weight.shape)} "
          f"LOCAL shard output dim={local_wq_out} (full d_model={d_model}, "
          f"expected local={d_model // world_size})")

    torch.manual_seed(5)
    idx = torch.randint(0, vocab_size, (3, block_size))
    target = torch.randint(0, vocab_size, (3, block_size))

    ref.eval()
    with torch.no_grad():
        ref_out = ref(idx)

    tp_model.eval()
    with torch.no_grad():
        tp_out = tp_model(idx)
    match = torch.allclose(ref_out, tp_out, atol=1e-4, rtol=1e-3)
    print(f"[rank {rank}] sharded TP forward matches reference: {match} "
          f"(max diff {(ref_out - tp_out).abs().max().item():.3e})")

    # one training step on both, compare updated output
    tp_model.train()
    opt = torch.optim.AdamW(tp_model.parameters(), lr=1e-2)
    logits = tp_model(idx)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

    ref.train()
    ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
    ref_logits = ref(idx)
    ref_loss = torch.nn.functional.cross_entropy(ref_logits.reshape(-1, vocab_size), target.reshape(-1))
    ref_opt.zero_grad()
    ref_loss.backward()
    ref_opt.step()

    print(f"[rank {rank}] loss before step: tp={loss.item():.6f} ref={ref_loss.item():.6f}")

    tp_model.eval()
    ref.eval()
    with torch.no_grad():
        tp_out2 = tp_model(idx)
        ref_out2 = ref(idx)
    match2 = torch.allclose(ref_out2, tp_out2, atol=1e-3, rtol=1e-2)
    print(f"[rank {rank}] post-1-step forward matches reference: {match2} "
          f"(max diff {(ref_out2 - tp_out2).abs().max().item():.3e})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
