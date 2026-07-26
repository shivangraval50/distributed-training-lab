"""Probe #4: full-fidelity PP probe with TinyGPT-shaped stages: integer
token-index input, embeddings on stage 0, LM head + cross-entropy loss on
stage 1, microbatching (n_microbatches=2), gradient correctness check
against a single-process reference over an identical (idx, target) batch.
"""
import os
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import sys
sys.path.insert(0, "/Users/shivangraval/Downloads/distributed-training-lab")
from src.model import TinyGPT


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29706"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    torch.manual_seed(11)
    vocab_size, block_size, d_model, n_layer, n_head = 13, 8, 16, 2, 2
    ref = TinyGPT(vocab_size, block_size, d_model, n_layer, n_head, dropout=0.0)
    ref.eval()  # avoid any nondeterminism; dropout=0 anyway

    class Stage0(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.tok_emb = m.tok_emb
            self.pos_emb = m.pos_emb
            self.layer0 = m.blocks.layers[0]
            self.register_buffer("mask", m.causal_mask, persistent=False)

        def forward(self, idx):
            b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            mask = self.mask[:t, :t]
            x = self.layer0(x, src_mask=mask, is_causal=True)
            return x

    class Stage1(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.layer1 = m.blocks.layers[1]
            self.ln_f = m.ln_f
            self.head = m.head
            self.register_buffer("mask", m.causal_mask, persistent=False)

        def forward(self, x):
            t = x.shape[1]
            mask = self.mask[:t, :t]
            x = self.layer1(x, src_mask=mask, is_causal=True)
            x = self.ln_f(x)
            return self.head(x)

    torch.manual_seed(99)
    idx = torch.randint(0, vocab_size, (4, block_size))
    target = torch.randint(0, vocab_size, (4, block_size))
    loss_fn = lambda logits, target: nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1)
    )

    # reference
    ref_logits = ref(idx)
    ref_loss = loss_fn(ref_logits, target)
    ref_loss.backward()
    # Key by the Parameter OBJECT's id, not name -- Stage0/Stage1 share the
    # exact same nn.Parameter objects as `ref` (submodule references, not
    # copies), so identity-based lookup is robust to the different
    # dotted-name prefixes each stage module sees for the same tensor.
    ref_grads = {id(p): p.grad.clone() for p in ref.parameters() if p.grad is not None}

    stages = [Stage0(ref), Stage1(ref)]
    # NOTE: reference already ran backward above and populated .grad on the
    # SAME underlying parameter tensors that stages[rank] shares (since Stage0/
    # Stage1 wrap submodules of `ref` directly) -- zero them before the
    # pipelined run so the pipelined backward starts from a clean slate.
    for p in ref.parameters():
        if p.grad is not None:
            p.grad = None

    my_stage_mod = stages[rank]
    stage = PipelineStage(my_stage_mod, stage_index=rank, num_stages=world_size, device=torch.device("cpu"))
    schedule = ScheduleGPipe(stage, n_microbatches=2, loss_fn=loss_fn)

    if rank == 0:
        schedule.step(idx)
        out = None
    else:
        losses = []
        out = schedule.step(target=target, losses=losses)
        mean_loss = torch.stack(losses).mean()
        match_out = torch.allclose(out, ref_logits, atol=1e-5, rtol=1e-4)
        print(f"[rank {rank}] pipelined mean_loss={mean_loss.item():.6f} ref_loss={ref_loss.item():.6f} "
              f"logits match ref: {match_out} (max diff {(out-ref_logits).abs().max().item():.3e})")

    my_grads = {id(p): p.grad for p in my_stage_mod.parameters()}
    ok = all(g is not None for g in my_grads.values())
    if ok:
        max_diff = max((my_grads[i] - ref_grads[i]).abs().max().item() for i in my_grads)
        print(f"[rank {rank}] all local grads populated: {ok}, max grad diff vs reference: {max_diff:.3e}")
    else:
        print(f"[rank {rank}] MISSING GRADS: {sum(1 for g in my_grads.values() if g is None)} params")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
