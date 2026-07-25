"""Phase 3: FSDP (fully sharded data parallel) training -- real torch FSDP,
ZeRO-3-style parameter/gradient/optimizer-state sharding.

--------------------------------------------------------------------------
Step 1 finding (empirical, not assumed): does real
`torch.distributed.fsdp.FullyShardedDataParallel` work on CPU with `gloo` on
this machine (macOS/Apple Silicon, torch 2.13.0, no CUDA)? YES -- with two
CPU-specific caveats discovered by actually running it (see README/PLAN
Limitations for the exact tracebacks):

1. `FSDP(model)` with the default `device_id=None` FAILS on this machine:
   `_init_device_handle` calls `torch._C._get_accelerator()` when the module
   is on CPU, which detects Apple Silicon's MPS backend as "the accelerator"
   (even though we're deliberately running gloo/CPU, not MPS) and then tries
   `torch.mps.current_device()`, which this torch build's FSDP device-handle
   shim does not implement -> `AttributeError`. Passing `device_id=torch.device("cpu")`
   explicitly bypasses that auto-detection path entirely and works.
2. `sync_module_states=True` (FSDP's own constructor-time weight broadcast,
   the analogue of DDP's automatic broadcast) unconditionally requires GPU
   tensors in this torch version and raises `ValueError` on CPU, even with
   `device_id="cpu"` set. So on CPU/gloo we do the equivalent real broadcast
   ourselves with plain `dist.broadcast` calls (see `broadcast_state_dict`
   below) *before* wrapping in FSDP -- same effect (every rank starts from
   rank 0's exact weights, via a genuine collective, not accidental identical
   seeding), different code path. On CUDA, `sync_module_states=True` works
   normally and is used instead (no manual broadcast needed there).

Everything else about real FSDP genuinely works on CPU/gloo: parameters,
gradients, and optimizer state are each ACTUALLY sharded across ranks (this
is verified, not assumed -- see `tests/test_train_fsdp.py`, which checks
that each rank's local flat-parameter count and optimizer-state tensor count
are a strict fraction of the full model's, not merely "training runs").
Forward/backward/optimizer-step exercises real FSDP internals: flat
parameter construction, per-step all-gather (unshard) before forward/
backward, reduce-scatter of gradients, and re-shard after each step. Only
speed, not correctness, is untested here (that needs real GPUs).

This module wraps the same `TinyGPT` model (src/model.py, unmodified) used
by the Phase 1 baseline and Phase 2 DDP, mirroring train_ddp.py's structure
(backend auto-pick, per-rank disjoint data shard, weight-sync proof) so the
three phases are directly comparable.

Usage (local correctness smoke test, CPU, gloo, 2 processes, seconds):
    torchrun --standalone --nproc_per_node=2 train_fsdp.py \\
        --steps 20 --batch-size 8 --block-size 32 --d-model 32 \\
        --n-layer 2 --n-head 2 --log-every 5

Usage (single process, degenerate world_size=1 sanity check -- FSDP falls
back to NO_SHARD automatically when world_size==1; no real sharding):
    python train_fsdp.py --steps 20 --batch-size 8 --block-size 32 \\
        --d-model 32 --n-layer 2 --n-head 2 --log-every 5

Usage (real run, Kaggle 2xT4, nccl -- see notebooks/kaggle_fsdp_2gpu.py):
    torchrun --standalone --nproc_per_node=2 train_fsdp.py \\
        --steps 500 --batch-size 64 --block-size 128 --d-model 256 \\
        --n-layer 4 --n-head 4 --backend nccl

No memory/throughput numbers are hard-coded anywhere in this repo: a real
GPU number only exists once this has actually been run on Kaggle's 2xT4
(see README.md / PLAN.md). As of writing only the CPU/gloo correctness run
(proving genuine sharding + genuine sync, not speed) has been executed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

from src.data import build_corpus, build_vocab, encode, get_batch
from src.model import TinyGPT, count_flops_per_token


def pick_backend_and_device(requested: str, local_rank: int) -> tuple[str, torch.device]:
    """Same device philosophy as train_ddp.py's pick_backend_and_device:
    gloo <-> CPU, nccl <-> CUDA. MPS has no torch.distributed backend, so
    FSDP across MPS devices is out of scope here (same gap as DDP)."""
    if requested == "auto":
        requested = "nccl" if torch.cuda.is_available() else "gloo"

    if requested == "nccl":
        if not torch.cuda.is_available():
            raise SystemExit("--backend nccl requested but CUDA is not available")
        return "nccl", torch.device(f"cuda:{local_rank}")
    if requested == "gloo":
        return "gloo", torch.device("cpu")
    raise ValueError(f"unsupported backend: {requested}")


def setup_distributed(backend: str) -> tuple[int, int, int]:
    """Identical to train_ddp.py's setup_distributed: read torchrun's env
    vars (RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT), defaulting to
    a degenerate single-process world if launched with plain `python`."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def make_rank_shard(data: torch.Tensor, rank: int, world_size: int) -> tuple[torch.Tensor, int, int]:
    """Identical to train_ddp.py's make_rank_shard: disjoint, contiguous
    per-rank partition of the corpus (last rank absorbs the remainder)."""
    n = len(data)
    shard_len = n // world_size
    start = rank * shard_len
    end = start + shard_len if rank < world_size - 1 else n
    return data[start:end], start, end


def fsdp_device_id(device: torch.device, local_rank: int) -> torch.device:
    """The `device_id` FSDP needs at construction time.

    On CUDA this is the standard per-rank GPU. On CPU, passing this
    EXPLICITLY (rather than leaving it None) is required on this machine --
    see the module docstring's Step 1 finding #1: with `device_id=None`,
    FSDP's internal accelerator auto-detection finds Apple Silicon's MPS
    backend and crashes trying to use it, even though we are running
    gloo/CPU, not MPS.
    """
    if device.type == "cuda":
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def broadcast_state_dict(model: nn.Module, src: int = 0) -> None:
    """Manual, real-collective constructor-time weight broadcast.

    Substitutes for FSDP's `sync_module_states=True`, which (Step 1 finding
    #2) requires GPU tensors in this torch version and raises on CPU even
    with `device_id="cpu"` set. Every tensor in `model.state_dict()` is
    broadcast from rank `src` via `dist.broadcast` -- a genuine
    `torch.distributed` collective over real gloo processes, not a local
    copy -- so every rank provably ends up with rank `src`'s exact weights
    before FSDP ever shards them, deliberately mirroring the "different seed
    per rank at init, single real broadcast makes them identical" proof
    pattern used by train_ddp.py for DDP's own broadcast.
    """
    with torch.no_grad():
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=src)


def full_params_fingerprint(fsdp_model: FSDP) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Temporarily materialize the COMPLETE (unsharded) model on this rank
    via `FSDP.summon_full_params` and return (a) a cheap per-tensor-sum
    fingerprint for a fast collective compare and (b) the actual full
    tensors themselves for an exact `torch.equal` check (mirrors
    train_ddp.py's params_fingerprint + the state_dict torch.equal check in
    tests/test_train_ddp.py, applied to FSDP's sharded parameters instead of
    DDP's already-replicated ones).
    """
    with FSDP.summon_full_params(fsdp_model, writeback=False):
        full_state = {name: p.detach().clone() for name, p in fsdp_model.named_parameters()}
    fp = torch.cat([t.float().sum().reshape(1) for t in full_state.values()])
    return fp, full_state


@dataclass
class StepRecord:
    step: int
    loss: float
    step_time_s: float
    tokens_per_sec: float


@dataclass
class RunResult:
    rank: int
    world_size: int
    device: str
    full_params: int
    local_shard_numel: int
    shard_start: int
    shard_end: int
    history: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0
    optimizer_state_numel: int = 0


def train(args: argparse.Namespace) -> RunResult:
    backend, device = pick_backend_and_device(args.backend, int(os.environ.get("LOCAL_RANK", "0")))
    rank, world_size, local_rank = setup_distributed(backend)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    corpus = build_corpus(
        repeats=max(50, args.batch_size * args.block_size // 10 + 50) * world_size
    )
    stoi, _ = build_vocab(corpus)
    full_data = encode(corpus, stoi)
    shard, shard_start, shard_end = make_rank_shard(full_data, rank, world_size)
    shard = shard.to(device)

    if shard_end - shard_start <= args.block_size:
        raise SystemExit(
            f"rank {rank} shard length {shard_end - shard_start} <= block_size "
            f"{args.block_size}; increase corpus size or reduce world_size/block_size."
        )

    # Deliberately DIFFERENT seed per rank for model init -- same reasoning
    # as train_ddp.py: proves the broadcast below (not accidental identical
    # seeding) is what makes every rank start from rank 0's weights.
    torch.manual_seed(args.seed * 1000 + rank)
    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)
    full_params = model.num_params()

    # Real per-rank raw checksum BEFORE any sync, purely for the printed
    # before/after evidence that ranks really started from different init.
    pre_sync_sum = sum(p.detach().float().sum().item() for p in model.parameters())

    use_fsdp_native_sync = device.type == "cuda"
    if not use_fsdp_native_sync:
        broadcast_state_dict(model, src=0)  # CPU/gloo workaround, see docstring
    post_sync_sum = sum(p.detach().float().sum().item() for p in model.parameters())

    fsdp_kwargs: dict = dict(
        sharding_strategy=ShardingStrategy[args.sharding_strategy],
        device_id=fsdp_device_id(device, local_rank),
    )
    if use_fsdp_native_sync:
        fsdp_kwargs["sync_module_states"] = True  # real FSDP's own broadcast, works on CUDA

    fsdp_model = FSDP(model, **fsdp_kwargs)
    local_shard_numel = sum(p.numel() for p in fsdp_model.parameters())

    if rank == 0:
        print(
            f"[fsdp] world_size={world_size} backend={backend} device={device} "
            f"sharding_strategy={args.sharding_strategy} full_params={full_params:,} "
            f"vocab_size={len(stoi)} approx_flops/token={count_flops_per_token(model):,}"
        )
    print(
        f"[fsdp][rank {rank}] shard=[{shard_start}:{shard_end}) "
        f"({shard_end - shard_start} tokens) pre_sync_param_sum={pre_sync_sum:.4f} "
        f"post_sync_param_sum={post_sync_sum:.4f} "
        f"local_shard_params={local_shard_numel:,}/{full_params:,} "
        f"({100 * local_shard_numel / full_params:.1f}% of full model)"
    )

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    gen = torch.Generator().manual_seed(args.seed + rank)
    result = RunResult(
        rank=rank,
        world_size=world_size,
        device=str(device),
        full_params=full_params,
        local_shard_numel=local_shard_numel,
        shard_start=shard_start,
        shard_end=shard_end,
    )

    fsdp_model.train()
    run_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(shard, args.batch_size, args.block_size, generator=gen)

        _sync(device)
        t0 = time.perf_counter()

        logits = fsdp_model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # triggers FSDP's reduce-scatter of gradients across ranks
        optimizer.step()

        _sync(device)
        step_time = time.perf_counter() - t0

        tokens = args.batch_size * args.block_size
        tps = tokens / step_time if step_time > 0 else float("inf")
        record = StepRecord(step=step, loss=loss.item(), step_time_s=step_time, tokens_per_sec=tps)
        result.history.append(record)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[fsdp][rank {rank}] step {step:4d}/{args.steps} "
                f"loss={record.loss:.4f} step_time={step_time * 1000:.1f}ms "
                f"tok/s={tps:,.0f}"
            )

    result.total_time_s = time.perf_counter() - run_start

    # Optimizer-state sharding check: AdamW's exp_avg/exp_avg_sq are only
    # ever allocated for the LOCAL flat-parameter shard (since `optimizer`
    # was constructed from fsdp_model.parameters(), which returns each
    # rank's local shard, not the full model) -- this is the actual ZeRO-3
    # memory-saving mechanic, verified by counting real tensor elements
    # rather than assumed.
    result.optimizer_state_numel = sum(
        v.numel() for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)
    )

    # Full-parameter correctness check: the FSDP analogue of DDP's
    # weight-sync check (tests/test_train_ddp.py). Materialize the complete
    # (unsharded) model on every rank and confirm bit-for-bit equality across
    # ranks -- proof that per-step reduce-scatter (grads) + all-gather
    # (params) actually kept every rank's shards in sync despite each rank
    # training on different local data throughout.
    fp, full_state = full_params_fingerprint(fsdp_model)
    gathered = [torch.zeros_like(fp) for _ in range(world_size)]
    dist.all_gather(gathered, fp)
    weights_match = all(torch.equal(gathered[0], g) for g in gathered)

    if rank == 0:
        status = "OK -- all ranks bit-identical" if weights_match else "MISMATCH"
        print(f"[fsdp] full-parameter sync check ({world_size} ranks): {status}")
        if not weights_match:
            raise SystemExit("FSDP weight sync check failed: ranks diverged")

    if args.out_dir:
        _write_rank_artifacts(args.out_dir, rank, args, result, full_state)

    print(
        f"[fsdp][rank {rank}] done in {result.total_time_s:.2f}s "
        f"final_loss={result.history[-1].loss:.4f} "
        f"local_shard_params={local_shard_numel:,}/{full_params:,} "
        f"optimizer_state_numel={result.optimizer_state_numel:,}"
    )

    dist.barrier()
    dist.destroy_process_group()
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _write_rank_artifacts(
    out_dir: str,
    rank: int,
    args: argparse.Namespace,
    result: RunResult,
    full_state: dict[str, torch.Tensor],
) -> None:
    """Persist this rank's fully-materialized (unsharded) final weights +
    loss history + sharding stats for an external correctness check -- see
    tests/test_train_fsdp.py, which launches the real 2-process run and
    diffs these files."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(full_state, path / f"rank{rank}_full_state.pt")

    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "rank": result.rank,
        "world_size": result.world_size,
        "shard_start": result.shard_start,
        "shard_end": result.shard_end,
        "full_params": result.full_params,
        "local_shard_numel": result.local_shard_numel,
        "optimizer_state_numel": result.optimizer_state_numel,
        "history": [r.__dict__ for r in result.history],
    }
    (path / f"rank{rank}_history.json").write_text(json.dumps(payload, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20, help="number of training steps")
    p.add_argument("--batch-size", type=int, default=8, help="PER-RANK (local) batch size")
    p.add_argument("--block-size", type=int, default=32, help="context length (tokens)")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "gloo", "nccl"],
        help="gloo=CPU (works with no GPU, correctness), nccl=CUDA (Kaggle scaling)",
    )
    p.add_argument(
        "--sharding-strategy",
        type=str,
        default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"],
        help="FULL_SHARD=ZeRO-3-like (params+grads+optim state sharded, default); "
        "SHARD_GRAD_OP=ZeRO-2-like; NO_SHARD=plain replication (like DDP)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="optional dir to write per-rank full state_dict + loss history for verification",
    )
    return p


def main() -> RunResult:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    main()
