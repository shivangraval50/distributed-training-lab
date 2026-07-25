"""Phase 2: DDP (data parallel) training.

Wraps the same `TinyGPT` model (src/model.py) used by the Phase 1 baseline
(`train_baseline.py`) in `torch.nn.parallel.DistributedDataParallel`, and
launches with `torchrun` for multi-process data parallelism.

Backend is selectable and auto-picks like the rest of this repo's device
philosophy: `gloo` on CPU (works with zero GPUs -- this is what makes DDP
*correctness* genuinely verifiable on this dev machine, unlike throughput),
`nccl` on CUDA (the real backend for the Kaggle 2xT4 run). Gloo is not a
mock or a stand-in for nccl: it is a real torch.distributed backend, and
`init_process_group` / gradient allreduce actually execute across real OS
processes -- only the *speed* numbers require real GPUs, not the mechanics.

Each rank trains on a disjoint, contiguous shard of the corpus (no two ranks
ever see the same underlying text window, other than at most a
block_size-sized seam at a shard boundary) -- this is the data-parallel part:
different ranks process different data every step, and DDP's backward-pass
gradient allreduce is what keeps the model weights identical across ranks
despite that. See tests/test_train_ddp.py for the actual proof (2-process
local gloo run: per-rank loss histories differ, but final weights match
bit-for-bit).

Usage (local correctness smoke test, CPU, gloo, 2 processes, seconds):
    torchrun --standalone --nproc_per_node=2 train_ddp.py \\
        --steps 20 --batch-size 8 --block-size 32 --d-model 32 \\
        --n-layer 2 --n-head 2 --log-every 5

Usage (single process, degenerate world_size=1 "DDP" -- quick sanity check
that the script runs without torchrun at all; no real parallelism):
    python train_ddp.py --steps 20 --batch-size 8 --block-size 32 \\
        --d-model 32 --n-layer 2 --n-head 2 --log-every 5

Usage (real run, Kaggle 2xT4, nccl -- see notebooks/kaggle_ddp_2gpu.py for
the ready-to-run remote script that actually launches 2 processes across
the 2 T4s):
    torchrun --standalone --nproc_per_node=2 train_ddp.py \\
        --steps 500 --batch-size 64 --block-size 128 --d-model 256 \\
        --n-layer 4 --n-head 4 --backend nccl

No throughput/speedup numbers are hard-coded anywhere in this repo: a real
GPU scaling number only exists once this has actually been run on Kaggle's
2xT4 (see README.md / PLAN.md). As of writing only the CPU/gloo correctness
run has been executed.
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
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import build_corpus, build_vocab, encode, get_batch
from src.model import TinyGPT, count_flops_per_token


def pick_backend_and_device(requested: str, local_rank: int) -> tuple[str, torch.device]:
    """Mirror train_baseline.py's pick_device philosophy, but for a backend.

    gloo <-> CPU, nccl <-> CUDA. "auto" picks nccl if CUDA is visible on this
    process, else gloo -- exactly the CUDA > CPU half of the baseline's
    CUDA > MPS > CPU ladder (MPS has no torch.distributed backend, so DDP
    across MPS devices is out of scope here; that's a genuine gap, not an
    oversight -- see README Limitations).
    """
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
    """Read torchrun's env vars and init the process group.

    torchrun sets RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT before
    the script starts. If launched with plain `python` (no torchrun), we
    default to a degenerate single-process world (rank 0 of 1) so the script
    is still runnable for a fast sanity check -- not a substitute for the
    real 2-process test.
    """
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
    """Split `data` into `world_size` disjoint, contiguous shards; return rank's shard.

    Contiguous partitioning (not e.g. "every world_size-th index") is used so
    disjointness is trivially checkable by comparing [start, end) ranges --
    the last rank absorbs the remainder so every token is covered exactly
    once. This is what guarantees no duplicate work across ranks: rank i
    never samples a training window from rank j's [start, end) range.
    """
    n = len(data)
    shard_len = n // world_size
    start = rank * shard_len
    end = start + shard_len if rank < world_size - 1 else n
    return data[start:end], start, end


def params_fingerprint(model: nn.Module) -> torch.Tensor:
    """A cheap, exact per-parameter checksum (sum of each tensor).

    Used as a real, independent (of DDP's own allreduce) collective check:
    after training, every rank all_gathers its fingerprint and rank 0
    compares them bit-for-bit. If DDP's gradient sync were broken (e.g. each
    rank silently training independently on its own shard), these would
    differ given that ranks see different data.
    """
    with torch.no_grad():
        parts = [p.detach().float().sum().reshape(1) for p in model.parameters()]
    return torch.cat(parts)


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
    num_params: int
    shard_start: int
    shard_end: int
    history: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0


def train(args: argparse.Namespace) -> RunResult:
    backend, device = pick_backend_and_device(args.backend, int(os.environ.get("LOCAL_RANK", "0")))
    rank, world_size, local_rank = setup_distributed(backend)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    # Same corpus/vocab on every rank (pure function of args -- no
    # communication needed to agree on it), scaled by world_size so each
    # rank's shard stays roughly baseline-sized rather than shrinking as
    # world_size grows.
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

    # Deliberately DIFFERENT seed per rank for model init: this proves DDP's
    # constructor-time broadcast (not "we happened to seed identically") is
    # what makes rank 0's initial weights the ones every rank trains from.
    torch.manual_seed(args.seed * 1000 + rank)
    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    pre_ddp_fp = params_fingerprint(model)

    if device.type == "cuda":
        ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        ddp_model = DDP(model)

    post_ddp_fp = params_fingerprint(ddp_model.module)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    if rank == 0:
        print(
            f"[ddp] world_size={world_size} backend={backend} device={device} "
            f"params={model.num_params():,} vocab_size={len(stoi)} "
            f"approx_flops/token={count_flops_per_token(model):,}"
        )
    print(
        f"[ddp][rank {rank}] shard=[{shard_start}:{shard_end}) "
        f"({shard_end - shard_start} tokens) pre_ddp_fp_sample={pre_ddp_fp[0].item():.4f}"
    )

    # Per-rank generator: different seed per rank means even if shards
    # somehow overlapped, sampled batches would still differ -- belt and
    # suspenders on top of the disjoint shard partition above.
    gen = torch.Generator().manual_seed(args.seed + rank)
    result = RunResult(
        rank=rank,
        world_size=world_size,
        device=str(device),
        num_params=model.num_params(),
        shard_start=shard_start,
        shard_end=shard_end,
    )

    ddp_model.train()
    run_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(shard, args.batch_size, args.block_size, generator=gen)

        _sync(device)
        t0 = time.perf_counter()

        logits = ddp_model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # triggers DDP's gradient allreduce across ranks
        optimizer.step()

        _sync(device)
        step_time = time.perf_counter() - t0

        tokens = args.batch_size * args.block_size
        tps = tokens / step_time if step_time > 0 else float("inf")
        record = StepRecord(step=step, loss=loss.item(), step_time_s=step_time, tokens_per_sec=tps)
        result.history.append(record)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[ddp][rank {rank}] step {step:4d}/{args.steps} "
                f"loss={record.loss:.4f} step_time={step_time * 1000:.1f}ms "
                f"tok/s={tps:,.0f}"
            )

    result.total_time_s = time.perf_counter() - run_start

    # Independent collective check (not DDP's own allreduce): every rank's
    # final parameters must be bit-identical if gradient sync actually
    # happened, despite every rank having trained on different local data.
    local_fp = params_fingerprint(ddp_model.module)
    gathered = [torch.zeros_like(local_fp) for _ in range(world_size)]
    dist.all_gather(gathered, local_fp)
    weights_match = all(torch.equal(gathered[0], g) for g in gathered)

    if rank == 0:
        status = "OK -- all ranks bit-identical" if weights_match else "MISMATCH"
        print(f"[ddp] weight sync check ({world_size} ranks): {status}")
        if not weights_match:
            raise SystemExit("DDP weight sync check failed: ranks diverged")

    if args.out_dir:
        _write_rank_artifacts(args.out_dir, rank, args, result, ddp_model.module)

    print(
        f"[ddp][rank {rank}] done in {result.total_time_s:.2f}s "
        f"final_loss={result.history[-1].loss:.4f}"
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
    out_dir: str, rank: int, args: argparse.Namespace, result: RunResult, module: nn.Module
) -> None:
    """Persist this rank's final weights + loss history for an external
    (out-of-process) correctness check -- see tests/test_train_ddp.py, which
    launches the real 2-process torchrun run and diffs these files."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    torch.save(module.state_dict(), path / f"rank{rank}_state.pt")

    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "rank": result.rank,
        "world_size": result.world_size,
        "shard_start": result.shard_start,
        "shard_end": result.shard_end,
        "num_params": result.num_params,
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
        "--out-dir",
        type=str,
        default=None,
        help="optional dir to write per-rank state_dict + loss history for verification",
    )
    return p


def main() -> RunResult:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    main()
