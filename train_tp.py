"""Phase 4a: Tensor parallelism (TP) training -- real
`torch.distributed.tensor` (DTensor) `ColwiseParallel`/`RowwiseParallel` +
`DeviceMesh`, Megatron-style attention/MLP head-/hidden-dim sharding.

--------------------------------------------------------------------------
Step 1 finding (empirical -- see `.probe/tp_probe*.py` and
`src/tensor_parallel.py`'s module docstring for the full investigation):
the real DTensor-based TP API genuinely works AND is numerically correct on
CPU/gloo (verified: sharded output matches an unsharded reference to ~1e-7,
float32 noise), but only for actual `nn.Linear`/`nn.Embedding` submodules --
it cannot target `nn.MultiheadAttention`'s fused `in_proj_weight` (a single
raw Parameter, not three Linear submodules). TinyGPT's attention (built on
`nn.MultiheadAttention` via `nn.TransformerEncoderLayer`, see src/model.py)
hits exactly that limitation. So this module trains a SEPARATE model class,
`TPTinyGPT` (src/tensor_parallel.py) -- numerically equivalent to TinyGPT,
verified in `tests/test_train_tp.py`, but built from explicit wq/wk/wv/wo +
fc1/fc2 `nn.Linear` submodules so the real API's `parallelize_module` can
shard them -- rather than either (a) claiming the real API works on
`nn.MultiheadAttention` when it demonstrably doesn't, or (b) hand-writing
`dist.all_reduce` calls ourselves when PyTorch's own DTensor machinery
already does exactly that correctly. This is the real API, applied to a
TP-shardable rewrite of the same architecture -- exactly what production TP
frameworks (Megatron-LM, torchtitan) do for the same underlying reason.

KEY DIFFERENCE FROM DDP/FSDP (both Phase 2/3): TP shards the MODEL across
ranks, not the DATA. Every rank sees the IDENTICAL batch every step (same
seed, no per-rank offset) -- unlike DDP/FSDP, where per-rank loss histories
differ because ranks train on disjoint data. Here, per-rank loss histories
must be IDENTICAL (same data + same effective computation, just split
across ranks and recombined via all_reduce) -- this is the actual TP
correctness invariant checked by `tests/test_train_tp.py`, deliberately the
opposite assertion from `tests/test_train_ddp.py`/`test_train_fsdp.py`.

Requires `n_head % world_size == 0` (whole attention heads per rank) and
`(4 * d_model) % world_size == 0` (FFN hidden dim per rank) -- documented
constraints of head-/hidden-dim-parallel TP, not an artifact of this repo.

Usage (local correctness smoke test, CPU, gloo, 2 processes, seconds):
    torchrun --standalone --nproc_per_node=2 train_tp.py \\
        --steps 20 --batch-size 8 --block-size 32 --d-model 32 \\
        --n-layer 2 --n-head 4 --log-every 5

Usage (single process, degenerate world_size=1 sanity check -- TP mesh of
size 1, no real sharding):
    python train_tp.py --steps 20 --batch-size 8 --block-size 32 \\
        --d-model 32 --n-layer 2 --n-head 4 --log-every 5

Usage (real run, Kaggle 2xT4, nccl -- see notebooks/kaggle_tp_2gpu.py):
    torchrun --standalone --nproc_per_node=2 train_tp.py \\
        --steps 500 --batch-size 64 --block-size 128 --d-model 256 \\
        --n-layer 4 --n-head 8 --backend nccl

No throughput/scaling numbers are hard-coded anywhere in this repo: a real
GPU number only exists once this has actually been run on Kaggle's 2xT4
(see README.md / PLAN.md). As of writing only the CPU/gloo correctness run
(proving genuine sharding + numerical correctness, not speed) has executed.
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

from src.data import build_corpus, build_vocab, encode, get_batch
from src.model import TinyGPT, count_flops_per_token
from src.tensor_parallel import TPTinyGPT, load_from_tinygpt, build_tp_plan


def pick_backend_and_device(requested: str, local_rank: int) -> tuple[str, torch.device]:
    """Same device philosophy as train_ddp.py/train_fsdp.py: gloo <-> CPU,
    nccl <-> CUDA. MPS has no torch.distributed backend."""
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
    """Identical to train_ddp.py's / train_fsdp.py's setup_distributed."""
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


def broadcast_state_dict(model: nn.Module, src: int = 0) -> None:
    """Real constructor-time weight broadcast (same pattern as
    train_fsdp.py's `broadcast_state_dict`): every rank starts model
    construction from a DIFFERENT seed, then this makes every rank's
    reference model bit-identical to rank `src`'s via a genuine
    `dist.broadcast` collective, before TP sharding ever splits it."""
    with torch.no_grad():
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=src)


def local_shard_fraction(tp_model: TPTinyGPT) -> tuple[int, int]:
    """(local numel actually held by this rank, full/global numel) summed
    over every parameter -- DTensor params report LOCAL numel via
    `.to_local()`, replicated params (embeddings/norms/head) report their
    full numel on every rank (they are not sharded, by design -- see module
    docstring). Real per-tensor introspection, not an assumed 1/world_size.
    """
    local_total = 0
    global_total = 0
    for p in tp_model.parameters():
        if hasattr(p, "to_local"):
            local_total += p.to_local().numel()
            global_total += p.numel()  # DTensor.numel() reports the GLOBAL logical size
        else:
            local_total += p.numel()
            global_total += p.numel()
    return local_total, global_total


def full_params_fingerprint(tp_model: TPTinyGPT) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Materialize every parameter's FULL (unsharded) tensor on this rank
    (`DTensor.full_tensor()` triggers a real all_gather; plain nn.Parameters
    are already full) and return a cheap sum-fingerprint + the full state,
    mirroring train_fsdp.py's `full_params_fingerprint`."""
    full_state = {}
    for name, p in tp_model.named_parameters():
        full_state[name] = p.full_tensor().detach().clone() if hasattr(p, "full_tensor") else p.detach().clone()
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
    history: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0


def train(args: argparse.Namespace) -> RunResult:
    backend, device = pick_backend_and_device(args.backend, int(os.environ.get("LOCAL_RANK", "0")))
    rank, world_size, local_rank = setup_distributed(backend)

    if args.n_head % world_size != 0:
        raise SystemExit(
            f"--n-head {args.n_head} must be divisible by world_size {world_size} "
            "(TP shards attention by whole heads per rank)"
        )
    if (4 * args.d_model) % world_size != 0:
        raise SystemExit(
            f"4 * --d-model ({4 * args.d_model}) must be divisible by world_size {world_size} "
            "(TP shards the FFN hidden dim per rank)"
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)

    # TP does NOT shard data -- every rank must see IDENTICAL batches, so
    # (unlike train_ddp.py/train_fsdp.py) there is no per-rank data shard and
    # no per-rank sampling-generator offset.
    corpus = build_corpus(repeats=max(50, args.batch_size * args.block_size // 10 + 50))
    stoi, _ = build_vocab(corpus)
    full_data = encode(corpus, stoi).to(device)

    # Deliberately DIFFERENT seed per rank for the reference model's init --
    # proves the broadcast below (not accidental identical seeding) is what
    # gives every rank the same starting weights, same proof pattern as
    # train_ddp.py/train_fsdp.py.
    torch.manual_seed(args.seed * 1000 + rank)
    reference = TinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=0.0,
    ).to(device)
    full_params = reference.num_params()
    pre_sync_sum = sum(p.detach().float().sum().item() for p in reference.parameters())
    broadcast_state_dict(reference, src=0)
    post_sync_sum = sum(p.detach().float().sum().item() for p in reference.parameters())

    tp_model = TPTinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
    ).to(device)
    load_from_tinygpt(tp_model, reference)  # every rank: identical copy (reference is now synced)

    mesh_device = "cuda" if device.type == "cuda" else "cpu"
    from torch.distributed.tensor import DeviceMesh
    from torch.distributed.tensor.parallel import parallelize_module

    mesh = DeviceMesh(mesh_device, list(range(world_size)))
    parallelize_module(tp_model, mesh, build_tp_plan(args.n_layer))

    local_shard_numel, _ = local_shard_fraction(tp_model)

    if rank == 0:
        print(
            f"[tp] world_size={world_size} backend={backend} device={device} "
            f"full_params={full_params:,} vocab_size={len(stoi)} "
            f"approx_flops/token={count_flops_per_token(reference):,}"
        )
    print(
        f"[tp][rank {rank}] pre_sync_param_sum={pre_sync_sum:.4f} "
        f"post_sync_param_sum={post_sync_sum:.4f} "
        f"local_shard_params={local_shard_numel:,}/{full_params:,} "
        f"({100 * local_shard_numel / full_params:.1f}% of full model -- expect >50% since "
        f"embeddings/norms/head stay replicated, only attn+MLP are sharded)"
    )

    # Real, non-trivial correctness assertion (not "just check it runs"):
    # the TP-sharded model's forward on a batch must match the reference's
    # forward on the SAME batch/weights. Uses a fixed (non-rank-dependent)
    # seed so every rank builds the identical probe batch.
    gen = torch.Generator().manual_seed(args.seed)
    probe_x, _ = get_batch(full_data, args.batch_size, args.block_size, generator=gen)
    reference.eval()
    tp_model.eval()
    with torch.no_grad():
        ref_logits = reference(probe_x)
        tp_logits = tp_model(probe_x)
    fwd_match = torch.allclose(ref_logits, tp_logits, atol=1e-4, rtol=1e-3)
    max_diff = (ref_logits - tp_logits).abs().max().item()
    if rank == 0:
        print(f"[tp] pre-training forward-equivalence check vs single-process reference: "
              f"match={fwd_match} max_abs_diff={max_diff:.3e}")
        if not fwd_match:
            raise SystemExit("TP forward does not match single-process reference: sharding is broken")
    tp_model.train()

    optimizer = torch.optim.AdamW(tp_model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # SAME seed across every rank (not `+ rank`) -- every rank samples the
    # IDENTICAL sequence of batches. This is the key TP-specific invariant:
    # unlike DDP/FSDP, TP ranks must see identical data every step.
    gen = torch.Generator().manual_seed(args.seed + 1)
    result = RunResult(
        rank=rank, world_size=world_size, device=str(device),
        full_params=full_params, local_shard_numel=local_shard_numel,
    )

    run_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(full_data, args.batch_size, args.block_size, generator=gen)

        _sync(device)
        t0 = time.perf_counter()

        logits = tp_model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # triggers DTensor's implicit all_reduce (autograd of RowwiseParallel's redistribute)
        optimizer.step()

        _sync(device)
        step_time = time.perf_counter() - t0

        tokens = args.batch_size * args.block_size
        tps = tokens / step_time if step_time > 0 else float("inf")
        record = StepRecord(step=step, loss=loss.item(), step_time_s=step_time, tokens_per_sec=tps)
        result.history.append(record)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(
                f"[tp][rank {rank}] step {step:4d}/{args.steps} "
                f"loss={record.loss:.4f} step_time={step_time * 1000:.1f}ms "
                f"tok/s={tps:,.0f}"
            )

    result.total_time_s = time.perf_counter() - run_start

    # Cross-rank check (independent of the reference-equivalence check
    # above): every rank's per-step loss must be IDENTICAL (same data + same
    # combined computation), the TP analogue of DDP/FSDP's weight-sync
    # check, but on LOSS rather than final weights, since here it's the
    # per-step values (not just the endpoint) that should match.
    local_losses = torch.tensor([r.loss for r in result.history])
    gathered_losses = [torch.zeros_like(local_losses) for _ in range(world_size)]
    dist.all_gather(gathered_losses, local_losses)
    losses_match = all(torch.equal(gathered_losses[0], g) for g in gathered_losses)

    fp, full_state = full_params_fingerprint(tp_model)
    gathered_fp = [torch.zeros_like(fp) for _ in range(world_size)]
    dist.all_gather(gathered_fp, fp)
    weights_match = all(torch.equal(gathered_fp[0], g) for g in gathered_fp)

    if rank == 0:
        print(f"[tp] cross-rank identical-loss check ({world_size} ranks): "
              f"{'OK -- all ranks bit-identical per-step loss' if losses_match else 'MISMATCH'}")
        print(f"[tp] cross-rank full-parameter check ({world_size} ranks): "
              f"{'OK -- all ranks bit-identical' if weights_match else 'MISMATCH'}")
        if not losses_match or not weights_match:
            raise SystemExit("TP cross-rank check failed: ranks diverged")

    if args.out_dir:
        _write_rank_artifacts(args.out_dir, rank, args, result, full_state)

    print(
        f"[tp][rank {rank}] done in {result.total_time_s:.2f}s "
        f"final_loss={result.history[-1].loss:.4f} "
        f"local_shard_params={local_shard_numel:,}/{full_params:,}"
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
    out_dir: str, rank: int, args: argparse.Namespace, result: RunResult, full_state: dict[str, torch.Tensor]
) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(full_state, path / f"rank{rank}_full_state.pt")
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "rank": result.rank,
        "world_size": result.world_size,
        "full_params": result.full_params,
        "local_shard_numel": result.local_shard_numel,
        "history": [r.__dict__ for r in result.history],
    }
    (path / f"rank{rank}_history.json").write_text(json.dumps(payload, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20, help="number of training steps")
    p.add_argument("--batch-size", type=int, default=8, help="GLOBAL batch size (replicated on every rank, NOT sharded)")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument(
        "--backend", type=str, default="auto", choices=["auto", "gloo", "nccl"],
        help="gloo=CPU (works with no GPU, correctness), nccl=CUDA (Kaggle scaling)",
    )
    p.add_argument("--out-dir", type=str, default=None)
    return p


def main() -> RunResult:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    main()
