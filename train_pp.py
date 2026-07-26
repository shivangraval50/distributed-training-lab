"""Phase 4b: Pipeline parallelism (PP) training -- real
`torch.distributed.pipelining` (`PipelineStage` + `ScheduleGPipe`), splitting
TinyGPT's transformer blocks across ranks by LAYER, with real `dist.send`/
`dist.recv` of activations/gradients between stages (not simulated).

--------------------------------------------------------------------------
Step 1 finding (empirical -- see `.probe/pp_probe*.py` and
`src/pipeline_model.py`'s module docstring for the full investigation and
exact numbers): the real API works out of the box on CPU/gloo, is BIT-FOR-
BIT correct on forward (max diff 0.0 vs a single-process reference) and
matches reference gradients to ~1e-8 on backward -- for TinyGPT's ACTUAL,
UNMODIFIED architecture (src/model.py), no rewrite needed (unlike Phase 4a's
TP, which needed a Megatron-style attention rewrite because
`nn.MultiheadAttention`'s fused weight can't be targeted by the DTensor TP
API -- PP splits at LAYER boundaries, which is architecture-agnostic).

UNLIKE DDP/FSDP/TP: PP ranks do NOT hold overlapping/replicated parameters,
so there is no "prove the broadcast synced everyone" step -- each rank owns
a strictly DISJOINT slice of the model's layers (stage 0 = embeddings +
first `n_layer // world_size` blocks; the last rank = remaining blocks +
final norm + LM head). Every rank constructs its OWN stage module from a
model built with the SAME fixed seed (deterministic CPU init is bit-
identical across processes given the same seed -- the same assumption
tests/test_train_tp.py's reference-reconstruction already relies on), so
there is nothing to synchronize at construction time. The actual
correctness question is instead: does the ASSEMBLED (2-stage,
cross-process, real send/recv) computation match a single-process
reference running the SAME full model on the SAME data? See
`tests/test_train_pp.py`.

Microbatching: `ScheduleGPipe(n_microbatches=args.microbatches)` -- GPipe's
"all microbatch forwards, then all backwards" schedule. HONESTLY: on 2 local
CPU processes with blocking `gloo` P2P, no wall-clock overlap is measured or
claimed here; only the real multi-microbatch data flow and cross-process
communication are verified (see src/pipeline_model.py docstring).

Usage (local correctness smoke test, CPU, gloo, 2 processes, seconds):
    torchrun --standalone --nproc_per_node=2 train_pp.py \\
        --steps 20 --batch-size 8 --block-size 32 --d-model 32 \\
        --n-layer 2 --n-head 2 --microbatches 2 --log-every 5

Usage (single process, degenerate world_size=1 sanity check -- one stage
holds the ENTIRE model, no real pipelining):
    python train_pp.py --steps 20 --batch-size 8 --block-size 32 \\
        --d-model 32 --n-layer 2 --n-head 2 --log-every 5

world_size=1 fix note: this degenerate case originally crashed with two
compounding bugs (found running the command above), both now fixed:
  1. `train()`'s step loop branched on `is_first`/`is_last` as if they were
     mutually exclusive ranks (`if is_first: ... elif is_last: ...`). At
     world_size=1, rank 0 is BOTH simultaneously, so the `is_first` branch
     ran first and never passed `target`/`losses`, and `ScheduleGPipe`
     raised `PipeliningMetadataError` ("loss_fn and target required for
     last stage"). Fixed by adding an explicit `is_first and is_last`
     branch that calls `schedule.step(x, target=y, losses=losses)`.
  2. `src/pipeline_model.py`'s `build_pp_stage_module` checked
     `stage_index == 0` before `stage_index == num_stages - 1`, so at
     world_size=1 stage 0 was built as `PPFirstStage` (embeddings + blocks
     only) instead of a stage that also owns the final LayerNorm/LM head --
     the forward pass silently produced d_model-sized output instead of
     vocab_size-sized logits, and cross-entropy against real token targets
     raised `IndexError: Target ... is out of bounds`. Fixed by adding a
     `PPSingleStage` (embeddings + ALL blocks + final norm + head) used
     whenever `num_stages == 1`, checked before the other branches.
Both are covered by the single-process command above, which now runs
successfully end to end (see README/PLAN for the actual observed output).

Usage (real run, Kaggle 2xT4, nccl -- see notebooks/kaggle_pp_2gpu.py):
    torchrun --standalone --nproc_per_node=2 train_pp.py \\
        --steps 500 --batch-size 64 --block-size 128 --d-model 256 \\
        --n-layer 4 --n-head 4 --microbatches 4 --backend nccl

No throughput/scaling numbers are hard-coded anywhere in this repo: a real
GPU number (including any real pipeline-bubble/overlap measurement) only
exists once this has actually been run on Kaggle's 2xT4 (see README.md /
PLAN.md). As of writing only the CPU/gloo correctness run (proving genuine
cross-process layer split + numerical correctness, not speed/overlap) has
executed.
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
from src.pipeline_model import build_pp_stage_module, stage_layer_ranges


def pick_backend_and_device(requested: str, local_rank: int) -> tuple[str, torch.device]:
    """Same device philosophy as train_ddp.py/train_fsdp.py/train_tp.py."""
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
    """Identical to train_ddp.py's/train_fsdp.py's/train_tp.py's setup_distributed."""
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


@dataclass
class StepRecord:
    step: int
    loss: float | None
    step_time_s: float
    tokens_per_sec: float


@dataclass
class RunResult:
    rank: int
    world_size: int
    device: str
    full_params: int
    local_stage_numel: int
    layer_start: int
    layer_end: int
    history: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0


def train(args: argparse.Namespace) -> RunResult:
    from torch.distributed.pipelining import ScheduleGPipe, PipelineStage

    backend, device = pick_backend_and_device(args.backend, int(os.environ.get("LOCAL_RANK", "0")))
    rank, world_size, local_rank = setup_distributed(backend)

    if device.type == "cuda":
        torch.cuda.set_device(device)

    if args.n_layer < world_size:
        raise SystemExit(f"--n-layer {args.n_layer} must be >= world_size {world_size} (>=1 layer/stage)")

    corpus = build_corpus(repeats=max(50, args.batch_size * args.block_size // 10 + 50))
    stoi, _ = build_vocab(corpus)
    full_data = encode(corpus, stoi).to(device)

    # SAME fixed seed on EVERY rank (not per-rank-different): unlike
    # DDP/FSDP/TP, PP ranks own strictly DISJOINT parameters, so there is no
    # risk of "different init per rank" desyncing anything -- there is
    # nothing shared to desync. Deterministic CPU/GPU init with a fixed seed
    # is bit-identical across independent processes (same assumption
    # tests/test_train_tp.py's reference-reconstruction already relies on).
    torch.manual_seed(args.seed)
    full_model = TinyGPT(
        vocab_size=len(stoi), block_size=args.block_size, d_model=args.d_model,
        n_layer=args.n_layer, n_head=args.n_head, dropout=0.0,
    ).to(device)
    full_params = full_model.num_params()

    layer_start, layer_end = stage_layer_ranges(args.n_layer, world_size)[rank]
    stage_module = build_pp_stage_module(full_model, rank, world_size).to(device)
    local_stage_numel = sum(p.numel() for p in stage_module.parameters())

    if rank == 0:
        print(
            f"[pp] world_size={world_size} backend={backend} device={device} "
            f"full_params={full_params:,} vocab_size={len(stoi)} "
            f"approx_flops/token={count_flops_per_token(full_model):,} "
            f"microbatches={args.microbatches}"
        )
    print(
        f"[pp][rank {rank}] owns layers[{layer_start}:{layer_end}) "
        f"local_stage_params={local_stage_numel:,}/{full_params:,} "
        f"({100 * local_stage_numel / full_params:.1f}% of full model)"
    )

    stage = PipelineStage(stage_module, stage_index=rank, num_stages=world_size, device=device)

    def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

    schedule = ScheduleGPipe(stage, n_microbatches=args.microbatches, loss_fn=loss_fn)
    optimizer = torch.optim.AdamW(stage_module.parameters(), lr=args.lr)

    # SAME data seed on every rank (every rank can independently reproduce
    # the identical (x, y) batch -- only rank 0 actually feeds x in and only
    # the last rank actually consumes y, but generating it identically
    # everywhere keeps this simple and avoids any need to broadcast the
    # batch itself).
    gen = torch.Generator().manual_seed(args.seed + 1)
    result = RunResult(
        rank=rank, world_size=world_size, device=str(device), full_params=full_params,
        local_stage_numel=local_stage_numel, layer_start=layer_start, layer_end=layer_end,
    )

    is_first = rank == 0
    is_last = rank == world_size - 1

    run_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(full_data, args.batch_size, args.block_size, generator=gen)

        _sync(device)
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        # NOTE: is_first and is_last are NOT mutually exclusive -- at
        # world_size=1 (degenerate single-stage sanity check, see module
        # docstring), rank 0 is BOTH the first stage (needs `x` fed in) AND
        # the last stage (ScheduleGPipe requires `target`/`losses` for
        # whichever stage_index == num_stages - 1, even if that's also
        # stage 0). Four branches, not three, to cover that overlap.
        if is_first and is_last:
            schedule.step(x, target=y, losses=losses)
            loss_val = torch.stack(losses).mean().item() if losses else None
        elif is_first:
            schedule.step(x)
            loss_val = None
        elif is_last:
            schedule.step(target=y, losses=losses)
            loss_val = torch.stack(losses).mean().item() if losses else None
        else:
            schedule.step()
            loss_val = None
        optimizer.step()

        _sync(device)
        step_time = time.perf_counter() - t0

        tokens = args.batch_size * args.block_size
        tps = tokens / step_time if step_time > 0 else float("inf")
        record = StepRecord(step=step, loss=loss_val, step_time_s=step_time, tokens_per_sec=tps)
        result.history.append(record)

        if is_last and (step % args.log_every == 0 or step == 1 or step == args.steps):
            print(
                f"[pp][rank {rank}] step {step:4d}/{args.steps} "
                f"loss={loss_val:.4f} step_time={step_time * 1000:.1f}ms tok/s={tps:,.0f}"
            )

    result.total_time_s = time.perf_counter() - run_start

    if args.out_dir:
        _write_rank_artifacts(args.out_dir, rank, args, result, stage_module)

    print(f"[pp][rank {rank}] done in {result.total_time_s:.2f}s local_stage_params={local_stage_numel:,}/{full_params:,}")

    dist.barrier()
    dist.destroy_process_group()
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _write_rank_artifacts(
    out_dir: str, rank: int, args: argparse.Namespace, result: RunResult, stage_module: nn.Module
) -> None:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(stage_module.state_dict(), path / f"rank{rank}_stage_state.pt")
    payload = {
        "config": {k: v for k, v in vars(args).items()},
        "rank": result.rank,
        "world_size": result.world_size,
        "full_params": result.full_params,
        "local_stage_numel": result.local_stage_numel,
        "layer_start": result.layer_start,
        "layer_end": result.layer_end,
        "history": [r.__dict__ for r in result.history],
    }
    (path / f"rank{rank}_history.json").write_text(json.dumps(payload, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8, help="GLOBAL batch size, split into microbatches per step")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--microbatches", type=int, default=2)
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
