"""Phase 5: profiling CLI -- wraps any of this repo's five training
strategies (baseline / ddp / fsdp / tp / pp) with a real `torch.profiler`
capture and writes a JSON summary of the communication-vs-compute time
breakdown, wall-clock timing, and peak memory.

Deliberately reuses each strategy's OWN, UNMODIFIED `train(args)` (and
`build_arg_parser()`, for that script's strategy-specific defaults) from
`train_baseline.py`/`train_ddp.py`/`train_fsdp.py`/`train_tp.py`/
`train_pp.py` -- this script does not reimplement any training loop or
model-building code. It only wraps the WHOLE `train(args)` call in
`src.profiling.profile_call` (a single `torch.profiler.profile` capture
spanning construction, every training step, and that script's own
end-of-run correctness check), classifies the captured ops via
`src.profiling.is_comm_event`, and records peak CPU memory (always) / peak
CUDA memory (only if `torch.cuda.is_available()`).

For `--strategy baseline`, this runs single-process, no `torch.distributed`
involved at all -- the expected/checkable result is ~0 communication time.
For `--strategy {ddp,fsdp,tp,pp}`, `train(args)` internally calls
`dist.init_process_group` exactly like those scripts already do standalone
(same env-var convention: reads torchrun's RANK/WORLD_SIZE/LOCAL_RANK/
MASTER_ADDR/MASTER_PORT, defaulting to a degenerate single-process world if
launched with plain `python`) -- so a REAL multi-rank profiling run requires
launching this script itself with `torchrun`, same as the underlying
train_*.py script would need. Each rank writes its OWN summary JSON to
`--out-dir` (mirrors train_ddp.py's/train_fsdp.py's per-rank artifact
convention).

Usage (local smoke test, CPU, no comms, single process, seconds):
    python profile_run.py --strategy baseline --steps 15 --batch-size 8 \\
        --block-size 32 --d-model 32 --n-layer 2 --n-head 2 \\
        --out-dir /tmp/profile_baseline

Usage (local correctness smoke test, CPU, gloo, 2 processes -- the real,
checkable Phase 5 claim: DDP shows nonzero allreduce-shaped comm time,
baseline shows ~0):
    torchrun --standalone --nproc_per_node=2 profile_run.py --strategy ddp \\
        --steps 15 --batch-size 8 --block-size 32 --d-model 32 \\
        --n-layer 2 --n-head 2 --out-dir /tmp/profile_ddp

Usage (real run, Kaggle 2xT4, nccl -- see notebooks/kaggle_profiling_2gpu.py,
which runs all 5 strategies through this exact script):
    torchrun --standalone --nproc_per_node=2 profile_run.py --strategy fsdp \\
        --steps 100 --batch-size 64 --block-size 128 --d-model 256 \\
        --n-layer 4 --n-head 4 --backend nccl --out-dir /kaggle/working/profile_fsdp

No comms-overhead-% or scaling-efficiency number in this repo is measured on
real GPUs yet -- CPU/gloo can only prove the CLASSIFIER genuinely
distinguishes communication-heavy strategies from the no-communication
baseline (real, checkable claim), not report a meaningful throughput/speedup
number (needs >1 real GPU; TODO, see README.md/PLAN.md).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import train_baseline
import train_ddp
import train_fsdp
import train_pp
import train_tp

from src.profiling import ProfileResult, profile_call

_STRATEGY_MODULES = {
    "baseline": train_baseline,
    "ddp": train_ddp,
    "fsdp": train_fsdp,
    "tp": train_tp,
    "pp": train_pp,
}

# Attribute names that hold "loss/params" info differ slightly per
# train_*.py's own RunResult dataclass (see each script) -- checked in this
# priority order rather than assumed to be named identically everywhere.
_PARAM_COUNT_ATTRS = ("full_params", "num_params")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True, choices=list(_STRATEGY_MODULES))
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    # High default log-every: this is a profiling run, not a training demo --
    # per-step prints from the wrapped train_*.py are still real but kept
    # quiet by default so profiler output isn't buried.
    p.add_argument("--log-every", type=int, default=1_000_000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="baseline only (single-device strategy)")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "gloo", "nccl"],
                    help="ddp/fsdp/tp/pp only (gloo=CPU correctness, nccl=CUDA/Kaggle)")
    p.add_argument("--sharding-strategy", type=str, default="FULL_SHARD",
                    choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"], help="fsdp only")
    p.add_argument("--microbatches", type=int, default=2, help="pp only")
    p.add_argument("--out-dir", type=str, required=True)
    return p


def _namespace_for(strategy: str, args: argparse.Namespace) -> argparse.Namespace:
    """Build the exact `argparse.Namespace` `train_<strategy>.py`'s own
    `train(args)` expects: start from THAT script's own `build_arg_parser()`
    defaults (so any strategy-specific flag this CLI doesn't surface --
    e.g. baseline's `--out` file path -- still gets that script's own sane
    default, left unset/None here), then overlay the shared flags this CLI
    does expose. This is how `train(args)` is reused completely unmodified.
    """
    module = _STRATEGY_MODULES[strategy]
    base = module.build_arg_parser().parse_args([])
    for key, value in vars(args).items():
        if key == "strategy":
            continue
        if hasattr(base, key):
            setattr(base, key, value)
    return base


def _extract_meta(result) -> dict:
    """Pull whatever metadata fields exist on this strategy's own RunResult
    (field names differ slightly across train_*.py's dataclasses -- see
    _PARAM_COUNT_ATTRS)."""
    meta = {
        "rank": getattr(result, "rank", 0),
        "world_size": getattr(result, "world_size", 1),
        "device": getattr(result, "device", None),
        "total_time_s": getattr(result, "total_time_s", None),
    }
    for attr in _PARAM_COUNT_ATTRS:
        if hasattr(result, attr):
            meta["params"] = getattr(result, attr)
            break
    losses = [r.loss for r in getattr(result, "history", []) if getattr(r, "loss", None) is not None]
    meta["final_loss"] = losses[-1] if losses else None
    step_times = [r.step_time_s for r in getattr(result, "history", [])]
    meta["avg_step_time_s"] = sum(step_times) / len(step_times) if step_times else None
    return meta


def run(args: argparse.Namespace) -> tuple[dict, ProfileResult]:
    module = _STRATEGY_MODULES[args.strategy]
    ns = _namespace_for(args.strategy, args)

    result, profile_result = profile_call(lambda: module.train(ns))

    meta = _extract_meta(result)
    summary = {
        "strategy": args.strategy,
        **meta,
        "config": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "d_model": args.d_model,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "backend": args.backend,
        },
        **profile_result.to_dict(),
    }
    return summary, profile_result


def main() -> None:
    args = build_arg_parser().parse_args()
    summary, profile_result = run(args)

    rank = summary["rank"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"profile_{args.strategy}_rank{rank}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(
        f"[profile_run][rank {rank}] strategy={args.strategy} "
        f"comm_fraction={profile_result.comm_fraction:.4f} "
        f"comm_self_us={profile_result.comm_self_cpu_time_us:.1f} "
        f"compute_self_us={profile_result.compute_self_cpu_time_us:.1f} "
        f"wall_time_s={profile_result.wall_time_s:.3f} "
        f"peak_cpu_mem_after_bytes={profile_result.peak_cpu_memory_after_bytes:,} "
        f"peak_cuda_mem_bytes={profile_result.peak_cuda_memory_bytes}"
    )
    print(f"[profile_run][rank {rank}] wrote summary to {out_path}")


if __name__ == "__main__":
    main()
