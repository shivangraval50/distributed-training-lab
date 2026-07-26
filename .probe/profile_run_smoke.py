"""Phase 5 smoke test (ad hoc, pre-pytest): real 2-process gloo run of
profile_run.py's own `run()` for both baseline and ddp, via
torch.multiprocessing.spawn (same reasoning as tests/test_train_ddp.py's
docstring: torchrun's elastic-launcher rendezvous needs network access this
sandbox doesn't grant; mp.spawn is functionally equivalent -- real OS
processes, real gloo collectives, same init_process_group code path)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _worker(rank: int, world_size: int, strategy: str, out_dir: str, port: str, extra: list[str] | None = None) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port

    sys.path.insert(0, str(REPO_ROOT))
    from profile_run import build_arg_parser, run  # noqa: E402

    n_head = "4" if strategy == "tp" else "2"
    argv = [
        "--strategy", strategy,
        "--steps", "15",
        "--batch-size", "8",
        "--block-size", "32",
        "--d-model", "32",
        "--n-layer", "2",
        "--n-head", n_head,
        "--backend", "gloo",
        "--out-dir", out_dir,
    ] + (extra or [])
    args = build_arg_parser().parse_args(argv)
    summary, profile_result = run(args)

    out_path = Path(out_dir) / f"profile_{strategy}_rank{rank}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(
        f"[smoke][rank {rank}] strategy={strategy} "
        f"comm_fraction={profile_result.comm_fraction:.4f} "
        f"comm_self_us={profile_result.comm_self_cpu_time_us:.1f} "
        f"compute_self_us={profile_result.compute_self_cpu_time_us:.1f} "
        f"n_comm_ops={len(profile_result.comm_ops)} "
        f"top_comm_ops={[o.name for o in profile_result.comm_ops[:5]]}"
    )


def main() -> None:
    out_dir = "/tmp/profile_run_smoke"
    print("=== running DDP (2 ranks, gloo) through profile_run.run() ===")
    mp.spawn(_worker, args=(2, "ddp", out_dir + "/ddp", "29521", None), nprocs=2, join=True)

    print("\n=== running baseline (1 rank, no dist) through profile_run.run() ===")
    mp.spawn(_worker, args=(1, "baseline", out_dir + "/baseline", "29522", None), nprocs=1, join=True)

    print("\n=== running FSDP (2 ranks, gloo) through profile_run.run() ===")
    mp.spawn(_worker, args=(2, "fsdp", out_dir + "/fsdp", "29523", None), nprocs=2, join=True)

    print("\n=== running TP (2 ranks, gloo) through profile_run.run() ===")
    mp.spawn(_worker, args=(2, "tp", out_dir + "/tp", "29524", None), nprocs=2, join=True)

    print("\n=== running PP (2 ranks, gloo) through profile_run.run() ===")
    mp.spawn(_worker, args=(2, "pp", out_dir + "/pp", "29525", ["--microbatches", "2"]), nprocs=2, join=True)


if __name__ == "__main__":
    main()
