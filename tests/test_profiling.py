"""Real tests for Phase 5 (src/profiling.py + profile_run.py).

Two layers, mirroring this repo's existing DDP/FSDP/TP/PP test style:

1. Unit-level: `is_comm_event` against the ACTUAL event names observed by
   running real gloo collectives on this machine (see
   `.probe/profiling_probe*.py` for the original investigation -- these
   names are pasted here as literal, previously-observed strings, not
   guessed), plus `peak_cpu_memory_bytes` sanity (bytes, monotonic
   nondecreasing after a real allocation).
2. Integration-level, real 2-process `gloo` runs via
   `torch.multiprocessing.spawn` (same reasoning as tests/test_train_ddp.py:
   torchrun's elastic-launcher rendezvous needs network access this sandbox
   doesn't grant; mp.spawn is a real, functionally equivalent alternative --
   real OS processes, real `dist.init_process_group`, real collectives):
   the actual Phase 5 claim under test is that `profile_run.py`'s profiler
   genuinely tells baseline (no `torch.distributed` at all) apart from a
   real distributed strategy -- baseline must show EXACTLY zero
   communication time/ops, DDP must show a nonzero, allreduce-shaped comm
   time.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.profiling import is_comm_event, peak_cpu_memory_bytes, profile_steps  # noqa: E402


# --------------------------------------------------------------------------
# Unit tests: classifier against real, previously-observed event names.
# --------------------------------------------------------------------------

# Literal event names captured by .probe/profiling_probe.py (DDP allreduce),
# profiling_probe3.py (FSDP all_gather/reduce_scatter), profiling_probe4.py
# (send/recv), profiling_probe5.py (TP/DTensor functional collectives) --
# real torch.profiler output on this machine (torch 2.13.0, macOS, gloo),
# not invented strings.
_REAL_COMM_NAMES = [
    "c10d::allreduce_",
    "gloo:all_reduce",
    "c10d::allgather_",
    "gloo:all_gather",
    "c10d::_reduce_scatter_base_",
    "c10d::send",
    "gloo:send",
    "c10d::recv_",
    "gloo:recv",
    "c10d::broadcast_",
    "gloo:broadcast",
    "c10d::barrier",
    "_c10d_functional::all_reduce",
    "_c10d_functional::wait_tensor",
    "_c10d_functional::all_gather_into_tensor",
]

# Literal compute/bookkeeping event names captured in the same probes --
# must NOT be classified as communication even though several are directly
# caused by a distributed strategy's internal bookkeeping.
_REAL_COMPUTE_NAMES = [
    "aten::addmm",
    "aten::linear",
    "aten::mm",
    "DistributedDataParallel.forward",
    "torch::distributed::reducer::mul_out",
    "torch.distributed.ddp.reducer::copy_bucket_to_grad",
    "FullyShardedDataParallel.forward",
    "FullyShardedDataParallel._post_backward_hook",
    "aten::chunk",
    "aten::split_with_sizes",
    "Optimizer.step#SGD.step",
    "autograd::engine::evaluate_function: AddmmBackward0",
]


@pytest.mark.parametrize("name", _REAL_COMM_NAMES)
def test_is_comm_event_true_for_observed_comm_names(name):
    assert is_comm_event(name) is True


@pytest.mark.parametrize("name", _REAL_COMPUTE_NAMES)
def test_is_comm_event_false_for_observed_compute_names(name):
    assert is_comm_event(name) is False


def test_peak_cpu_memory_bytes_is_positive_and_monotonic():
    before = peak_cpu_memory_bytes()
    assert before > 0
    # Real 100MB allocation must push the process's peak RSS up (or leave it
    # unchanged if some larger peak already occurred earlier in this test
    # session -- ru_maxrss is a historical high-water mark, never decreases).
    _ = bytearray(100 * 1024 * 1024)
    after = peak_cpu_memory_bytes()
    assert after >= before


def test_profile_steps_pure_compute_has_no_comm():
    """A step_fn with zero torch.distributed calls must show exactly zero
    communication time and zero comm ops -- the baseline case."""
    x = torch.randn(64, 64)
    w = torch.randn(64, 64)

    def step_fn():
        (x @ w).sum().backward() if x.requires_grad else (x @ w).sum()

    result = profile_steps(step_fn, n_steps=5, warmup_steps=2)
    assert result.comm_self_cpu_time_us == 0.0
    assert result.comm_ops == []
    assert result.comm_fraction == 0.0
    assert result.compute_self_cpu_time_us > 0.0


# --------------------------------------------------------------------------
# Integration tests: real 2-process gloo collectives via mp.spawn.
# --------------------------------------------------------------------------


def _allreduce_worker(rank: int, world_size: int, port: int, result_path: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    t = torch.ones(64)

    def step_fn():
        dist.all_reduce(t)

    result = profile_steps(step_fn, n_steps=5, warmup_steps=2)

    if rank == 0:
        payload = {
            "comm_fraction": result.comm_fraction,
            "comm_self_cpu_time_us": result.comm_self_cpu_time_us,
            "comm_op_names": [o.name for o in result.comm_ops],
        }
        Path(result_path).write_text(json.dumps(payload))

    dist.barrier()
    dist.destroy_process_group()


def test_profile_steps_real_allreduce_shows_nonzero_comm(tmp_path):
    result_path = tmp_path / "allreduce_result.json"
    mp.spawn(
        _allreduce_worker, args=(2, 29531, str(result_path)), nprocs=2, join=True
    )
    payload = json.loads(result_path.read_text())
    assert payload["comm_fraction"] > 0.0
    assert payload["comm_self_cpu_time_us"] > 0.0
    assert any("reduce" in name.lower() for name in payload["comm_op_names"])


# --------------------------------------------------------------------------
# Integration test: profile_run.py's own run() for baseline vs ddp.
# --------------------------------------------------------------------------


def _profile_run_worker(
    rank: int, world_size: int, strategy: str, port: int, result_path: str
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    sys.path.insert(0, str(REPO_ROOT))
    from profile_run import build_arg_parser, run  # noqa: E402

    argv = [
        "--strategy", strategy,
        "--steps", "10",
        "--batch-size", "4",
        "--block-size", "16",
        "--d-model", "16",
        "--n-layer", "1",
        "--n-head", "2",
        "--backend", "gloo",
        "--log-every", "1000",
        "--out-dir", str(Path(result_path).parent),
    ]
    args = build_arg_parser().parse_args(argv)
    summary, profile_result = run(args)

    if rank == 0:
        Path(result_path).write_text(
            json.dumps(
                {
                    "comm_fraction": profile_result.comm_fraction,
                    "comm_self_cpu_time_us": profile_result.comm_self_cpu_time_us,
                    "n_comm_ops": len(profile_result.comm_ops),
                    "comm_op_names": [o.name for o in profile_result.comm_ops],
                }
            )
        )


def test_profile_run_baseline_has_zero_comm(tmp_path):
    result_path = tmp_path / "baseline_result.json"
    mp.spawn(
        _profile_run_worker, args=(1, "baseline", 29532, str(result_path)), nprocs=1, join=True
    )
    payload = json.loads(result_path.read_text())
    assert payload["comm_fraction"] == 0.0
    assert payload["comm_self_cpu_time_us"] == 0.0
    assert payload["n_comm_ops"] == 0


def test_profile_run_ddp_has_nonzero_allreduce_shaped_comm(tmp_path):
    result_path = tmp_path / "ddp_result.json"
    mp.spawn(
        _profile_run_worker, args=(2, "ddp", 29533, str(result_path)), nprocs=2, join=True
    )
    payload = json.loads(result_path.read_text())
    assert payload["comm_fraction"] > 0.0
    assert payload["comm_self_cpu_time_us"] > 0.0
    assert payload["n_comm_ops"] > 0
    assert any(
        "allreduce" in name.lower() or "all_reduce" in name.lower()
        for name in payload["comm_op_names"]
    )


def test_profile_run_baseline_vs_ddp_comm_fraction_genuinely_differs(tmp_path):
    """The end-to-end honest claim for Phase 5: the SAME profiling mechanism
    (profile_run.py -> src.profiling.profile_call) produces genuinely
    different comm_fraction for a no-communication strategy vs a real
    distributed one -- not two hand-picked numbers, the actual live
    computation each time."""
    baseline_path = tmp_path / "baseline2_result.json"
    mp.spawn(
        _profile_run_worker, args=(1, "baseline", 29534, str(baseline_path)), nprocs=1, join=True
    )
    ddp_path = tmp_path / "ddp2_result.json"
    mp.spawn(
        _profile_run_worker, args=(2, "ddp", 29535, str(ddp_path)), nprocs=2, join=True
    )

    baseline_payload = json.loads(baseline_path.read_text())
    ddp_payload = json.loads(ddp_path.read_text())

    assert baseline_payload["comm_fraction"] == 0.0
    assert ddp_payload["comm_fraction"] > baseline_payload["comm_fraction"]
