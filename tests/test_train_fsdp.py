"""Real correctness test for Phase 3 FSDP (train_fsdp.py).

Launches a genuine multi-process torch.distributed run on CPU with the real
`torch.distributed.fsdp.FullyShardedDataParallel` (NOT a mock/hand-rolled
stand-in -- see train_fsdp.py's module docstring for the Step 1 investigation
of exactly what works/doesn't on CPU/gloo in this torch version, and the two
documented workarounds needed to make it run at all here) and verifies the
actual point of FSDP sharding, analogous in spirit to tests/test_train_ddp.py
but checking sharding, not just gradient sync:

1. Each rank's LOCAL flat-parameter count is a genuine fraction of the full
   model's parameter count (proof params are actually sharded, not just
   "training runs") -- and, for world_size=2, exactly half.
2. Each rank's optimizer-state tensor count (AdamW's exp_avg/exp_avg_sq) is
   likewise only allocated for the local shard, not the full model -- the
   actual ZeRO-3 memory-saving mechanic, checked by counting real tensor
   elements, not assumed.
3. Each rank trains on a disjoint, contiguous shard of the corpus and its
   local per-step loss differs from the other ranks' (real different data).
4. After training, the COMPLETE (unsharded) model -- materialized on every
   rank via FSDP.summon_full_params -- is bit-for-bit identical across all
   ranks (proof FSDP's reduce-scatter/all-gather actually kept the shards in
   sync, since independently-trained-on-different-data shards would
   otherwise diverge).

Uses torch.multiprocessing.spawn, same reasoning as test_train_ddp.py:
functionally equivalent to torchrun (same init_process_group/FSDP code path,
real OS processes, real gloo collectives) and avoids torchrun's elastic
rendezvous hanging in this sandbox. `torchrun` is still the documented way to
run train_fsdp.py directly (see its docstring / README) -- that's what
notebooks/kaggle_fsdp_2gpu.py uses for the real 2xT4/nccl run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker(rank: int, world_size: int, out_dir: str, argv: list[str]) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29611")

    sys.path.insert(0, str(REPO_ROOT))
    from train_fsdp import build_arg_parser, train  # noqa: E402

    args = build_arg_parser().parse_args(argv)
    train(args)


def _run_fsdp(tmp_path: Path, world_size: int = 2, master_port: str = "29611", **overrides) -> Path:
    out_dir = tmp_path / "fsdp_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MASTER_PORT"] = master_port

    defaults = dict(
        steps=20,
        batch_size=4,
        block_size=16,
        d_model=32,
        n_layer=2,
        n_head=2,
        log_every=1000,
        backend="gloo",
        sharding_strategy="FULL_SHARD",
    )
    defaults.update(overrides)
    argv = [
        "--steps", str(defaults["steps"]),
        "--batch-size", str(defaults["batch_size"]),
        "--block-size", str(defaults["block_size"]),
        "--d-model", str(defaults["d_model"]),
        "--n-layer", str(defaults["n_layer"]),
        "--n-head", str(defaults["n_head"]),
        "--log-every", str(defaults["log_every"]),
        "--backend", defaults["backend"],
        "--sharding-strategy", defaults["sharding_strategy"],
        "--out-dir", str(out_dir),
    ]

    mp.spawn(_worker, args=(world_size, str(out_dir), argv), nprocs=world_size, join=True)
    return out_dir


def test_params_and_optimizer_state_are_genuinely_sharded(tmp_path):
    out_dir = _run_fsdp(tmp_path, world_size=2, master_port="29612")

    hist0 = json.loads((out_dir / "rank0_history.json").read_text())
    hist1 = json.loads((out_dir / "rank1_history.json").read_text())

    full_params = hist0["full_params"]
    assert hist1["full_params"] == full_params

    # Each rank must hold a STRICT fraction of the full model's parameters,
    # not the whole thing -- with 2 ranks and FULL_SHARD, exactly half.
    assert hist0["local_shard_numel"] < full_params
    assert hist1["local_shard_numel"] < full_params
    assert hist0["local_shard_numel"] == full_params // 2
    assert hist1["local_shard_numel"] == full_params // 2

    # AdamW keeps 2 state tensors (exp_avg, exp_avg_sq) per parameter --
    # these must be sized to the LOCAL shard, not the full model. If they
    # were allocated for the full model (no real sharding), this would be
    # >= 2 * full_params on every rank.
    assert hist0["optimizer_state_numel"] < 2 * full_params
    assert hist0["optimizer_state_numel"] >= 2 * hist0["local_shard_numel"]
    assert hist1["optimizer_state_numel"] < 2 * full_params


def test_ranks_get_disjoint_shards(tmp_path):
    out_dir = _run_fsdp(tmp_path, master_port="29613")

    hist0 = json.loads((out_dir / "rank0_history.json").read_text())
    hist1 = json.loads((out_dir / "rank1_history.json").read_text())

    s0, e0 = hist0["shard_start"], hist0["shard_end"]
    s1, e1 = hist1["shard_start"], hist1["shard_end"]

    assert e0 > s0 and e1 > s1
    assert e0 == s1
    assert max(s0, s1) >= min(e0, e1)


def test_ranks_train_on_different_data_but_converge_to_identical_full_weights(tmp_path):
    out_dir = _run_fsdp(tmp_path, master_port="29614", steps=25)

    hist0 = json.loads((out_dir / "rank0_history.json").read_text())
    hist1 = json.loads((out_dir / "rank1_history.json").read_text())

    losses0 = [r["loss"] for r in hist0["history"]]
    losses1 = [r["loss"] for r in hist1["history"]]

    # Different local shards + different per-rank sampling seed => real
    # per-step loss values must differ (proof each rank processed different
    # data, not duplicate/replicated work).
    assert losses0 != pytest.approx(losses1)

    # Despite training on different data, FSDP's reduce-scatter (grads) +
    # all-gather (params) each step must keep every rank's fully-materialized
    # weights bit-for-bit identical.
    state0 = torch.load(out_dir / "rank0_full_state.pt", map_location="cpu")
    state1 = torch.load(out_dir / "rank1_full_state.pt", map_location="cpu")

    assert state0.keys() == state1.keys()
    for key in state0:
        assert torch.equal(state0[key], state1[key]), f"parameter {key!r} diverged across ranks"


def test_weight_sync_holds_over_many_steps(tmp_path):
    out_dir = _run_fsdp(tmp_path, master_port="29615", steps=60, log_every=1000)

    state0 = torch.load(out_dir / "rank0_full_state.pt", map_location="cpu")
    state1 = torch.load(out_dir / "rank1_full_state.pt", map_location="cpu")
    for key in state0:
        assert torch.equal(state0[key], state1[key])


def test_three_ranks_shard_disjointly_and_sync(tmp_path):
    out_dir = _run_fsdp(tmp_path, world_size=3, master_port="29616", steps=10)

    hists = [
        json.loads((out_dir / f"rank{r}_history.json").read_text()) for r in range(3)
    ]
    ranges = sorted((h["shard_start"], h["shard_end"]) for h in hists)
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1

    full_params = hists[0]["full_params"]
    for h in hists:
        # 3-way sharding: no single rank holds the full model.
        assert h["local_shard_numel"] < full_params

    states = [torch.load(out_dir / f"rank{r}_full_state.pt", map_location="cpu") for r in range(3)]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])
