"""Real correctness test for Phase 2 DDP (train_ddp.py).

Launches a genuine 2-process torch.distributed run on CPU with the `gloo`
backend (no GPU needed -- gloo is a real backend, not a mock) and verifies
the actual point of data parallelism:

1. Each rank is assigned a disjoint, contiguous shard of the corpus (no
   duplicate work).
2. Each rank's local per-step loss differs from the other rank's (proof they
   really trained on different data, not e.g. accidentally identical batches).
3. After training, every rank's final model weights are bit-for-bit
   identical (proof DDP's constructor-time broadcast + per-step gradient
   allreduce actually synchronized the ranks, since otherwise -- training
   independently on different data -- they would have diverged).

Uses torch.multiprocessing.spawn rather than shelling out to `torchrun`:
functionally equivalent (same init_process_group/DDP code path, real OS
processes, real gloo collectives), and avoids torchrun's elastic-launcher
rendezvous (which resolves `socket.gethostname()` rather than honoring
MASTER_ADDR, and hangs in some sandboxed/offline dev environments with no
DNS for the local hostname). The documented, recommended way to run this
script directly is still `torchrun` (see train_ddp.py's docstring / README) --
that is what should be used on Kaggle.
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
    # Fresh env per spawned process; MASTER_ADDR is a literal IP (not a
    # hostname) so init_process_group's TCPStore never needs DNS.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29511")

    sys.path.insert(0, str(REPO_ROOT))
    from train_ddp import build_arg_parser, train  # noqa: E402

    args = build_arg_parser().parse_args(argv)
    train(args)


def _run_ddp(tmp_path: Path, world_size: int = 2, master_port: str = "29511", **overrides) -> Path:
    out_dir = tmp_path / "ddp_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MASTER_PORT"] = master_port

    defaults = dict(
        steps=20,
        batch_size=4,
        block_size=16,
        d_model=16,
        n_layer=1,
        n_head=2,
        log_every=1000,
        backend="gloo",
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
        "--out-dir", str(out_dir),
    ]

    mp.spawn(_worker, args=(world_size, str(out_dir), argv), nprocs=world_size, join=True)
    return out_dir


def test_ranks_get_disjoint_shards(tmp_path):
    out_dir = _run_ddp(tmp_path, master_port="29512")

    hist0 = json.loads((out_dir / "rank0_history.json").read_text())
    hist1 = json.loads((out_dir / "rank1_history.json").read_text())

    s0, e0 = hist0["shard_start"], hist0["shard_end"]
    s1, e1 = hist1["shard_start"], hist1["shard_end"]

    assert e0 > s0 and e1 > s1
    # Contiguous partition: rank 0's range ends exactly where rank 1's
    # begins, and the two [start, end) ranges do not overlap.
    assert e0 == s1
    assert max(s0, s1) >= min(e0, e1)


def test_ranks_train_on_different_data_but_converge_to_identical_weights(tmp_path):
    out_dir = _run_ddp(tmp_path, master_port="29513", steps=25)

    hist0 = json.loads((out_dir / "rank0_history.json").read_text())
    hist1 = json.loads((out_dir / "rank1_history.json").read_text())

    losses0 = [r["loss"] for r in hist0["history"]]
    losses1 = [r["loss"] for r in hist1["history"]]

    # Different local shards + different per-rank sampling seed => the
    # actual per-step loss values must differ (this is what proves each
    # rank really processed different data, not duplicate work).
    assert losses0 != pytest.approx(losses1)

    # But despite training on different data, DDP's gradient allreduce must
    # keep every rank's final weights bit-for-bit identical.
    state0 = torch.load(out_dir / "rank0_state.pt", map_location="cpu")
    state1 = torch.load(out_dir / "rank1_state.pt", map_location="cpu")

    assert state0.keys() == state1.keys()
    for key in state0:
        assert torch.equal(state0[key], state1[key]), f"parameter {key!r} diverged across ranks"


def test_weight_sync_holds_over_many_steps(tmp_path):
    # Run longer to make sure sync isn't a one-step fluke.
    out_dir = _run_ddp(tmp_path, master_port="29514", steps=60, log_every=1000)

    state0 = torch.load(out_dir / "rank0_state.pt", map_location="cpu")
    state1 = torch.load(out_dir / "rank1_state.pt", map_location="cpu")
    for key in state0:
        assert torch.equal(state0[key], state1[key])


def test_three_ranks_shard_disjointly_and_sync(tmp_path):
    out_dir = _run_ddp(tmp_path, world_size=3, master_port="29515", steps=10)

    hists = [
        json.loads((out_dir / f"rank{r}_history.json").read_text()) for r in range(3)
    ]
    ranges = sorted((h["shard_start"], h["shard_end"]) for h in hists)
    # Every shard's end is the next shard's start: a clean, non-overlapping
    # partition of the full corpus across all 3 ranks.
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1

    states = [torch.load(out_dir / f"rank{r}_state.pt", map_location="cpu") for r in range(3)]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key])
        assert torch.equal(states[0][key], states[2][key])
