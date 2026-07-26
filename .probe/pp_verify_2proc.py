"""Throwaway verification script (not part of the test suite): launches
train_pp.py's train() across 2 real OS processes via mp.spawn, to confirm
the is_first/is_last (non-overlapping, 2-stage) branches -- which the
world_size=1 fix in train_pp.py/src/pipeline_model.py did NOT change --
still work after editing the step loop.

CORRECTED RESULT (2026-07-25): an earlier version of this script raised
`ModuleNotFoundError: No module named 'train_pp'` because it didn't add
the repo root to sys.path when run from a subprocess/agent context (the
test suite's own test files do this explicitly, e.g. test_train_tp.py's
`sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`). That
import failure was mischaracterized as a rendezvous hang in an earlier
draft of this docstring -- it was not. With the sys.path fix below, 2-process
mp.spawn training runs correctly and quickly (real gloo collectives,
real cross-process send/recv of activations/gradients): rank 0 owns
layers[0:1), rank 1 owns layers[1:2), loss drops step over step. There is
no sandbox networking limitation for mp.spawn here -- this matches
test_train_ddp.py/test_train_fsdp.py/test_train_tp.py, all of which
already do real 2/3-process mp.spawn runs successfully in this same
environment.
"""
import os
import sys
from pathlib import Path

import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_pp import build_arg_parser, train


def worker(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    args = build_arg_parser().parse_args([
        "--steps", "10", "--batch-size", "8", "--block-size", "16",
        "--d-model", "24", "--n-layer", "2", "--n-head", "4",
        "--microbatches", "2", "--log-every", "5", "--backend", "gloo",
    ])
    train(args)


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
