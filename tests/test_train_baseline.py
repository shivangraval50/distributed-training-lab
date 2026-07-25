"""Real tests for the Phase 1 single-device baseline (train_baseline.py).

Runs the actual training loop (tiny model/data, few steps) on CPU -- this
must work in CI (no GPU there), so CPU correctness is the bar, not speed.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_baseline import build_arg_parser, pick_device, train  # noqa: E402


def _make_args(**overrides):
    args = build_arg_parser().parse_args([])
    defaults = dict(
        steps=6,
        batch_size=4,
        block_size=16,
        d_model=16,
        n_layer=1,
        n_head=2,
        device="cpu",
        log_every=100,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args


def test_pick_device_cpu_explicit():
    assert pick_device("cpu") == pick_device("cpu")
    import torch
    assert pick_device("cpu") == torch.device("cpu")


def test_pick_device_auto_never_raises():
    # Should never fail: falls back to cpu if no accelerator is present.
    device = pick_device("auto")
    assert device.type in ("cuda", "mps", "cpu")


def test_train_runs_on_cpu_and_returns_history():
    args = _make_args()
    result = train(args)

    assert result.device == "cpu"
    assert result.num_params > 0
    assert len(result.history) == args.steps

    for record in result.history:
        assert math.isfinite(record.loss)
        assert record.loss > 0  # cross-entropy over a real (non-trivial) vocab
        assert record.step_time_s >= 0
        assert record.tokens_per_sec > 0


def test_loss_decreases_over_more_steps():
    # Not a strict monotonic check (mini-batch noise), but over enough steps
    # on this trivially learnable synthetic corpus, loss should clearly drop.
    args = _make_args(steps=150, batch_size=8, block_size=24, d_model=24, n_layer=2)
    result = train(args)

    first_five_avg = sum(r.loss for r in result.history[:5]) / 5
    last_five_avg = sum(r.loss for r in result.history[-5:]) / 5
    assert last_five_avg < first_five_avg


def test_writes_csv_log(tmp_path):
    out_path = tmp_path / "log.csv"
    args = _make_args(out=str(out_path))
    train(args)

    assert out_path.exists()
    content = out_path.read_text()
    lines = content.strip().splitlines()
    assert lines[0] == "step,loss,step_time_s,tokens_per_sec"
    assert len(lines) == args.steps + 1  # header + one row per step


def test_writes_json_log(tmp_path):
    import json

    out_path = tmp_path / "log.json"
    args = _make_args(out=str(out_path))
    train(args)

    payload = json.loads(out_path.read_text())
    assert payload["device"] == "cpu"
    assert len(payload["history"]) == args.steps
