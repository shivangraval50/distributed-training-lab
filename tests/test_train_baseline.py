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
    result = train(args)

    payload = json.loads(out_path.read_text())
    assert payload["device"] == "cpu"
    assert len(payload["history"]) == args.steps
    # Full content check, not just presence: config round-trips, and per-step
    # numbers in the JSON match the in-memory result exactly.
    assert payload["num_params"] == result.num_params
    assert payload["config"]["steps"] == args.steps
    assert payload["config"]["batch_size"] == args.batch_size
    assert payload["total_time_s"] >= 0
    for row, record in zip(payload["history"], result.history):
        assert row["step"] == record.step
        assert row["loss"] == pytest.approx(record.loss)
        assert row["tokens_per_sec"] == pytest.approx(record.tokens_per_sec)


def test_csv_log_content_matches_history(tmp_path):
    # Parse the CSV for real (not just header/line-count) and check every
    # row's values against the in-memory history, with correct types.
    import csv

    out_path = tmp_path / "log.csv"
    args = _make_args(out=str(out_path))
    result = train(args)

    with out_path.open(newline="") as f:
        rows = list(csv.reader(f))

    header, data_rows = rows[0], rows[1:]
    assert header == ["step", "loss", "step_time_s", "tokens_per_sec"]
    assert len(data_rows) == len(result.history)
    for row, record in zip(data_rows, result.history):
        assert int(row[0]) == record.step
        assert float(row[1]) == pytest.approx(record.loss)
        assert float(row[2]) == pytest.approx(record.step_time_s)
        assert float(row[3]) == pytest.approx(record.tokens_per_sec)


def test_out_path_without_json_suffix_defaults_to_csv(tmp_path):
    # _write_log's branch is "if suffix == '.json' else csv" -- lock in that
    # an extensionless/unknown-suffix --out still produces a parseable CSV,
    # not something silently malformed.
    out_path = tmp_path / "log.dat"
    args = _make_args(out=str(out_path))
    train(args)

    lines = out_path.read_text().strip().splitlines()
    assert lines[0] == "step,loss,step_time_s,tokens_per_sec"
    assert len(lines) == args.steps + 1


def test_same_seed_is_reproducible():
    # Same seed/config should give bit-identical loss curves -- this is what
    # later phases (DDP/FSDP) will rely on to isolate "did parallelism change
    # the math" from "did we just get a different random draw".
    args_a = _make_args(seed=123)
    args_b = _make_args(seed=123)
    result_a = train(args_a)
    result_b = train(args_b)

    losses_a = [r.loss for r in result_a.history]
    losses_b = [r.loss for r in result_b.history]
    assert losses_a == pytest.approx(losses_b)


def test_different_seeds_diverge():
    args_a = _make_args(seed=1)
    args_b = _make_args(seed=2)
    result_a = train(args_a)
    result_b = train(args_b)

    losses_a = [r.loss for r in result_a.history]
    losses_b = [r.loss for r in result_b.history]
    assert losses_a != pytest.approx(losses_b)


def test_cli_arg_parsing_defaults():
    args = build_arg_parser().parse_args([])
    assert args.steps == 20
    assert args.batch_size == 8
    assert args.block_size == 32
    assert args.d_model == 32
    assert args.n_layer == 2
    assert args.n_head == 2
    assert args.dropout == 0.0
    assert args.lr == pytest.approx(3e-4)
    assert args.device == "auto"
    assert args.out is None


def test_cli_arg_parsing_overrides():
    args = build_arg_parser().parse_args(
        [
            "--steps", "5",
            "--batch-size", "2",
            "--block-size", "8",
            "--d-model", "8",
            "--n-layer", "1",
            "--n-head", "1",
            "--dropout", "0.1",
            "--lr", "0.01",
            "--seed", "7",
            "--log-every", "1",
            "--device", "cpu",
            "--out", "run.csv",
        ]
    )
    assert args.steps == 5
    assert args.batch_size == 2
    assert args.block_size == 8
    assert args.d_model == 8
    assert args.n_layer == 1
    assert args.n_head == 1
    assert args.dropout == pytest.approx(0.1)
    assert args.lr == pytest.approx(0.01)
    assert args.seed == 7
    assert args.log_every == 1
    assert args.device == "cpu"
    assert args.out == "run.csv"


def test_cli_rejects_invalid_device_choice():
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(["--device", "tpu"])
