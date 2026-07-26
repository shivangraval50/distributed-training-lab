"""Checks for notebooks/kaggle_profiling_2gpu.py.

This script targets a remote 2xT4 Kaggle runtime and cannot actually be
executed end-to-end here (macOS, no CUDA, one process not two GPUs). What we
*can* verify locally, mirroring tests/test_kaggle_fsdp_notebook.py:
  1. It's syntactically valid Python (compiles).
  2. Its explicit "refuse to run without 2 real GPUs" guards really fire in
     this environment (no CUDA at all here) -- a real behavioral check, not
     a stub.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "notebooks" / "kaggle_profiling_2gpu.py"

sys.path.insert(0, str(REPO_ROOT))


def test_script_is_syntactically_valid_python():
    source = SCRIPT_PATH.read_text()
    ast.parse(source)  # raises SyntaxError if invalid


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="guard only fires when CUDA is unavailable"
)
def test_main_refuses_to_run_without_cuda():
    kaggle_script = importlib.import_module("notebooks.kaggle_profiling_2gpu")
    with pytest.raises(SystemExit, match="No CUDA device visible"):
        kaggle_script.main()


def test_main_refuses_to_run_without_exactly_two_gpus(monkeypatch):
    """Same reasoning as test_kaggle_fsdp_notebook.py's equivalent:
    monkeypatch only the CUDA *detection* functions to simulate "CUDA
    available, but only 1 GPU visible" and confirm the real
    device_count()!=2 branch fires."""
    kaggle_script = importlib.import_module("notebooks.kaggle_profiling_2gpu")

    monkeypatch.setattr(kaggle_script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(kaggle_script.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(SystemExit, match="Expected exactly 2 visible GPUs, got 1"):
        kaggle_script.main()


def test_combined_tokens_per_sec_sums_for_data_parallel_but_not_model_parallel():
    """Real unit check of the throughput-combination logic itself (not
    gated on CUDA): DDP/FSDP sum per-rank tokens/sec (data-parallel, each
    rank processes disjoint data), TP/PP use a single rank's number
    unsummed (model-parallel, both ranks collaborate on one shared batch)."""
    kaggle_script = importlib.import_module("notebooks.kaggle_profiling_2gpu")

    summaries = [{"avg_step_time_s": 0.1}, {"avg_step_time_s": 0.1}]
    tokens_per_step = 64 * 128
    expected_per_rank = tokens_per_step / 0.1

    ddp_combined = kaggle_script._combined_tokens_per_sec("ddp", summaries)
    assert ddp_combined == pytest.approx(2 * expected_per_rank)

    tp_combined = kaggle_script._combined_tokens_per_sec("tp", summaries)
    assert tp_combined == pytest.approx(expected_per_rank)
