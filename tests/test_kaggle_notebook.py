"""Checks for notebooks/kaggle_single_gpu_baseline.py.

This script targets a remote single-GPU Kaggle runtime and cannot actually be
executed end-to-end here (macOS, no CUDA). What we *can* verify locally:
  1. It's syntactically valid Python (compiles).
  2. Its explicit "refuse to silently fall back to CPU" guard really fires
     when no CUDA device is visible -- which is exactly this environment, so
     this is a real behavioral check, not a stub.
"""

import ast
import importlib
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "notebooks" / "kaggle_single_gpu_baseline.py"

sys.path.insert(0, str(REPO_ROOT))


def test_script_is_syntactically_valid_python():
    source = SCRIPT_PATH.read_text()
    ast.parse(source)  # raises SyntaxError if invalid


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="guard only fires when CUDA is unavailable"
)
def test_main_refuses_to_run_without_cuda():
    kaggle_script = importlib.import_module(
        "notebooks.kaggle_single_gpu_baseline"
    )
    with pytest.raises(SystemExit, match="No CUDA device visible"):
        kaggle_script.main()
