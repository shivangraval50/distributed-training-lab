"""Phase 1 remote run: single-GPU baseline on Kaggle (2xT4 instance, 1 GPU used).

This script is NOT run locally -- this machine (macOS, no CUDA) can only smoke
-test train_baseline.py on CPU/MPS. This is the ready-to-run script for the
*real* single-device number: same model, same synthetic data, same training
loop, on one T4, so Phase 2 (DDP across both T4s) has a real single-GPU
baseline to compare speedup against.

It deliberately restricts itself to ONE GPU even though the Kaggle instance
has two (2xT4) -- that's the point of "single-GPU baseline"; DDP across both
GPUs is a later phase.

--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2". (We'll only use one
   T4 in this phase; the second is idle. That's intentional and documented,
   not a bug.)
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, run this script:
       !python notebooks/kaggle_single_gpu_baseline.py
5. The script prints per-step loss/timing/throughput to stdout and writes a
   JSON log to /kaggle/working/baseline_gpu_log.json (or ./baseline_gpu_log.json
   if not running on Kaggle's filesystem). Download that file (or copy the
   printed summary) and paste the real numbers into this repo's README.md
   Results table -- do NOT hand-edit numbers into README without a log to
   back them up.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on a real GPU. No throughput/loss numbers from a GPU run exist anywhere in
this repo until it has. Do not treat any number in this file as measured --
there are none; only the config below is fixed.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Restrict to exactly one GPU (device 0) *before* importing torch, so CUDA
# only ever sees one device -- this is what makes "single-GPU baseline" a
# true single-GPU run on a 2-GPU box, not an accident of not calling DDP.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from train_baseline import build_arg_parser, train  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script intentionally refuses "
            "to silently fall back to CPU for a 'single-GPU baseline' run."
        )
    visible = torch.cuda.device_count()
    if visible != 1:
        raise SystemExit(
            f"Expected exactly 1 visible GPU (CUDA_VISIBLE_DEVICES=0), got "
            f"{visible}. Refusing to run -- this phase measures a true "
            f"single-GPU baseline, not multi-GPU."
        )

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"
    out_path = out_dir / "baseline_gpu_log.json"

    # Modest-but-real config: big enough that a T4 is actually doing work
    # (not dominated by Python/launch overhead), small enough to finish in
    # minutes, not hours -- consistent with the project's "small by design"
    # thesis. Feel free to bump --steps for a longer curve.
    argv = [
        "--steps", "500",
        "--batch-size", "64",
        "--block-size", "128",
        "--d-model", "256",
        "--n-layer", "4",
        "--n-head", "4",
        "--lr", "3e-4",
        "--log-every", "25",
        "--device", "cuda",
        "--out", str(out_path),
    ]
    args = build_arg_parser().parse_args(argv)
    result = train(args)

    print()
    print("=== Phase 1 single-GPU baseline: summary (paste into README.md) ===")
    print(f"device: {result.device} (torch.cuda.get_device_name(0)="
          f"{torch.cuda.get_device_name(0)!r})")
    print(f"params: {result.num_params:,}")
    print(f"steps: {len(result.history)}  total_time_s: {result.total_time_s:.2f}")
    print(f"final_loss: {result.history[-1].loss:.4f}")
    avg_tps = sum(r.tokens_per_sec for r in result.history) / len(result.history)
    print(f"avg tokens/sec (post-warmup steps included): {avg_tps:,.0f}")
    print(f"log written to: {out_path}")


if __name__ == "__main__":
    main()
