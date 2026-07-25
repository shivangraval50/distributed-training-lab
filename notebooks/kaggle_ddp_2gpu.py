"""Phase 2 remote run: real DDP across Kaggle's 2xT4 (nccl backend).

This script is NOT run locally -- this machine (macOS, no CUDA) can only
verify DDP *correctness* via CPU/gloo (see tests/test_train_ddp.py). This is
the ready-to-run script for the *real* multi-GPU number: same model
(TinyGPT), same DDP code path (train_ddp.py), same shard-per-rank data
splitting, but launched with `torchrun --nproc_per_node=2` across the two
real T4s with the `nccl` backend, so Phase 5 (profiling/scaling writeup) has
a real speedup-vs-single-GPU-baseline number to compare against
notebooks/kaggle_single_gpu_baseline.py.

--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2".
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, launch the real 2-GPU DDP run with torchrun (NOT plain
   python -- torchrun is what spawns the 2 worker processes, one per GPU):
       !python notebooks/kaggle_ddp_2gpu.py
   (this script shells out to `torchrun --standalone --nproc_per_node=2
   train_ddp.py ...` itself, and also runs the single-GPU baseline first for
   a same-session, same-hardware speedup comparison.)
5. Per-rank stdout (loss/timing/throughput) and the weight-sync check are
   printed live. A combined JSON summary (both runs' timings + the computed
   speedup) is written to /kaggle/working/ddp_2gpu_summary.json. Download it
   (or copy the printed summary) and paste the real numbers into this repo's
   README.md Results table -- do NOT hand-edit numbers into README without a
   log to back them up.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on real GPUs. No multi-GPU throughput/speedup numbers exist anywhere in this
repo until it has. Do not treat any number in this file as measured -- there
are none; only the config below is fixed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


# Same modest-but-real config as notebooks/kaggle_single_gpu_baseline.py, so
# the single-GPU vs 2-GPU comparison isn't confounded by a config change.
# NOTE: --batch-size here is the PER-RANK (per-GPU) batch size, consistent
# with train_ddp.py's docs -- so global batch size across 2 GPUs is 2x the
# single-GPU baseline's. This is standard DDP practice (each GPU processes
# its own local batch), and is exactly what a real speedup number should
# reflect: same per-GPU work, 2x the total tokens/sec if scaling is linear.
DDP_ARGS = [
    "--steps", "500",
    "--batch-size", "64",
    "--block-size", "128",
    "--d-model", "256",
    "--n-layer", "4",
    "--n-head", "4",
    "--lr", "3e-4",
    "--log-every", "25",
    "--backend", "nccl",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script refuses to silently "
            "fall back to CPU/gloo for a 'real 2xT4 DDP' run -- that's what "
            "tests/test_train_ddp.py already covers locally."
        )
    visible = torch.cuda.device_count()
    if visible != 2:
        raise SystemExit(
            f"Expected exactly 2 visible GPUs, got {visible}. This phase "
            f"measures DDP scaling across Kaggle's 2xT4 specifically -- "
            f"re-check Settings -> Accelerator -> 'GPU T4 x2'."
        )

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"
    ddp_out_dir = out_dir / "ddp_run_artifacts"

    print("=== Phase 2: launching real 2-GPU DDP run (torchrun, nccl) ===")
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2",
        str(REPO_ROOT / "train_ddp.py"),
        *DDP_ARGS,
        "--out-dir", str(ddp_out_dir),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"torchrun DDP run failed with exit code {result.returncode}")

    rank0_history = json.loads((ddp_out_dir / "rank0_history.json").read_text())
    rank1_history = json.loads((ddp_out_dir / "rank1_history.json").read_text())

    def _avg_tps(history: dict) -> float:
        steps = history["history"]
        return sum(r["tokens_per_sec"] for r in steps) / len(steps)

    avg_tps_rank0 = _avg_tps(rank0_history)
    avg_tps_rank1 = _avg_tps(rank1_history)
    # Real, global (2-GPU) throughput: sum of each rank's local tokens/sec,
    # since both ranks process their own batch concurrently every step.
    combined_tps = avg_tps_rank0 + avg_tps_rank1

    print()
    print("=== Phase 2 DDP (2xT4, nccl): summary (paste into README.md) ===")
    print(f"world_size=2 gpus={[torch.cuda.get_device_name(i) for i in range(2)]}")
    print(f"params: {rank0_history['num_params']:,}")
    print(f"steps: {len(rank0_history['history'])}")
    print(f"final_loss rank0={rank0_history['history'][-1]['loss']:.4f} "
          f"rank1={rank1_history['history'][-1]['loss']:.4f}")
    print(f"avg tokens/sec: rank0={avg_tps_rank0:,.0f} rank1={avg_tps_rank1:,.0f} "
          f"combined={combined_tps:,.0f}")
    print()
    print("NOTE: to compute a real speedup-vs-single-GPU number, also run "
          "notebooks/kaggle_single_gpu_baseline.py in the same session and "
          "divide combined_tps above by that script's avg tokens/sec. This "
          "script does NOT hard-code or assume that number -- TODO until "
          "both scripts have actually been run back to back on Kaggle.")

    summary = {
        "world_size": 2,
        "backend": "nccl",
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "num_params": rank0_history["num_params"],
        "steps": len(rank0_history["history"]),
        "avg_tokens_per_sec_per_rank": {"rank0": avg_tps_rank0, "rank1": avg_tps_rank1},
        "avg_tokens_per_sec_combined": combined_tps,
        "final_loss_per_rank": {
            "rank0": rank0_history["history"][-1]["loss"],
            "rank1": rank1_history["history"][-1]["loss"],
        },
    }
    summary_path = out_dir / "ddp_2gpu_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
