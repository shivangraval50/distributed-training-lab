"""Phase 3 remote run: real FSDP across Kaggle's 2xT4 (nccl backend).

This script is NOT run locally -- this machine (macOS, no CUDA) can only
verify FSDP *correctness/sharding* via CPU/gloo (see tests/test_train_fsdp.py
and train_fsdp.py's module docstring for the exact CPU-only workarounds that
were needed: `device_id=torch.device("cpu")` to dodge an MPS-accelerator
auto-detection bug, and a manual `dist.broadcast` instead of
`sync_module_states=True`, which requires GPU tensors in this torch version).

On real GPUs neither workaround should be necessary: this script uses
`sync_module_states=True` (FSDP's own real broadcast) and per-rank CUDA
`device_id`s -- the code path this repo's CPU run could NOT exercise, so
Kaggle also validates the "normal" GPU path, not just the CPU workaround
path.

Same model (TinyGPT), same FSDP code path (train_fsdp.py), same
shard-per-rank data splitting as Phase 2's kaggle_ddp_2gpu.py, but wrapped in
real `FullyShardedDataParallel` (ZeRO-3-style: params + grads + optimizer
state all sharded) instead of `DistributedDataParallel` (full replication).
Compares against both the single-GPU baseline and DDP once all three have
been run, for a real memory/throughput picture of "what does sharding
actually buy you at this (tiny) scale" -- Phase 5's honest writeup.

--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2".
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, launch the real 2-GPU FSDP run with torchrun:
       !python notebooks/kaggle_fsdp_2gpu.py
   (this script shells out to `torchrun --standalone --nproc_per_node=2
   train_fsdp.py ...` itself.)
5. Per-rank stdout (loss/timing/throughput, local-shard param count, and the
   full-parameter sync check) is printed live. Also prints
   `torch.cuda.max_memory_allocated()` per rank -- the real number that
   should show FSDP's per-GPU memory savings vs DDP (each rank only holding
   ~1/world_size of params+grads+optimizer state) at this model size. A
   combined JSON summary is written to
   /kaggle/working/fsdp_2gpu_summary.json. Paste the real numbers into this
   repo's README.md Results table -- do NOT hand-edit numbers into README
   without a log to back them up.
6. For the memory comparison to mean anything, also run
   notebooks/kaggle_ddp_2gpu.py in the same session (same model config) and
   compare its per-rank `torch.cuda.max_memory_allocated()` against this
   script's. Neither script hard-codes or assumes the other's number.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on real GPUs. No multi-GPU throughput/memory numbers exist anywhere in this
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


# Same modest-but-real config as notebooks/kaggle_ddp_2gpu.py, so the DDP vs
# FSDP comparison isn't confounded by a config change. --batch-size is the
# PER-RANK (per-GPU) batch size, consistent with train_fsdp.py's docs.
FSDP_ARGS = [
    "--steps", "500",
    "--batch-size", "64",
    "--block-size", "128",
    "--d-model", "256",
    "--n-layer", "4",
    "--n-head", "4",
    "--lr", "3e-4",
    "--log-every", "25",
    "--backend", "nccl",
    "--sharding-strategy", "FULL_SHARD",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script refuses to silently "
            "fall back to CPU/gloo for a 'real 2xT4 FSDP' run -- that's what "
            "tests/test_train_fsdp.py already covers locally (sharding "
            "correctness, not GPU memory/speed)."
        )
    visible = torch.cuda.device_count()
    if visible != 2:
        raise SystemExit(
            f"Expected exactly 2 visible GPUs, got {visible}. This phase "
            f"measures FSDP scaling/memory across Kaggle's 2xT4 specifically -- "
            f"re-check Settings -> Accelerator -> 'GPU T4 x2'."
        )

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"
    fsdp_out_dir = out_dir / "fsdp_run_artifacts"

    print("=== Phase 3: launching real 2-GPU FSDP run (torchrun, nccl, FULL_SHARD) ===")
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2",
        str(REPO_ROOT / "train_fsdp.py"),
        *FSDP_ARGS,
        "--out-dir", str(fsdp_out_dir),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"torchrun FSDP run failed with exit code {result.returncode}")

    rank0_history = json.loads((fsdp_out_dir / "rank0_history.json").read_text())
    rank1_history = json.loads((fsdp_out_dir / "rank1_history.json").read_text())

    def _avg_tps(history: dict) -> float:
        steps = history["history"]
        return sum(r["tokens_per_sec"] for r in steps) / len(steps)

    avg_tps_rank0 = _avg_tps(rank0_history)
    avg_tps_rank1 = _avg_tps(rank1_history)
    combined_tps = avg_tps_rank0 + avg_tps_rank1

    print()
    print("=== Phase 3 FSDP (2xT4, nccl, FULL_SHARD): summary (paste into README.md) ===")
    print(f"world_size=2 gpus={[torch.cuda.get_device_name(i) for i in range(2)]}")
    print(f"full_params: {rank0_history['full_params']:,}")
    print(
        f"local_shard_numel: rank0={rank0_history['local_shard_numel']:,} "
        f"rank1={rank1_history['local_shard_numel']:,} "
        f"(each should be ~1/2 of full_params if sharding is real)"
    )
    print(
        f"optimizer_state_numel: rank0={rank0_history['optimizer_state_numel']:,} "
        f"rank1={rank1_history['optimizer_state_numel']:,}"
    )
    print(f"steps: {len(rank0_history['history'])}")
    print(f"final_loss rank0={rank0_history['history'][-1]['loss']:.4f} "
          f"rank1={rank1_history['history'][-1]['loss']:.4f}")
    print(f"avg tokens/sec: rank0={avg_tps_rank0:,.0f} rank1={avg_tps_rank1:,.0f} "
          f"combined={combined_tps:,.0f}")
    print()
    print("NOTE: this script does not itself capture torch.cuda.max_memory_allocated() "
          "per rank (train_fsdp.py doesn't instrument that yet) -- TODO: add "
          "peak-memory logging to train_fsdp.py/train_ddp.py before treating any "
          "cross-strategy memory-savings number as measured. Until then, only "
          "sharding correctness (params/optimizer-state fraction) and "
          "throughput are real numbers here; memory comparison vs DDP is TODO.")
    print("NOTE: to compute a real speedup-vs-single-GPU or vs-DDP number, also "
          "run notebooks/kaggle_single_gpu_baseline.py and "
          "notebooks/kaggle_ddp_2gpu.py in the same session. This script does "
          "NOT hard-code or assume either number.")

    summary = {
        "world_size": 2,
        "backend": "nccl",
        "sharding_strategy": "FULL_SHARD",
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "full_params": rank0_history["full_params"],
        "local_shard_numel": {
            "rank0": rank0_history["local_shard_numel"],
            "rank1": rank1_history["local_shard_numel"],
        },
        "optimizer_state_numel": {
            "rank0": rank0_history["optimizer_state_numel"],
            "rank1": rank1_history["optimizer_state_numel"],
        },
        "steps": len(rank0_history["history"]),
        "avg_tokens_per_sec_per_rank": {"rank0": avg_tps_rank0, "rank1": avg_tps_rank1},
        "avg_tokens_per_sec_combined": combined_tps,
        "final_loss_per_rank": {
            "rank0": rank0_history["history"][-1]["loss"],
            "rank1": rank1_history["history"][-1]["loss"],
        },
    }
    summary_path = out_dir / "fsdp_2gpu_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
