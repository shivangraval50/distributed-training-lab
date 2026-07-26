"""Phase 4a remote run: real tensor parallelism (TP) across Kaggle's 2xT4
(nccl backend).

This script is NOT run locally -- this machine (macOS, no CUDA) can only
verify TP *correctness* via CPU/gloo (see tests/test_train_tp.py and
train_tp.py's module docstring for the Step 1 finding: the real
`torch.distributed.tensor` DTensor API -- `ColwiseParallel`/`RowwiseParallel`
+ `DeviceMesh` -- works and is numerically correct on CPU/gloo, but only
against explicit `nn.Linear` submodules, which is why TinyGPT's attention
(built on `nn.MultiheadAttention`) is trained here via the separate,
numerically-equivalent `TPTinyGPT` class instead).

Same underlying architecture (equivalent to TinyGPT, verified in
tests/test_train_tp.py), same real DTensor sharding code path
(train_tp.py), but launched with `torchrun --nproc_per_node=2` across the
two real T4s with the `nccl` backend, so Phase 5 (profiling/scaling
writeup) has a real number for "what does MODEL (not data) parallelism cost/
buy at this scale" to compare against DDP/FSDP.

KEY DIFFERENCE FROM kaggle_ddp_2gpu.py / kaggle_fsdp_2gpu.py: TP shards the
MODEL, not the data. Every rank sees the IDENTICAL batch every step, so
--batch-size below is the GLOBAL batch size (NOT per-rank, NOT sharded) --
see train_tp.py's own argparse help text. Per-rank loss histories are
expected to be BIT-IDENTICAL across ranks (proof the sharded computation
recombines correctly via DTensor's implicit all_reduce), the opposite
invariant from DDP/FSDP where per-rank losses differ (disjoint data).

--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2".
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, launch the real 2-GPU TP run with torchrun:
       !python notebooks/kaggle_tp_2gpu.py
   (this script shells out to `torchrun --standalone --nproc_per_node=2
   train_tp.py ...` itself.)
5. Per-rank stdout (loss/timing/throughput, local-shard param fraction, the
   pre-training forward-equivalence check vs a single-process reference,
   and the cross-rank identical-loss / identical-weights checks) is printed
   live. A combined JSON summary is written to
   /kaggle/working/tp_2gpu_summary.json. Paste the real numbers into this
   repo's README.md Results table -- do NOT hand-edit numbers into README
   without a log to back them up.
6. For a same-scale comparison against DDP/FSDP, also run
   notebooks/kaggle_ddp_2gpu.py and notebooks/kaggle_fsdp_2gpu.py in the
   same session. This script does NOT hard-code or assume either number.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on real GPUs. No multi-GPU throughput/scaling numbers exist anywhere in this
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


# Same modest-but-real config as train_tp.py's own "real run" docstring
# example, so this file introduces no undocumented config drift.
# NOTE: --batch-size here is the GLOBAL batch size (every rank sees the
# IDENTICAL batch, unlike DDP/FSDP's per-rank sharded batch) -- see
# train_tp.py's argparse help text.
TP_ARGS = [
    "--steps", "500",
    "--batch-size", "64",
    "--block-size", "128",
    "--d-model", "256",
    "--n-layer", "4",
    "--n-head", "8",
    "--lr", "3e-4",
    "--log-every", "25",
    "--backend", "nccl",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script refuses to silently "
            "fall back to CPU/gloo for a 'real 2xT4 TP' run -- that's what "
            "tests/test_train_tp.py already covers locally (sharding "
            "correctness, not GPU speed)."
        )
    visible = torch.cuda.device_count()
    if visible != 2:
        raise SystemExit(
            f"Expected exactly 2 visible GPUs, got {visible}. This phase "
            f"measures TP scaling across Kaggle's 2xT4 specifically -- "
            f"re-check Settings -> Accelerator -> 'GPU T4 x2'."
        )

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"
    tp_out_dir = out_dir / "tp_run_artifacts"

    print("=== Phase 4a: launching real 2-GPU TP run (torchrun, nccl, DTensor) ===")
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2",
        str(REPO_ROOT / "train_tp.py"),
        *TP_ARGS,
        "--out-dir", str(tp_out_dir),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"torchrun TP run failed with exit code {result.returncode}")

    rank0_history = json.loads((tp_out_dir / "rank0_history.json").read_text())
    rank1_history = json.loads((tp_out_dir / "rank1_history.json").read_text())

    def _avg_tps(history: dict) -> float:
        steps = history["history"]
        return sum(r["tokens_per_sec"] for r in steps) / len(steps)

    avg_tps_rank0 = _avg_tps(rank0_history)
    avg_tps_rank1 = _avg_tps(rank1_history)
    # NOTE: unlike DDP/FSDP (data-parallel, so per-rank tok/s summed), TP
    # ranks cooperate on the SAME batch every step -- combined throughput is
    # NOT simply rank0+rank1 tok/s (that would double-count the same
    # tokens). We report both ranks' local step-time-derived tok/s as-is
    # (should be near-identical, since both do the same amount of wall-clock
    # work per step) rather than fabricate a "combined" number that doesn't
    # correspond to anything real for model-parallel throughput.
    losses_identical = rank0_history["history"] == rank1_history["history"] or all(
        abs(a["loss"] - b["loss"]) < 1e-4
        for a, b in zip(rank0_history["history"], rank1_history["history"])
    )

    print()
    print("=== Phase 4a TP (2xT4, nccl, DTensor Colwise/Rowwise): summary (paste into README.md) ===")
    print(f"world_size=2 gpus={[torch.cuda.get_device_name(i) for i in range(2)]}")
    print(f"full_params: {rank0_history['full_params']:,}")
    print(
        f"local_shard_numel: rank0={rank0_history['local_shard_numel']:,} "
        f"rank1={rank1_history['local_shard_numel']:,} "
        f"(embeddings/norms/head stay replicated -- only attn+MLP are sharded, "
        f"so this is expected to be well above 50% of full_params, not ~50%)"
    )
    print(f"steps: {len(rank0_history['history'])}")
    print(f"final_loss rank0={rank0_history['history'][-1]['loss']:.4f} "
          f"rank1={rank1_history['history'][-1]['loss']:.4f} "
          f"(TP invariant: these must match -- checked per-step already by "
          f"train_tp.py's own cross-rank assertion, re-derived here as a sanity check: "
          f"{'match' if losses_identical else 'MISMATCH -- see train_tp.py stdout above'})")
    print(f"per-rank tok/s (NOT summed -- both ranks cooperate on the same "
          f"batch, unlike DDP/FSDP): rank0={avg_tps_rank0:,.0f} rank1={avg_tps_rank1:,.0f}")
    print()
    print("NOTE: this script does not compute a 'TP speedup' number -- TP's value "
          "proposition is fitting a model too large for one GPU's memory, not "
          "raw throughput at a scale this small already fits on. A real "
          "memory-headroom comparison vs single-GPU/DDP/FSDP is TODO until "
          "peak-memory instrumentation exists across all four scripts.")

    summary = {
        "world_size": 2,
        "backend": "nccl",
        "parallelism": "tensor (DTensor Colwise/Rowwise, TPTinyGPT)",
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "full_params": rank0_history["full_params"],
        "local_shard_numel": {
            "rank0": rank0_history["local_shard_numel"],
            "rank1": rank1_history["local_shard_numel"],
        },
        "steps": len(rank0_history["history"]),
        "final_loss_per_rank": {
            "rank0": rank0_history["history"][-1]["loss"],
            "rank1": rank1_history["history"][-1]["loss"],
        },
        "losses_identical_across_ranks": losses_identical,
        "avg_tokens_per_sec_per_rank": {"rank0": avg_tps_rank0, "rank1": avg_tps_rank1},
    }
    summary_path = out_dir / "tp_2gpu_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
