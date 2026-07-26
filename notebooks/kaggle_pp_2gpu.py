"""Phase 4b remote run: real pipeline parallelism (PP) across Kaggle's 2xT4
(nccl backend).

This script is NOT run locally -- this machine (macOS, no CUDA) can only
verify PP *correctness* via CPU/gloo (see tests/test_train_pp.py and
train_pp.py's module docstring for the Step 1 finding: the real
`torch.distributed.pipelining` API -- `PipelineStage` + `ScheduleGPipe` --
works out of the box and is numerically correct on CPU/gloo against
TinyGPT's actual, unmodified architecture -- no rewrite needed, unlike
Phase 4a's TP, which required a Megatron-style attention rewrite because PP
splits at LAYER boundaries instead of inside `nn.MultiheadAttention`).

Same underlying architecture (TinyGPT, unmodified), same real
send/recv-based pipeline code path (train_pp.py), but launched with
`torchrun --nproc_per_node=2` across the two real T4s with the `nccl`
backend, so Phase 5 (profiling/scaling writeup) has a real number for
LAYER-split model parallelism to compare against DDP/FSDP/TP.

KEY DIFFERENCE FROM DDP/FSDP/TP: PP ranks own a strictly DISJOINT slice of
the model's LAYERS (not overlapping/replicated parameters, and not sharded
inside a layer). --batch-size below is the GLOBAL batch size, split into
`--microbatches` per step by `ScheduleGPipe` -- see train_pp.py's own
argparse help text.

ASYMMETRIC PER-RANK OUTPUT (IMPORTANT, unlike every other script in this
repo): only the LAST rank (stage `world_size - 1`) is the loss-computing
stage in `ScheduleGPipe`, so only its `history` entries have a real `loss`
value -- every other rank's `StepRecord.loss` is `None` (see train_pp.py's
`RunResult`/`StepRecord` and its step loop: `loss_val = None` on the
`elif is_first:` / `else:` branches). This script's summary logic reads
loss from the LAST rank's history file (`rank{world_size-1}_history.json`),
NOT rank 0 -- reading rank 0 here would silently produce `None`/a crash for
every "final_loss" style number a DDP/FSDP/TP script would compute from
rank 0.

--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2".
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, launch the real 2-GPU PP run with torchrun:
       !python notebooks/kaggle_pp_2gpu.py
   (this script shells out to `torchrun --standalone --nproc_per_node=2
   train_pp.py ...` itself.)
5. Per-rank stdout (loss only from the last stage, timing/throughput,
   local-stage param fraction and owned layer range) is printed live. A
   combined JSON summary is written to
   /kaggle/working/pp_2gpu_summary.json. Paste the real numbers into this
   repo's README.md Results table -- do NOT hand-edit numbers into README
   without a log to back them up.
6. For a same-scale comparison against DDP/FSDP/TP, also run
   notebooks/kaggle_ddp_2gpu.py, notebooks/kaggle_fsdp_2gpu.py, and
   notebooks/kaggle_tp_2gpu.py in the same session. This script does NOT
   hard-code or assume any of their numbers.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on real GPUs. No multi-GPU throughput/scaling numbers -- and no real
pipeline-bubble/overlap measurement -- exist anywhere in this repo until it
has. Do not treat any number in this file as measured -- there are none;
only the config below is fixed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402


# Same modest-but-real config as train_pp.py's own "real run" docstring
# example, so this file introduces no undocumented config drift.
# NOTE: --batch-size here is the GLOBAL batch size, split into
# --microbatches per step by ScheduleGPipe -- see train_pp.py's argparse
# help text.
PP_ARGS = [
    "--steps", "500",
    "--batch-size", "64",
    "--block-size", "128",
    "--d-model", "256",
    "--n-layer", "4",
    "--n-head", "4",
    "--microbatches", "4",
    "--lr", "3e-4",
    "--log-every", "25",
    "--backend", "nccl",
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script refuses to silently "
            "fall back to CPU/gloo for a 'real 2xT4 PP' run -- that's what "
            "tests/test_train_pp.py already covers locally (cross-process "
            "layer-split correctness, not GPU speed/overlap)."
        )
    visible = torch.cuda.device_count()
    if visible != 2:
        raise SystemExit(
            f"Expected exactly 2 visible GPUs, got {visible}. This phase "
            f"measures PP scaling across Kaggle's 2xT4 specifically -- "
            f"re-check Settings -> Accelerator -> 'GPU T4 x2'."
        )

    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"
    pp_out_dir = out_dir / "pp_run_artifacts"

    print("=== Phase 4b: launching real 2-GPU PP run (torchrun, nccl, ScheduleGPipe) ===")
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2",
        str(REPO_ROOT / "train_pp.py"),
        *PP_ARGS,
        "--out-dir", str(pp_out_dir),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"torchrun PP run failed with exit code {result.returncode}")

    world_size = 2
    last_rank = world_size - 1

    rank0_history = json.loads((pp_out_dir / "rank0_history.json").read_text())
    last_history = json.loads((pp_out_dir / f"rank{last_rank}_history.json").read_text())

    # IMPORTANT: rank 0 (the first stage) has `loss: null` for every step --
    # only the LAST rank is the loss-computing stage in ScheduleGPipe (see
    # this file's module docstring / train_pp.py's step loop). Reading loss
    # from rank 0 here would be wrong, not just less useful.
    assert all(r["loss"] is None for r in rank0_history["history"]), (
        "expected rank 0 (first stage, world_size=2) to have loss=None for "
        "every step -- if this fails, train_pp.py's stage assignment or "
        "this script's assumption about which rank computes loss is stale"
    )
    assert all(r["loss"] is not None for r in last_history["history"]), (
        f"expected rank {last_rank} (last stage) to have a real loss every "
        f"step -- got at least one None"
    )

    def _avg_tps(history: dict) -> float:
        steps = history["history"]
        return sum(r["tokens_per_sec"] for r in steps) / len(steps)

    avg_tps_rank0 = _avg_tps(rank0_history)
    avg_tps_last = _avg_tps(last_history)
    avg_loss = sum(r["loss"] for r in last_history["history"]) / len(last_history["history"])

    print()
    print("=== Phase 4b PP (2xT4, nccl, ScheduleGPipe, 2 stages): summary (paste into README.md) ===")
    print(f"world_size={world_size} gpus={[torch.cuda.get_device_name(i) for i in range(2)]}")
    print(f"full_params: {last_history['full_params']:,}")
    print(
        f"local_stage_params: rank0={rank0_history['local_stage_numel']:,} "
        f"(layers[{rank0_history['layer_start']}:{rank0_history['layer_end']}), stage 0) "
        f"rank{last_rank}={last_history['local_stage_numel']:,} "
        f"(layers[{last_history['layer_start']}:{last_history['layer_end']}), stage {last_rank})"
    )
    print(f"steps: {len(last_history['history'])} microbatches={PP_ARGS[PP_ARGS.index('--microbatches') + 1]}")
    print(
        f"final_loss (from last stage, rank {last_rank} -- the ONLY rank with a real "
        f"loss value; rank 0's is None by design, see module docstring): "
        f"{last_history['history'][-1]['loss']:.4f}"
    )
    print(f"avg tok/s per rank (each rank's own wall-clock per step -- NOT summed, "
          f"since ranks execute sequentially-dependent pipeline stages, not "
          f"independent replicas like DDP/FSDP): rank0={avg_tps_rank0:,.0f} "
          f"rank{last_rank}={avg_tps_last:,.0f}")
    print()
    print("NOTE: no pipeline-bubble/overlap number is computed here -- "
          "train_pp.py itself only measures each rank's total wall-clock per "
          "step (GPipe's fill+drain bubble is real but not separately "
          "isolated/measured yet). TODO before claiming any 'PP hides "
          "communication behind compute' number.")

    summary = {
        "world_size": world_size,
        "backend": "nccl",
        "parallelism": "pipeline (ScheduleGPipe, layer-split, unmodified TinyGPT)",
        "gpus": [torch.cuda.get_device_name(i) for i in range(2)],
        "full_params": last_history["full_params"],
        "local_stage_numel": {
            "rank0": rank0_history["local_stage_numel"],
            f"rank{last_rank}": last_history["local_stage_numel"],
        },
        "layer_ranges": {
            "rank0": [rank0_history["layer_start"], rank0_history["layer_end"]],
            f"rank{last_rank}": [last_history["layer_start"], last_history["layer_end"]],
        },
        "steps": len(last_history["history"]),
        "microbatches": PP_ARGS[PP_ARGS.index("--microbatches") + 1],
        "final_loss_last_stage": last_history["history"][-1]["loss"],
        "avg_loss_last_stage": avg_loss,
        "avg_tokens_per_sec_per_rank": {"rank0": avg_tps_rank0, f"rank{last_rank}": avg_tps_last},
        "loss_reported_from": f"rank{last_rank} (last stage) -- rank0/intermediate ranks have loss=None by design",
    }
    summary_path = out_dir / "pp_2gpu_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
