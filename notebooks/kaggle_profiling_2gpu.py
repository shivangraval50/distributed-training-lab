"""Phase 5 remote run: real profiling (comms overhead, memory, scaling) of
all five strategies across Kaggle's 2xT4 (nccl backend).

This script is NOT run locally -- this machine (macOS, no CUDA) can only
prove the profiler MECHANISM works (see tests/test_profiling.py and
src/profiling.py's module docstring: the CPU/gloo smoke test shows the
classifier genuinely tells a no-communication baseline apart from a real
distributed strategy's allreduce/all_gather/reduce_scatter/send-recv, with
comm_fraction == 0.0 for baseline and > 0.0 for every distributed strategy
tested). It CANNOT produce a real scaling-efficiency, comms-overhead-%, or
memory-savings NUMBER -- that requires actual GPU throughput/memory, which
does not exist on a CPU-only machine.

This is the ready-to-run script for the *real* GPU numbers: launches
`profile_run.py` (which itself reuses train_baseline.py/train_ddp.py/
train_fsdp.py/train_tp.py/train_pp.py's own train() unmodified -- see that
script's docstring) for all five strategies with the SAME model config, on
Kaggle's real 2xT4, nccl backend, and assembles:
  - avg per-step wall-clock time per strategy (real, from each train_*.py's
    own StepRecord.step_time_s -- not the profiler's own aggregate, which
    spans the WHOLE profiled call, construction included)
  - comms-overhead %: `profile_result.comm_fraction` (fraction of profiled
    self-CPU-equivalent time -- CUDA activity is also captured when
    available -- spent in collective ops) per strategy
  - peak CUDA memory per rank (torch.cuda.max_memory_allocated(), reset
    immediately before the profiled call) per strategy
  - a naive throughput-based "speedup vs single-GPU baseline" number

--------------------------------------------------------------------------
IMPORTANT caveat on "speedup", stated here rather than glossed over: DDP and
FSDP are DATA-parallel (each rank processes its OWN local batch every step,
so their real global throughput is the SUM of both ranks' tokens/sec --
same convention as notebooks/kaggle_ddp_2gpu.py/kaggle_fsdp_2gpu.py). TP and
PP are MODEL-parallel (both ranks collaborate on ONE shared global batch per
step, so a rank's own tokens/sec already IS the global throughput -- summing
across ranks would double-count). This script combines per-strategy
throughput accordingly, but a "speedup vs baseline" computed this way still
is NOT an apples-to-apples FLOPs/dollar comparison across strategies (DDP's
global batch is 2x baseline's per step; TP/PP's is not) -- it answers "how
many tokens/sec does this configuration of this strategy achieve", not "which
strategy is fundamentally more efficient at a fixed global batch size". A
true iso-batch comparison is a further TODO past this phase.
--------------------------------------------------------------------------
How to run this on Kaggle (exact steps):
--------------------------------------------------------------------------
1. https://www.kaggle.com/code -> New Notebook.
2. Settings (right panel) -> Accelerator -> "GPU T4 x2".
3. In the first cell, clone the repo:
       !git clone https://github.com/shivangraval50/distributed-training-lab.git
       %cd distributed-training-lab
4. In the next cell, run this script:
       !python notebooks/kaggle_profiling_2gpu.py
   (it shells out to `profile_run.py` directly for the single-GPU baseline,
   and via `torchrun --standalone --nproc_per_node=2` for ddp/fsdp/tp/pp.)
5. Per-rank stdout (comm_fraction, comm/compute self time, wall time, peak
   CPU/CUDA memory) is printed live for each strategy. A combined JSON
   summary (all 5 strategies) is written to
   /kaggle/working/profiling_2gpu_summary.json. Paste the real numbers into
   this repo's README.md Results table -- do NOT hand-edit numbers into
   README without a log to back them up.
--------------------------------------------------------------------------

TODO (tracked in PLAN.md / README.md): this script has not yet been executed
on real GPUs. No scaling-efficiency/comms-overhead-%/memory number exists
anywhere in this repo until it has. Do not treat any number in this file as
measured -- there are none; only the config below is fixed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# Modest-but-real config, consistent with the other kaggle_*.py scripts'
# --d-model/--n-layer/--n-head, but fewer steps than those scripts' 500:
# torch.profiler's CPU+CUDA activity tracing has real per-op recording
# overhead, and this phase cares about the op-time BREAKDOWN and peak
# memory, not a long loss curve (train_*.py's own 500-step scripts already
# cover that). 100 steps is enough for the profiler's steady-state window
# (after its own internal warmup) to be representative.
STEPS = 100
SHARED_ARGS = [
    "--steps", str(STEPS),
    "--batch-size", "64",
    "--block-size", "128",
    "--d-model", "256",
    "--n-layer", "4",
    "--n-head", "4",
    "--lr", "3e-4",
    "--log-every", "1000000",
]

# Data-parallel strategies: each rank's own tokens/sec is a LOCAL number;
# real global throughput sums across ranks (both ranks process disjoint
# data concurrently). Model-parallel strategies: both ranks collaborate on
# one shared global batch per step, so a rank's own tokens/sec already IS
# the global number -- see module docstring's throughput caveat.
_DATA_PARALLEL_STRATEGIES = {"ddp", "fsdp"}
_MODEL_PARALLEL_STRATEGIES = {"tp", "pp"}


def _run_baseline(out_dir: Path) -> dict:
    """Single-GPU baseline: plain python (no torchrun), pinned to GPU 0,
    mirroring notebooks/kaggle_single_gpu_baseline.py's CUDA_VISIBLE_DEVICES
    restriction so this is a true single-GPU number, not an accident of not
    calling dist.init_process_group."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        sys.executable, str(REPO_ROOT / "profile_run.py"),
        "--strategy", "baseline",
        *SHARED_ARGS,
        "--device", "cuda",
        "--out-dir", str(out_dir),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        raise SystemExit(f"baseline profiling run failed with exit code {result.returncode}")
    return json.loads((out_dir / "profile_baseline_rank0.json").read_text())


def _run_distributed(strategy: str, out_dir: Path, extra_args: list[str] | None = None) -> list[dict]:
    """ddp/fsdp/tp/pp: real 2-process torchrun launch, nccl backend."""
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=2",
        str(REPO_ROOT / "profile_run.py"),
        "--strategy", strategy,
        *SHARED_ARGS,
        "--backend", "nccl",
        "--out-dir", str(out_dir),
        *(extra_args or []),
    ]
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"torchrun {strategy} profiling run failed with exit code {result.returncode}")
    return [
        json.loads((out_dir / f"profile_{strategy}_rank{r}.json").read_text())
        for r in range(2)
    ]


def _combined_tokens_per_sec(strategy: str, summaries: list[dict]) -> float | None:
    tokens_per_step = 64 * 128  # SHARED_ARGS: --batch-size 64 --block-size 128
    step_times = [s.get("avg_step_time_s") for s in summaries if s.get("avg_step_time_s")]
    if not step_times:
        return None
    per_rank_tps = [tokens_per_step / t for t in step_times]
    if strategy in _DATA_PARALLEL_STRATEGIES:
        return sum(per_rank_tps)
    # Model-parallel (tp/pp) or single-rank baseline: ranks collaborate on
    # one shared global batch per step -- do not sum, use the (identical,
    # in lockstep) per-rank number.
    return per_rank_tps[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device visible. On Kaggle: Settings -> Accelerator -> "
            "GPU T4 x2, then Save & re-run. This script refuses to silently "
            "fall back to CPU/gloo for a 'real 2xT4 profiling' run -- that's "
            "what tests/test_profiling.py already covers locally (profiler "
            "mechanism correctness, not GPU numbers)."
        )
    visible = torch.cuda.device_count()
    if visible != 2:
        raise SystemExit(
            f"Expected exactly 2 visible GPUs, got {visible}. This phase "
            f"measures profiling/scaling across Kaggle's 2xT4 specifically -- "
            f"re-check Settings -> Accelerator -> 'GPU T4 x2'."
        )

    out_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else REPO_ROOT / "notebooks"

    print("=== Phase 5: baseline (1 GPU) ===")
    baseline_summary = _run_baseline(out_root / "profile_baseline")

    print("\n=== Phase 5: DDP (2 GPUs, nccl) ===")
    ddp_summaries = _run_distributed("ddp", out_root / "profile_ddp")

    print("\n=== Phase 5: FSDP (2 GPUs, nccl, FULL_SHARD) ===")
    fsdp_summaries = _run_distributed(
        "fsdp", out_root / "profile_fsdp", ["--sharding-strategy", "FULL_SHARD"]
    )

    print("\n=== Phase 5: TP (2 GPUs, nccl) ===")
    tp_summaries = _run_distributed("tp", out_root / "profile_tp")

    print("\n=== Phase 5: PP (2 GPUs, nccl) ===")
    pp_summaries = _run_distributed("pp", out_root / "profile_pp", ["--microbatches", "4"])

    all_summaries = {
        "baseline": [baseline_summary],
        "ddp": ddp_summaries,
        "fsdp": fsdp_summaries,
        "tp": tp_summaries,
        "pp": pp_summaries,
    }

    baseline_tps = _combined_tokens_per_sec("baseline", [baseline_summary])

    print()
    print("=== Phase 5 profiling (2xT4, nccl): summary (paste into README.md) ===")
    print(f"gpus={[torch.cuda.get_device_name(i) for i in range(2)]} steps={STEPS}")

    report = {}
    for strategy, summaries in all_summaries.items():
        combined_tps = _combined_tokens_per_sec(strategy, summaries)
        speedup_vs_baseline = (
            combined_tps / baseline_tps if combined_tps and baseline_tps else None
        )
        comm_fractions = [s.get("comm_fraction") for s in summaries]
        peak_cuda = [s.get("peak_cuda_memory_bytes") for s in summaries]
        report[strategy] = {
            "combined_tokens_per_sec": combined_tps,
            "speedup_vs_single_gpu_baseline": speedup_vs_baseline,
            "comm_fraction_per_rank": comm_fractions,
            "peak_cuda_memory_bytes_per_rank": peak_cuda,
        }
        print(
            f"[{strategy:8s}] combined_tok/s={combined_tps} "
            f"speedup_vs_baseline={speedup_vs_baseline} "
            f"comm_fraction_per_rank={comm_fractions} "
            f"peak_cuda_mem_bytes_per_rank={peak_cuda}"
        )

    print()
    print("NOTE: 'speedup_vs_single_gpu_baseline' for tp/pp compares a "
          "MODEL-parallel strategy's single shared-batch throughput against "
          "baseline's -- NOT an iso-global-batch comparison against ddp/fsdp "
          "(whose global batch is 2x baseline's). See module docstring's "
          "caveat before treating any cross-strategy ranking as a clean "
          "efficiency comparison.")

    summary_path = out_root / "profiling_2gpu_summary.json"
    summary_path.write_text(json.dumps({"gpus": [torch.cuda.get_device_name(i) for i in range(2)], "steps": STEPS, "report": report, "raw": all_summaries}, indent=2))
    print(f"summary written to: {summary_path}")


if __name__ == "__main__":
    main()
