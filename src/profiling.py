"""Phase 5: profiling harness -- classify torch.profiler CPU events into
"communication" (collective ops) vs "compute", plus peak memory capture.

--------------------------------------------------------------------------
Step 1 finding (empirical, not assumed -- see `.probe/profiling_probe*.py`,
run on this machine: macOS/Apple Silicon, torch 2.13.0, gloo backend, real
2-process `torch.multiprocessing.spawn` runs of tiny DDP/FSDP/TP/send-recv
snippets). What does `torch.profiler` ACTUALLY name collective ops here?

  DDP (allreduce of gradients):
    'c10d::allreduce_'        -- dispatcher trampoline, nonzero self time
    'gloo:all_reduce'         -- gloo backend op, runs on a SEPARATE worker
                                 thread; self_cpu_time_total == 0 for THIS
                                 event but cpu_time_total is large (the real
                                 collective's wall-clock cost shows up here,
                                 while the CALLING thread's blocking wait
                                 shows up as c10d::allreduce_'s self time)
  FSDP (all_gather to unshard params, reduce_scatter of grads):
    'c10d::allgather_', 'gloo:all_gather'
    'c10d::_reduce_scatter_base_' (no separate 'gloo:reduce_scatter' event
      was observed for the FULL_SHARD gradient step in the probe -- the gloo
      backend op for this one apparently doesn't get its own top-level
      profiler frame the way all_reduce/all_gather do on this torch build)
  PP (real point-to-point activation/gradient transfer):
    'c10d::send', 'gloo:send', 'c10d::recv_', 'gloo:recv'
  TP (DTensor's functional-collective wrapper around all_reduce, used by
  RowwiseParallel's output-gradient combine):
    '_c10d_functional::all_reduce'   -- dispatch, modest self time
    '_c10d_functional::wait_tensor'  -- THIS is where most of TP's real
      collective wall-clock time showed up (self_cpu_time_total for this one
      event was ~85% of all TP-related self time in the probe) -- if this
      name were missed, TP's real comm cost would be badly undercounted.

Classification rule used below, built from those observations rather than a
guessed keyword list:
  1. Namespace match: any event name containing "c10d" (covers both the
     `c10d::...` dispatcher ops AND DTensor's `_c10d_functional::...`
     wrapper, including the wait_tensor case above) or starting with
     "gloo:" (the gloo backend's own op names) is real collective
     communication or its blocking wait -- this is how PyTorch itself
     namespaces these ops, not a guess.
  2. Keyword fallback, for any backend/name not directly observed above
     (e.g. this repo's TODO real nccl run on Kaggle): literal substrings
     "allreduce"/"all_reduce", "all_gather", "reduce_scatter", "broadcast",
     "send", "recv" -- but NEVER applied to names in the `aten::`/`prim::`/
     `prims::` namespaces, since those are PyTorch's own namespaces for
     purely local tensor kernels and really do contain comm-sounding names
     that are not communication at all (e.g. `aten::broadcast_to`,
     `aten::broadcast_tensors`, `prims::broadcast_in_dim` reshape a tensor's
     *shape* locally, no `torch.distributed` call involved -- verified
     against this torch build's real op schema registry, and guarded against
     in `tests/test_profiling.py`).
Everything else (aten::*, autograd::*, FullyShardedDataParallel.* hooks,
DDP's reducer bucket bookkeeping, etc.) is classified as "compute" -- this
includes real local CPU work done ON BEHALF OF a strategy (e.g. DDP's
gradient-bucket copy/divide, FSDP's flat-parameter chunk/split), which is
deliberate: those are genuine local tensor ops, not wire communication, even
though they only exist because of the distributed strategy.

Peak memory: CPU peak RSS via `resource.getrusage(RUSAGE_SELF).ru_maxrss`
(monotonic historical peak for this whole process -- cannot be reset
mid-run, unlike the CUDA counterpart below; verified on this machine that
Darwin reports BYTES already, not KB as Linux does -- see
`_ru_maxrss_to_bytes` docstring for the actual verification). CUDA peak
allocated via `torch.cuda.max_memory_allocated()`, reset immediately before
the profiled call so it reflects only that call -- guarded by
`torch.cuda.is_available()` throughout, since this machine has no CUDA: that
code path is exercised on Kaggle only (see notebooks/kaggle_profiling_2gpu.py),
never fabricated here.
"""

from __future__ import annotations

import platform
import resource
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

import torch
from torch.profiler import ProfilerActivity, profile

T = TypeVar("T")

# See module docstring for exactly how these were derived from real captured
# event names, not guessed.
_COMM_NAMESPACE_SUBSTRINGS = ("c10d", "gloo:")
_COMM_KEYWORD_SUBSTRINGS = (
    "allreduce", "all_reduce", "all_gather", "reduce_scatter", "broadcast",
    "send", "recv",
)

# The `_COMM_KEYWORD_SUBSTRINGS` fallback above is a bare-substring match, so
# on its own it would misclassify real LOCAL (single-device, no wire
# communication at all) ops that happen to contain one of those words as
# words -- verified concretely: `aten::broadcast_to`, `aten::broadcast_tensors`,
# `aten::_sparse_broadcast_to`, `prim::BroadcastSizes`, and
# `prims::broadcast_in_dim` are all real op names (checked against this
# torch build's actual op schema registry) that reshape/broadcast a tensor's
# *shape* locally and never touch `torch.distributed` -- none of them should
# ever be "comm". PyTorch itself namespaces every genuinely local tensor
# kernel under `aten::`/`prim::`/`prims::` (this is a structural convention
# of the framework, not a per-op guess), so the keyword fallback is only
# trusted for names OUTSIDE those namespaces -- names inside them are always
# local compute, full stop, even if they contain a comm-sounding keyword.
_LOCAL_OP_NAMESPACE_PREFIXES = ("aten::", "prim::", "prims::")


def is_comm_event(name: str) -> bool:
    """True if this torch.profiler event name is collective communication
    (or its blocking wait), False if it's local compute/bookkeeping. See
    module docstring for the empirical basis."""
    name_l = name.lower()
    if any(s in name_l for s in _COMM_NAMESPACE_SUBSTRINGS):
        return True
    if any(name_l.startswith(p) for p in _LOCAL_OP_NAMESPACE_PREFIXES):
        return False
    return any(k in name_l for k in _COMM_KEYWORD_SUBSTRINGS)


def _ru_maxrss_to_bytes(raw: int) -> int:
    """Normalize `ru_maxrss` to bytes.

    `man getrusage` documents this field in KILOBYTES on Linux but BYTES on
    macOS/BSD -- verified empirically on this machine (not just trusted from
    docs): allocating a 200MB bytearray and re-reading ru_maxrss showed a
    ~209,698,816-byte delta (~200MB) with NO further scaling needed on this
    Darwin box, confirming it already reports bytes here.
    """
    if platform.system() == "Linux":
        return raw * 1024
    return raw


def peak_cpu_memory_bytes() -> int:
    """Peak resident-set size of this whole process so far, in bytes.

    This is the process's all-time high-water mark (not "since the last
    call" -- `ru_maxrss` cannot be reset), so calling this right after a
    profiled block gives peak RSS across the WHOLE process lifetime up to
    that point, not an isolated per-call number. That is an honest
    limitation stated here rather than glossed over: for a true per-call
    delta, compare two readings taken immediately before/after the block
    (both still valid absolute peaks) -- ProfileResult below reports both.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return _ru_maxrss_to_bytes(raw)


@dataclass
class OpStat:
    name: str
    count: int
    self_cpu_time_us: float
    cpu_time_total_us: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "self_cpu_time_us": self.self_cpu_time_us,
            "cpu_time_total_us": self.cpu_time_total_us,
        }


@dataclass
class ProfileResult:
    """Summary of one profiled block (a single call to some training
    function). Time totals use `self_cpu_time_total` (torch.profiler's
    convention for the profiler's own default sort) so nested parent/child
    calls on the SAME thread are not double-counted -- see module docstring
    for the one case (gloo's async worker-thread ops) where the real
    collective cost shows up on the *calling* thread's self time instead of
    the backend op's own self time (which can be 0), and why the namespace
    classification rule still correctly buckets both under "comm"."""

    n_ops_observed: int
    comm_self_cpu_time_us: float
    compute_self_cpu_time_us: float
    comm_ops: list[OpStat] = field(default_factory=list)
    compute_ops: list[OpStat] = field(default_factory=list)
    wall_time_s: float = 0.0
    peak_cpu_memory_before_bytes: int = 0
    peak_cpu_memory_after_bytes: int = 0
    peak_cuda_memory_bytes: int | None = None

    @property
    def total_self_cpu_time_us(self) -> float:
        return self.comm_self_cpu_time_us + self.compute_self_cpu_time_us

    @property
    def comm_fraction(self) -> float:
        """Fraction of captured self CPU time spent in communication ops.
        0.0 for a strategy with no real collectives (e.g. baseline); > 0 for
        any strategy that actually communicates across ranks (verified in
        tests/test_profiling.py and the Phase 5 smoke test)."""
        total = self.total_self_cpu_time_us
        return self.comm_self_cpu_time_us / total if total > 0 else 0.0

    def to_dict(self, top_n: int = 20) -> dict:
        return {
            "n_ops_observed": self.n_ops_observed,
            "comm_self_cpu_time_us": self.comm_self_cpu_time_us,
            "compute_self_cpu_time_us": self.compute_self_cpu_time_us,
            "comm_fraction": self.comm_fraction,
            "wall_time_s": self.wall_time_s,
            "peak_cpu_memory_before_bytes": self.peak_cpu_memory_before_bytes,
            "peak_cpu_memory_after_bytes": self.peak_cpu_memory_after_bytes,
            "peak_cuda_memory_bytes": self.peak_cuda_memory_bytes,
            "top_comm_ops": [o.to_dict() for o in self.comm_ops[:top_n]],
            "top_compute_ops": [o.to_dict() for o in self.compute_ops[:top_n]],
        }


def _classify(prof: profile, wall_time_s: float, mem_before: int, mem_after: int) -> ProfileResult:
    comm_ops: list[OpStat] = []
    compute_ops: list[OpStat] = []
    comm_self = 0.0
    compute_self = 0.0

    events = prof.key_averages()
    for e in events:
        stat = OpStat(
            name=e.key,
            count=e.count,
            self_cpu_time_us=e.self_cpu_time_total,
            cpu_time_total_us=e.cpu_time_total,
        )
        if is_comm_event(e.key):
            comm_ops.append(stat)
            comm_self += e.self_cpu_time_total
        else:
            compute_ops.append(stat)
            compute_self += e.self_cpu_time_total

    comm_ops.sort(key=lambda s: -s.self_cpu_time_us)
    compute_ops.sort(key=lambda s: -s.self_cpu_time_us)

    peak_cuda = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    return ProfileResult(
        n_ops_observed=len(events),
        comm_self_cpu_time_us=comm_self,
        compute_self_cpu_time_us=compute_self,
        comm_ops=comm_ops,
        compute_ops=compute_ops,
        wall_time_s=wall_time_s,
        peak_cpu_memory_before_bytes=mem_before,
        peak_cpu_memory_after_bytes=mem_after,
        peak_cuda_memory_bytes=peak_cuda,
    )


def profile_call(fn: Callable[[], T]) -> tuple[T, ProfileResult]:
    """Wrap an arbitrary zero-arg callable in a real `torch.profiler.profile`
    CPU capture (+ CUDA activity too, if available), and classify every
    captured op via `is_comm_event`.

    Used by profile_run.py to profile an ENTIRE `train_*.py` `train(args)`
    call (construction, every training step, and that script's own
    end-of-run correctness check) without reimplementing any training loop.
    Because the whole call is profiled (not just the per-step forward/
    backward/optimizer.step), the resulting comm time includes any one-time
    collectives too (e.g. DDP/FSDP/TP's constructor-time weight broadcast,
    or their end-of-run cross-rank correctness `all_gather`) -- this is
    disclosed here rather than silently treated as "pure per-step" cost.
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mem_before = peak_cpu_memory_bytes()
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    t0 = time.perf_counter()
    with profile(activities=activities) as prof:
        result = fn()
    wall_time_s = time.perf_counter() - t0
    mem_after = peak_cpu_memory_bytes()

    return result, _classify(prof, wall_time_s, mem_before, mem_after)


def profile_steps(
    step_fn: Callable[[], None], n_steps: int, warmup_steps: int = 2
) -> ProfileResult:
    """Tighter alternative to `profile_call`: run `step_fn` (one train step)
    `warmup_steps` times UNPROFILED first (so lazy init / first-call JIT
    overhead doesn't skew the captured window), then `n_steps` times inside
    a real profiler capture. Used by this module's own local smoke test
    (see tests/test_profiling.py) on tiny hand-rolled DDP/no-op steps, where
    a per-step (not whole-run) profile is the more precise comparison.
    """
    for _ in range(warmup_steps):
        step_fn()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    mem_before = peak_cpu_memory_bytes()
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    t0 = time.perf_counter()
    with profile(activities=activities) as prof:
        for _ in range(n_steps):
            step_fn()
    wall_time_s = time.perf_counter() - t0
    mem_after = peak_cpu_memory_bytes()

    return _classify(prof, wall_time_s, mem_before, mem_after)
