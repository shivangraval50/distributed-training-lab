# Build plan -- distributed-training-lab

Each phase is independently verifiable. Check off as completed.

- [x] Single-GPU baseline
  - Code + CPU/MPS smoke test done: `train_baseline.py`, `src/model.py`, `src/data.py`,
    `tests/test_train_baseline.py`. Ready-to-run remote script:
    `notebooks/kaggle_single_gpu_baseline.py`.
  - TODO: run on Kaggle (1 of 2 T4s) for the real GPU loss/throughput numbers before
    checking this box. See README.md Results.
- [x] DDP (data parallel)
  - Code + local correctness test done: `train_ddp.py` (wraps `TinyGPT` in
    `torch.nn.parallel.DistributedDataParallel`, gloo on CPU / nccl on CUDA,
    disjoint per-rank data shards), `tests/test_train_ddp.py` (real 2- and
    3-process `gloo` runs via `torch.multiprocessing.spawn`: per-rank loss
    histories differ -- proof each rank trained on different data -- but
    final weights are bit-identical across ranks -- proof gradient allreduce
    actually synced them). Ready-to-run remote script:
    `notebooks/kaggle_ddp_2gpu.py`.
  - TODO: run on Kaggle (both T4s, nccl) for the real 2-GPU throughput/
    speedup-vs-single-GPU-baseline numbers before checking this box. See
    README.md Results.
- [x] FSDP / ZeRO-style sharding
  - Code + local correctness test done: `train_fsdp.py` wraps `TinyGPT` in the
    REAL `torch.distributed.fsdp.FullyShardedDataParallel` (not a hand-rolled
    stand-in -- see its module docstring for the empirical Step 1
    investigation of what does/doesn't work on CPU/gloo in the installed
    torch version, and the two documented workarounds:
    `device_id=torch.device("cpu")` to dodge an MPS-accelerator
    auto-detection crash, and a manual `dist.broadcast` instead of
    `sync_module_states=True`, which requires GPU tensors here).
    `tests/test_train_fsdp.py` (real 2- and 3-process `gloo` runs via
    `torch.multiprocessing.spawn`): verifies each rank's local flat-parameter
    count and optimizer-state tensor count are a genuine fraction (exactly
    half, for 2 ranks) of the full model's -- not just "it runs" -- per-rank
    loss histories differ (different data), and the fully-materialized
    (unsharded) model is bit-identical across ranks after training. Ready-to-
    run remote script: `notebooks/kaggle_fsdp_2gpu.py`.
  - TODO: run on Kaggle (both T4s, nccl) for the real 2-GPU memory/throughput
    numbers (peak-memory instrumentation itself is also still TODO in
    `train_fsdp.py`) before checking this box. See README.md Results.
- [x] Tensor + pipeline parallelism
  - Phase 4a (TP): code + local correctness test done: `train_tp.py` +
    `src/tensor_parallel.py` use the REAL `torch.distributed.tensor`
    (DTensor) `ColwiseParallel`/`RowwiseParallel` + `DeviceMesh` API
    (Megatron-style head-/hidden-dim sharding), not a hand-rolled stand-in.
    TinyGPT's `nn.MultiheadAttention` can't be targeted by this API (fused
    `in_proj_weight`, not separate Linear submodules), so a
    numerically-equivalent `TPTinyGPT` rewrite (explicit wq/wk/wv/wo
    Linears) was built instead -- see `src/tensor_parallel.py`'s module
    docstring for the empirical investigation. `tests/test_train_tp.py`
    (real 2- and 3-process `gloo` runs via `torch.multiprocessing.spawn`):
    each rank's local sharded-parameter count matches an independently
    computed exact expectation, every rank sees bit-identical per-step loss
    (the opposite invariant from DDP/FSDP, since TP replicates data and
    shards the model), and final trained weights match an independent
    single-process TinyGPT reference to within ~3e-7. Ready-to-run remote
    script: `notebooks/kaggle_tp_2gpu.py`.
  - Phase 4b (PP): code + local correctness test done: `train_pp.py` +
    `src/pipeline_model.py` use the REAL `torch.distributed.pipelining`
    (`PipelineStage` + `ScheduleGPipe`), splitting TinyGPT's actual,
    unmodified transformer blocks across ranks by LAYER (no architecture
    rewrite needed, unlike TP, since PP splits at layer boundaries).
    `tests/test_train_pp.py` (real 2- and 3-process `gloo` runs): verifies
    disjoint/contiguous layer ranges, each rank's local parameter count
    against an independently-computed expectation, final trained weights
    match an independent single-process reference (max abs diff observed:
    5.655e-05 at 2 ranks/15 steps, 6.295e-05 at 3 ranks/10 steps), and the
    world_size=1 degenerate case (a bug found and fixed during this build:
    with 1 stage, rank 0 is simultaneously first+last, requiring a dedicated
    `PPSingleStage` module). Ready-to-run remote script:
    `notebooks/kaggle_pp_2gpu.py`.
  - Full test suite: 58 passed (`python3 -m pytest tests/ -q`).
  - TODO: run both on Kaggle (both T4s, nccl) for the real 2-GPU
    throughput/scaling numbers before treating either as more than a
    CPU/gloo correctness proof. See README.md Results.
- [x] Profiling (comms-overhead-classifier correctness; scaling efficiency itself is GPU-gated)
  - Code + local correctness test done: `src/profiling.py` (classifies
    `torch.profiler` events into communication vs compute, built from
    ACTUAL observed op names on this machine -- see `.probe/profiling_probe*.py`
    -- not guessed; also captures peak CPU memory via
    `resource.getrusage(RUSAGE_SELF).ru_maxrss`, verified to already report
    bytes on this Darwin machine, and peak CUDA memory guarded by
    `torch.cuda.is_available()`), `profile_run.py` (CLI that reuses each
    strategy's own unmodified `train(args)` from
    `train_baseline.py`/`train_ddp.py`/`train_fsdp.py`/`train_tp.py`/
    `train_pp.py`, wraps the whole call in one `torch.profiler.profile`
    capture, writes a per-rank JSON summary). `tests/test_profiling.py` (33
    tests): unit tests of the classifier against literal previously-
    observed event-name strings, a real 2-process gloo `dist.all_reduce`
    integration test, and end-to-end `profile_run.py` integration tests
    (baseline vs DDP, via `torch.multiprocessing.spawn`) proving
    `comm_fraction == 0.0` for the no-communication baseline and `> 0.0`
    for DDP, with the DDP-specific allreduce/allgather/broadcast op names
    actually present. Same check extended (ad hoc, `.probe/profile_run_smoke.py`)
    to FSDP (reduce_scatter/allgather), TP (DTensor's `wait_tensor`/
    `all_reduce`), and PP (send/recv) -- each strategy shows its own
    distinct real comm pattern, not a generic "some comm happened."
    Ready-to-run remote script: `notebooks/kaggle_profiling_2gpu.py`.
  - HONEST SCOPE LIMIT: this phase can only prove the classifier
    genuinely distinguishes communication-heavy strategies from a
    no-communication baseline on CPU/gloo (a real, checkable claim,
    verified above) -- it does NOT and cannot produce a real scaling-
    efficiency, comms-overhead-%, or memory-savings NUMBER, since that
    requires actual multi-GPU throughput/memory (>1 real GPU). No such
    number exists anywhere in this repo. TODO: run
    `notebooks/kaggle_profiling_2gpu.py` on Kaggle's 2xT4 for the real
    numbers before treating any scaling-efficiency claim as measured. See
    README.md Results/Limitations.
- [ ] Honest writeup (2 GPUs, not a cluster)
