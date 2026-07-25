# Build plan -- distributed-training-lab

Each phase is independently verifiable. Check off as completed.

- [ ] Single-GPU baseline
  - Code + CPU/MPS smoke test done: `train_baseline.py`, `src/model.py`, `src/data.py`,
    `tests/test_train_baseline.py`. Ready-to-run remote script:
    `notebooks/kaggle_single_gpu_baseline.py`.
  - TODO: run on Kaggle (1 of 2 T4s) for the real GPU loss/throughput numbers before
    checking this box. See README.md Results.
- [ ] DDP (data parallel)
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
- [ ] FSDP / ZeRO-style sharding
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
- [ ] Tensor + pipeline parallelism
- [ ] Profiling (scaling efficiency, comms overhead, memory)
- [ ] Honest writeup (2 GPUs, not a cluster)
