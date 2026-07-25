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
- [ ] Tensor + pipeline parallelism
- [ ] Profiling (scaling efficiency, comms overhead, memory)
- [ ] Honest writeup (2 GPUs, not a cluster)
