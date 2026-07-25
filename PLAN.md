# Build plan -- distributed-training-lab

Each phase is independently verifiable. Check off as completed.

- [ ] Single-GPU baseline
  - Code + CPU/MPS smoke test done: `train_baseline.py`, `src/model.py`, `src/data.py`,
    `tests/test_train_baseline.py`. Ready-to-run remote script:
    `notebooks/kaggle_single_gpu_baseline.py`.
  - TODO: run on Kaggle (1 of 2 T4s) for the real GPU loss/throughput numbers before
    checking this box. See README.md Results.
- [ ] DDP (data parallel)
- [ ] FSDP / ZeRO-style sharding
- [ ] Tensor + pipeline parallelism
- [ ] Profiling (scaling efficiency, comms overhead, memory)
- [ ] Honest writeup (2 GPUs, not a cluster)
