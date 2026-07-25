# distributed-training-lab

> Distributed training; profile scaling on Kaggle's 2xT4.

**Stack:** Python / PyTorch; gloo (CPU) for correctness, 2xT4 for scaling

> Scale is deliberately small and hardware-constrained (built on an M2 / 8GB, no local GPU).
> The point is the mechanics and honest measurement, not raw scale.

## Problem
<!-- TODO: the real problem this solves and why it matters. -->

## Approach
See PLAN.md for the phased build. Phase 1 (single-device baseline) is implemented:

- Model: a small decoder-only transformer, `TinyGPT` (`src/model.py`) --
  a couple of `nn.TransformerEncoderLayer`s with a causal mask, char-level.
  Default local smoke config is ~28K params; the Kaggle config below is larger
  but still modest (a few hundred K params).
- Data: a deterministic, offline synthetic corpus (`src/data.py`) -- a repeated
  pangram, char-tokenized. No downloads, so it runs identically on a laptop,
  in CI, and on Kaggle. It exists to give the model a genuinely learnable
  next-char task (loss measurably drops), not to be a meaningful language model.
- Training loop: `train_baseline.py` -- device-agnostic (picks CUDA > MPS > CPU,
  or `--device` override), logs per-step loss/time/throughput, dumps a CSV/JSON
  log for later phases (DDP/FSDP/profiling) to diff against.
- Remote (real GPU) run: `notebooks/kaggle_single_gpu_baseline.py` -- pins to
  exactly one of the two T4s on Kaggle (refuses to run on 0 or 2+ GPUs), reuses
  the exact same model/training code, and writes a JSON log for the real
  single-GPU number.

Phase 2 (DDP / data parallel) is implemented:

- `train_ddp.py` -- wraps the *same* `TinyGPT` model, unmodified, in
  `torch.nn.parallel.DistributedDataParallel`. Backend auto-picks `gloo` on
  CPU / `nccl` on CUDA (mirrors the baseline's device-selection philosophy,
  applied to the distributed backend). Launches with `torchrun` (reads the
  standard `RANK`/`WORLD_SIZE`/`LOCAL_RANK` env vars); also runs as a
  degenerate single-process `world_size=1` job with plain `python` for a
  fast sanity check.
- Each rank trains on a disjoint, contiguous shard of the corpus (a genuine
  partition, not e.g. all ranks re-sampling the same data) -- this is the
  actual data-parallel mechanic being tested, not just "does DDP wrap
  without crashing."
- Correctness, verified locally without any GPU: `gloo` is a real
  `torch.distributed` backend that runs over real OS processes on CPU, so
  DDP's constructor-time weight broadcast and per-step gradient allreduce
  are genuinely exercised -- only *speed*, not *correctness*, requires real
  GPUs. `tests/test_train_ddp.py` launches actual 2- and 3-process runs
  (via `torch.multiprocessing.spawn`) and checks: (a) shards are disjoint,
  (b) per-rank loss histories differ (proof ranks really see different
  data), and (c) final model weights are bit-for-bit identical across every
  rank (proof gradient sync, not independent per-rank training) -- see
  Results below for the real numbers this produced.
- Remote (real 2-GPU) run: `notebooks/kaggle_ddp_2gpu.py` -- launches the
  same `train_ddp.py` with `torchrun --nproc_per_node=2` and `nccl` across
  both real T4s, refuses to run on anything but exactly 2 GPUs, and writes a
  JSON summary (including the combined 2-GPU throughput) for a real
  speedup-vs-single-GPU number once run.

Phase 3 (FSDP / ZeRO-style sharding) is implemented:

- `train_fsdp.py` -- wraps the *same* `TinyGPT` model in the REAL
  `torch.distributed.fsdp.FullyShardedDataParallel` (FSDP), not a mock or a
  hand-rolled stand-in. Step 1 of this phase was to actually try real FSDP on
  CPU/gloo rather than assume either way, since PyTorch's FSDP has
  historically carried CUDA-oriented assumptions. What was actually found by
  running it on this machine (macOS/Apple Silicon, torch 2.13.0, no CUDA):
  - `FSDP(model)` with the default `device_id=None` **crashes** here:
    constructing FSDP on a CPU module makes it call
    `torch._C._get_accelerator()`, which detects Apple Silicon's MPS backend
    as "the accelerator" (even though gloo/CPU, not MPS, is what's actually
    being used), then calls `torch.mps.current_device()` -- which this
    torch build's internal FSDP device-handle shim does not implement,
    raising `AttributeError: module 'torch.mps' has no attribute
    'current_device'`. Passing `device_id=torch.device("cpu")` explicitly
    sidesteps that auto-detection path and works.
  - `sync_module_states=True` (FSDP's own constructor-time weight broadcast,
    the FSDP analogue of DDP's automatic broadcast) unconditionally requires
    GPU tensors in this torch version and raises `ValueError` on CPU even
    with `device_id="cpu"` set. The CPU/gloo path in `train_fsdp.py` instead
    does the same broadcast manually with plain `dist.broadcast` calls
    before wrapping in FSDP -- a real collective, not a workaround that
    weakens the test. On CUDA (Kaggle), `sync_module_states=True` is used as
    normal, since it isn't broken there.
  - With both of those addressed, real FSDP's actual sharding mechanics work
    correctly on CPU/gloo: parameters, gradients, and optimizer state are
    genuinely partitioned across ranks (`ShardingStrategy.FULL_SHARD`,
    ZeRO-3-style), and per-step all-gather (unshard)/reduce-scatter
    (gradients)/re-shard all really execute over real gloo collectives.
- Correctness, verified locally without any GPU
  (`tests/test_train_fsdp.py`, real 2- and 3-process `gloo` runs via
  `torch.multiprocessing.spawn`): (a) each rank's local flat-parameter count
  and AdamW optimizer-state tensor count are a genuine fraction of the full
  model's -- exactly half for 2 ranks, less than a third for 3 ranks -- not
  just "training runs without crashing"; (b) ranks train on disjoint,
  contiguous data shards with different per-step loss (real different data,
  same mechanic as Phase 2); (c) after training, the fully-materialized
  (unsharded) model -- reconstructed on every rank via
  `FSDP.summon_full_params` -- is bit-for-bit identical across all ranks,
  proving reduce-scatter + all-gather actually kept the shards in sync
  despite each rank training on different local data throughout (the FSDP
  analogue of Phase 2's weight-sync proof). See Results below for the real
  numbers this produced.
- Remote (real 2-GPU) run: `notebooks/kaggle_fsdp_2gpu.py` -- launches the
  same `train_fsdp.py` with `torchrun --nproc_per_node=2` and real
  `nccl`/CUDA `device_id`s + `sync_module_states=True` (the code path the CPU
  run above could *not* exercise) across both T4s, refuses to run on
  anything but exactly 2 GPUs, and writes a JSON summary for the real
  memory/throughput picture vs the Phase 1 baseline and Phase 2 DDP.

## Results
| Metric | Value |
| ------ | ----- |
| CPU smoke test (28K params, 300 steps, M2/8GB) | Runs correctly; loss drops 3.4565 -> 0.9396 over 300 steps (sanity check only, not a benchmark). Log: `logs/phase1_cpu_smoke_300steps.csv`. |
| MPS smoke test | Runs correctly (few steps). |
| Single T4 GPU (Kaggle) throughput/loss | TODO -- not yet measured. Script is ready: `notebooks/kaggle_single_gpu_baseline.py`. |
| DDP CPU/gloo correctness (2 ranks, 25 steps, 4.5K-param model) | Verified: ranks trained on disjoint shards (`[0:2520)` / `[2520:5040)`), per-rank loss differed throughout (e.g. final loss rank0=3.2988 vs rank1=3.2540), yet final weights were bit-identical across all ranks (`torch.equal` on every parameter tensor). Also verified with 3 ranks and over 60 steps. See `tests/test_train_ddp.py`. |
| DDP 2xT4 GPU (Kaggle) throughput/speedup | TODO -- not yet measured. Script is ready: `notebooks/kaggle_ddp_2gpu.py`. |
| FSDP CPU/gloo sharding correctness (2 ranks, 25 steps, 27,776-param model, `FULL_SHARD`) | Verified: each rank's local shard held exactly 13,888/27,776 params (50.0%) and 27,777 optimizer-state elements (vs. 55,552 = `2 x full_params` if unsharded) -- real ZeRO-3-style partitioning, not just "it ran." Ranks trained on disjoint shards (`[0:2520)` / `[2520:5040)`) with per-rank param sums starting different (rank0=220.2962, rank1=185.5377 pre-broadcast) and identical after the manual broadcast (both 220.2962); per-step loss differed throughout (e.g. final loss rank0=3.1668 vs rank1=3.1626); after training, the fully-materialized (unsharded) model was bit-identical across ranks (`torch.equal` on every parameter tensor). Also verified with 3 ranks. See `tests/test_train_fsdp.py`. |
| FSDP 2xT4 GPU (Kaggle) memory/throughput vs DDP | TODO -- not yet measured; `train_fsdp.py` also doesn't instrument peak memory yet. Script is ready: `notebooks/kaggle_fsdp_2gpu.py`. |
| TP / scaling efficiency | TODO -- later phases. |

## Limitations / what's unrealistic
- Small scale by design: ~28K-param toy transformer locally, a few hundred K
  on the planned Kaggle config -- not a real language model, "2 GPUs, not a cluster."
- Synthetic, repeated-pangram corpus: intentionally trivial/offline so the
  model has *something* learnable to demonstrate the loop is real, not a
  benchmark of language modeling quality.
- No real GPU numbers exist yet -- `train_baseline.py` has only been run on
  CPU/MPS locally. The Kaggle script exists and is documented but has not
  been executed; do not read any throughput number into this repo until it has.
- Per-step timing includes Python/optimizer overhead, not just kernel time --
  fine for a baseline comparison point, not a substitute for a real profiler
  (that's Phase 5).
- DDP (Phase 2) correctness has only been verified on CPU/gloo with a tiny
  (4.5K-param) model over up to 60 steps and up to 3 ranks -- real GPU
  scaling numbers (throughput, speedup vs the single-GPU baseline, nccl
  comms overhead) do not exist yet; `notebooks/kaggle_ddp_2gpu.py` is
  written and ready but has not been executed. Do not read any 2xT4 number
  into this repo until it has.
- The local DDP test uses `torch.multiprocessing.spawn`, not `torchrun`,
  to launch the 2/3 processes. `torchrun` itself is the documented/
  recommended way to invoke `train_ddp.py` (and is what the Kaggle script
  uses) and should work normally in typical environments; on this
  particular dev sandbox its elastic-launcher rendezvous resolves
  `socket.gethostname()` (ignoring `MASTER_ADDR`) and hung indefinitely on
  a broken local DNS lookup, so `mp.spawn` (which lets us set
  `MASTER_ADDR=127.0.0.1` directly for `init_process_group`) was used
  instead for the actual local verification. This is a launcher-level
  sandbox quirk, not a bug in `train_ddp.py`'s DDP logic.
- MPS (Apple GPU) has no `torch.distributed` backend, so DDP is CPU(gloo)-
  or CUDA(nccl)-only here; there is no "MPS DDP" fallback the way the
  Phase 1 baseline falls back to MPS for single-device training.
- FSDP (Phase 3) needed two CPU-only workarounds to run at all on this
  machine, both due to real, reproduced bugs/limitations in this torch
  version's FSDP on non-CUDA hardware (see README Approach and
  `train_fsdp.py`'s module docstring for the exact tracebacks):
  `device_id=torch.device("cpu")` must be passed explicitly (the default
  `device_id=None` crashes via an MPS-accelerator auto-detection path that
  doesn't apply to a gloo/CPU run at all), and `sync_module_states=True`
  cannot be used on CPU (requires GPU tensors), so the initial weight
  broadcast is done manually via `dist.broadcast`. These are genuine,
  reproducible quirks of running FSDP on CPU/gloo on Apple Silicon in this
  torch build, not something papered over or hidden -- the CUDA/Kaggle code
  path in `train_fsdp.py` does not need either workaround.
- FSDP (Phase 3) correctness has only been verified on CPU/gloo with a tiny
  (27,776-param) model over up to 60 steps and up to 3 ranks, and only for
  `ShardingStrategy.FULL_SHARD` -- real GPU memory/throughput numbers
  (peak memory per rank vs DDP's full replication, nccl comms overhead for
  the extra all-gather/reduce-scatter traffic FSDP adds over DDP's plain
  allreduce) do not exist yet; `train_fsdp.py` doesn't even instrument peak
  memory yet. `notebooks/kaggle_fsdp_2gpu.py` is written and ready but has
  not been executed. Do not read any 2xT4 FSDP number into this repo until
  it has.
- At this repo's toy model scale (tens of thousands of parameters), FSDP's
  extra communication (all-gather before every forward/backward, in
  addition to the reduce-scatter DDP-equivalent) is very unlikely to be
  worth it vs plain DDP -- FSDP exists to shard models too large to
  replicate in full on one GPU, which a 28K-param model obviously never is.
  This phase exists to build and honestly verify the *mechanism*, not to
  claim FSDP is the right tool at this scale.
