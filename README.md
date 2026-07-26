# distributed-training-lab

> Distributed training; profile scaling on Kaggle's 2xT4.

**Stack:** Python / PyTorch; gloo (CPU) for correctness, 2xT4 for scaling

> Scale is deliberately small and hardware-constrained (built on an M2 / 8GB, no local GPU).
> The point is the mechanics and honest measurement, not raw scale.

## Problem
This is a portfolio/learning exercise, not a production system or a novel research
contribution: the real question it explores is what it actually, mechanically takes to make
a model train across more than one device, and where the added complexity and overhead
really come from as you move from replicating the whole model per-device (DDP), to sharding
the model's parameters/optimizer state across devices while still replicating data (FSDP),
to sharding individual layers' weights (tensor parallelism), to sharding the model's layers
themselves across devices (pipeline parallelism). It's deliberately built on hardware that is
NOT a datacenter -- an 8GB M2 laptop with no local GPU for correctness work, and Kaggle's free
2xT4 (not a cluster) for anything at real GPU scale -- because the hardware constraint is the
point: proving each mechanism actually works with the free/small hardware available, rather
than assuming it away with a bigger machine. What a reader should get out of this repo: a
working, tested proof of each parallelism strategy's actual mechanism (real PyTorch
distributed/FSDP/DTensor/pipelining APIs, not hand-rolled stand-ins), honestly measured on
whatever hardware was actually used to run it -- CPU/gloo correctness today, 2xT4/nccl
throughput and memory numbers only once those Kaggle runs have actually happened (see Status
below for exactly what that split is right now).

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

Phase 4 (tensor + pipeline parallelism) is implemented:

- Phase 4a, tensor parallelism (TP) -- `train_tp.py` + `src/tensor_parallel.py`
  use the REAL `torch.distributed.tensor` (DTensor) `ColwiseParallel`/
  `RowwiseParallel` + `DeviceMesh` API, not a hand-rolled stand-in for the
  collectives. Step 1 of this phase found that API works and is numerically
  correct on CPU/gloo for plain `nn.Linear`-based modules, but does NOT
  extend to TinyGPT's actual attention module: `nn.MultiheadAttention`
  packs Q/K/V into one fused `in_proj_weight` `nn.Parameter`, not separate
  `nn.Linear` submodules, and `parallelize_module`'s plan mechanism only
  targets named `Linear`/`Embedding` submodules -- pointing a plan at the
  string `"in_proj_weight"` silently no-ops (PyTorch warns, doesn't error).
  This is a genuine limitation of applying the API to
  `nn.MultiheadAttention` specifically (would reproduce on GPU identically),
  and is exactly why production TP frameworks (Megatron-LM, torchtitan, HF)
  define their own attention modules instead of using
  `nn.MultiheadAttention`. So this phase does the same: `TPTinyGPT`
  (`src/tensor_parallel.py`) is a numerically-equivalent rewrite with
  explicit `wq`/`wk`/`wv`/`wo` `nn.Linear`s, Colwise-sharded on Q/K/V/`fc1`
  (head-/hidden-dim-parallel, no communication) and Rowwise-sharded on
  `wo`/`fc2` (implicit `all_reduce` on the way out) -- the standard Megatron
  attention+MLP TP plan, using only the library's own default layouts. The
  vocab/position embeddings, both LayerNorms, and the LM head stay fully
  replicated on every rank (not sharded) -- see Limitations.
- Correctness, verified locally without any GPU (`tests/test_train_tp.py`,
  real 2- and 3-process `gloo` runs via `torch.multiprocessing.spawn`):
  (a) each rank's local sharded-parameter count matches an independently
  computed exact expectation (not just "less than the full model"); (b)
  every rank's per-step loss is bit-identical across ranks -- the correct TP
  invariant, and the *opposite* of DDP/FSDP's "losses differ, weights
  match" invariant, since TP replicates data and shards the model rather
  than the reverse; (c) the TP-sharded model's final, fully-materialized
  weights match an independent single-process TinyGPT reference trained
  end to end on identical initial weights/data/hyperparameters, to within
  ~3e-7 (float32 noise). Also verified with 3 ranks. Remote (real 2-GPU)
  run: `notebooks/kaggle_tp_2gpu.py`.
- Phase 4b, pipeline parallelism (PP) -- `train_pp.py` +
  `src/pipeline_model.py` use the REAL `torch.distributed.pipelining`
  (`PipelineStage` + `ScheduleGPipe`), with real `dist.send`/`dist.recv` of
  activations/gradients between OS processes. Unlike TP, this needed NO
  architecture rewrite: PP splits the model at LAYER boundaries (a sequence
  of whole `nn.Module`s), so TinyGPT's actual, unmodified transformer
  blocks are sliced by layer index (`stage_layer_ranges`, same "last stage
  absorbs the remainder" convention Phase 2/3 use for data shards) into
  per-stage modules that share the exact same parameter objects as the
  source model, not copies. A world_size=1 degenerate bug was found and
  fixed during this build: with 1 stage, rank 0 is simultaneously first and
  last, and the original first/last/middle branching didn't handle that
  overlap (silently producing d_model-sized output instead of vocab-sized
  logits); a dedicated `PPSingleStage` module now handles it.
- Correctness, verified locally without any GPU (`tests/test_train_pp.py`,
  real 2- and 3-process `gloo` runs): (a) each stage's layer range is
  disjoint, contiguous, and covers every layer exactly once across ranks;
  (b) each rank's local parameter count matches an independently computed
  expectation, and (since PP shards layers with no replication, unlike
  DDP/FSDP/TP) the sum across all ranks equals the full model's parameter
  count exactly; (c) final trained per-stage weights match an independent
  single-process TinyGPT reference to within ~5.7e-5 (2 ranks, 15 steps) /
  ~6.3e-5 (3 ranks, 10 steps) -- real, measured noise from compounded AdamW
  steps on top of the ~1e-8 per-step gradient noise already found at the
  raw pipelining-API level (`.probe/pp_probe3.py`/`pp_probe4.py`); (d) the
  world_size=1 degenerate case also matches the same reference. Remote
  (real 2-GPU) run: `notebooks/kaggle_pp_2gpu.py`.

Phase 5 (profiling: comms overhead, memory) is implemented:

- `src/profiling.py` -- wraps `torch.profiler.profile(activities=[CPU (+
  CUDA if available)])` around a block of training and classifies every
  captured op as communication or compute. The classification rule was
  built from ACTUAL observed `torch.profiler` event names on this machine
  (torch 2.13.0, macOS, gloo), not guessed -- see `.probe/profiling_probe*.py`
  for the raw investigation. Real names found: DDP's gradient allreduce
  shows up as `c10d::allreduce_` (dispatcher, real self time) and
  `gloo:all_reduce` (backend op, runs on a separate worker thread -- its
  own self time is 0, but its `cpu_time_total` is where the real collective
  cost is); FSDP's all-gather/reduce-scatter show as `c10d::allgather_` /
  `c10d::_reduce_scatter_base_` / `gloo:all_gather`; PP's real point-to-point
  transfer shows as `c10d::send` / `c10d::recv_` / `gloo:send` / `gloo:recv`;
  TP's DTensor-based collectives show as `_c10d_functional::all_reduce` and
  (importantly -- this is where MOST of TP's real comm time showed up in the
  probe, ~85% of TP-related self time) `_c10d_functional::wait_tensor`. The
  classifier matches on the `c10d`/`gloo:` namespace (covers all of the
  above, including the wait_tensor case) with a keyword fallback
  (allreduce/all_gather/reduce_scatter/broadcast/send/recv) for names not
  yet observed. Also captures peak CPU memory (`resource.getrusage(
  RUSAGE_SELF).ru_maxrss`, verified on this machine to already report BYTES
  on Darwin, unlike Linux's KB) and peak CUDA memory
  (`torch.cuda.max_memory_allocated()`, guarded by `torch.cuda.is_available()`
  so the exact same code path is what will run on Kaggle).
- `profile_run.py` -- a CLI that reuses each strategy's OWN, UNMODIFIED
  `train(args)` from `train_baseline.py`/`train_ddp.py`/`train_fsdp.py`/
  `train_tp.py`/`train_pp.py` (no training loop is reimplemented), wraps the
  whole call in `src.profiling.profile_call`, and writes a per-rank JSON
  summary (comm/compute self-time breakdown, comm_fraction, wall time, peak
  CPU/CUDA memory) to `--out-dir`. For the 4 distributed strategies this
  needs a real `dist.init_process_group`, exactly like the underlying
  `train_*.py` scripts (same torchrun/env-var convention).
- Local CPU/gloo smoke test (the real, checkable claim at this scale --
  see Results): running all 5 strategies through `profile_run.py` shows
  `comm_fraction == 0.0` for baseline (zero `torch.distributed` calls, zero
  comm ops observed) and `comm_fraction > 0.0` for every one of DDP/FSDP/
  TP/PP, each showing the strategy-specific op names above (not the same
  generic "some comm happened" -- DDP's top comm op is allreduce/allgather/
  broadcast, FSDP's is reduce_scatter/allgather/broadcast, PP's is send/recv,
  TP's is wait_tensor/all_reduce/all_gather_into_tensor). `tests/test_profiling.py`
  (33 tests): unit tests of the classifier against literal, previously-
  observed event-name strings; a real 2-process gloo `dist.all_reduce`
  integration test; and end-to-end `profile_run.py` integration tests
  (baseline vs DDP) via `torch.multiprocessing.spawn` (same launcher
  workaround as Phase 2-4, see Limitations).
- Remote (real 2-GPU) run: `notebooks/kaggle_profiling_2gpu.py` -- runs all
  5 strategies (baseline pinned to 1 GPU, ddp/fsdp/tp/pp via
  `torchrun --nproc_per_node=2`/nccl) through `profile_run.py`, refuses to
  run on anything but exactly 2 GPUs, and assembles per-strategy comms-
  overhead %, peak CUDA memory, and a throughput-based speedup-vs-baseline
  number (with an explicit documented caveat: DDP/FSDP's global batch is
  2x baseline's per step, TP/PP's is not, so this is NOT yet an
  iso-batch efficiency comparison across strategies).

CPU-only, HONEST scope limit: this phase can only prove the profiler
correctly DISTINGUISHES communication-heavy strategies from a
no-communication baseline (a real, checkable claim, verified above) -- it
cannot produce a meaningful scaling-efficiency, comms-overhead-%, or
memory-savings NUMBER, since that requires actual multi-GPU throughput/
memory. No such number exists anywhere in this repo yet; `notebooks/
kaggle_profiling_2gpu.py` is ready but has not been executed.

## Status / what's real right now
The bird's-eye version, before the detailed Approach/Results above and below: every phase's
*mechanism* is real code, exercised by real multi-process tests on CPU/gloo (no GPU used or
needed for correctness). Every phase's *GPU number* is TODO -- this repo has never run on a
GPU of any kind, let alone 2. Per PLAN.md's Phase 6 title, "2 GPUs, not a cluster": nothing
in this repo claims or measures anything beyond Kaggle's free 2xT4, and even that hasn't
happened yet.

| Phase | Mechanism | Local correctness verified | Real GPU numbers | Kaggle script |
| ----- | --------- | --------------------------- | ----------------- | -------------- |
| 1. Single-GPU baseline | `train_baseline.py` | Yes (CPU/MPS smoke test) | TODO | `notebooks/kaggle_single_gpu_baseline.py` |
| 2. DDP (data parallel) | `train_ddp.py` | Yes (2/3-rank gloo) | TODO | `notebooks/kaggle_ddp_2gpu.py` |
| 3. FSDP (ZeRO-style sharding) | `train_fsdp.py` | Yes (2/3-rank gloo) | TODO | `notebooks/kaggle_fsdp_2gpu.py` |
| 4a. Tensor parallelism | `train_tp.py` | Yes (2/3-rank gloo) | TODO | `notebooks/kaggle_tp_2gpu.py` |
| 4b. Pipeline parallelism | `train_pp.py` | Yes (2/3-rank gloo) | TODO | `notebooks/kaggle_pp_2gpu.py` |
| 5. Profiling (comm/compute classifier) | `profile_run.py` | Yes (classifier distinguishes strategies on gloo) | TODO | `notebooks/kaggle_profiling_2gpu.py` |

What running each of those 6 scripts on Kaggle's 2xT4 would actually show (not what it "should"
or is "expected" to show, since that isn't known): real single- and dual-T4 throughput and loss
curves; whether DDP's allreduce, FSDP's extra all-gather/reduce-scatter traffic, TP's per-block
all_reduce, and PP's cross-stage send/recv behave the way their CPU/gloo mechanics predict once
real nccl and real GPU memory/interconnect are involved; and whatever the profiler's comm/compute
classifier reports for comms-overhead percentage and peak memory per strategy on real hardware.
None of that is known yet -- see Results and Limitations below for the exact TODOs.

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
| TP CPU/gloo correctness (2 ranks, 20-25 steps, real `ColwiseParallel`/`RowwiseParallel` + `DeviceMesh`) | Verified: each rank's local sharded-parameter count matched an independently computed exact expectation; every rank's per-step loss was bit-identical (correct TP invariant -- data replicated, model sharded); final trained weights matched an independent single-process reference to within ~3e-7 (float32 noise). Also verified with 3 ranks. See `tests/test_train_tp.py`. |
| TP 2xT4 GPU (Kaggle) throughput/scaling | TODO -- not yet measured. Script is ready: `notebooks/kaggle_tp_2gpu.py`. |
| PP CPU/gloo correctness (2 and 3 ranks, 10-15 steps, real `PipelineStage` + `ScheduleGPipe`) | Verified: layer ranges disjoint/contiguous across ranks; each rank's local parameter count matched an independently computed expectation (summing exactly to the full model's count, no replication); final trained per-stage weights matched an independent single-process reference to within 5.655e-05 (2 ranks, 15 steps) / 6.295e-05 (3 ranks, 10 steps); world_size=1 degenerate case also verified. See `tests/test_train_pp.py`. |
| PP 2xT4 GPU (Kaggle) throughput/pipeline-bubble overlap | TODO -- not yet measured; no wall-clock overlap benefit has been measured even locally (2 CPU processes, blocking gloo P2P) -- that's a Kaggle-only question. Script is ready: `notebooks/kaggle_pp_2gpu.py`. |
| Profiling CPU/gloo classifier check (2 ranks, 15 steps, 28,288-param model, one real run via `.probe/profile_run_smoke.py`) | Verified the classifier genuinely distinguishes strategies: baseline `comm_fraction=0.0000` (0 comm ops observed) vs DDP `comm_fraction=0.0069-0.0074` (top ops: `c10d::broadcast_`/`allreduce_`/`allgather_`), FSDP `comm_fraction=0.0114-0.0116` (top ops: `broadcast_`/`allgather_`/`_reduce_scatter_base_`), TP `comm_fraction=0.1843-0.1929` (top op: `_c10d_functional::wait_tensor`), PP `comm_fraction=0.0058-0.0068` (top ops: `send`/`recv_`). These are CPU/gloo self-time fractions on a tiny model over 15 steps -- real, reproducible per-run, but noisy and NOT a GPU comms-overhead number; see `tests/test_profiling.py` for the pytest-checked version (33 tests). |
| Profiling 2xT4 GPU (Kaggle) comms-overhead %/peak-memory/scaling-efficiency | TODO -- not yet measured; requires real multi-GPU throughput/memory, meaningless on CPU. Script is ready: `notebooks/kaggle_profiling_2gpu.py`. |

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
- TP (Phase 4a) does NOT shard the vocab/position embeddings, either
  LayerNorm, or the LM head -- they stay fully replicated on every rank.
  Megatron also supports vocab-parallel embeddings; this repo doesn't
  implement that, so TP's real memory savings here are limited to the
  attention/MLP `Linear` weights, not the whole model. A real, documented
  simplification, not something hidden.
- TP (Phase 4a) required a full model rewrite (`TPTinyGPT`, explicit
  wq/wk/wv/wo Linears) because `nn.MultiheadAttention`'s fused
  `in_proj_weight` can't be targeted by `parallelize_module`'s plan
  mechanism (it only matches named `Linear`/`Embedding` submodules). This
  is a genuine limitation of the real API applied to that specific module,
  not a CPU/gloo-only quirk -- it would reproduce identically on GPU. See
  `src/tensor_parallel.py`'s module docstring for the empirical
  investigation.
- PP (Phase 4b) has NOT measured any pipeline-bubble/overlap benefit --
  `ScheduleGPipe`'s schedule allows overlapping microbatch forward/backward
  across stages, but on this machine (2 local CPU processes, blocking gloo
  P2P) no wall-clock overlap was measured or claimed, only that the
  multi-microbatch data flow and cross-process communication are real and
  correct. Actual pipeline-bubble reduction from overlap is a throughput
  question that needs the Kaggle 2xT4 run; `notebooks/kaggle_pp_2gpu.py`
  is written and ready but has not been executed.
- TP (Phase 4a) and PP (Phase 4b) correctness have only been verified on
  CPU/gloo with tiny (tens-of-thousands-of-parameters) models over up to 25
  steps and up to 3 ranks -- real GPU scaling numbers (throughput,
  nccl comms overhead for TP's per-block all_reduce and PP's per-microbatch
  send/recv) do not exist yet; `notebooks/kaggle_tp_2gpu.py` and
  `notebooks/kaggle_pp_2gpu.py` are written and ready but have not been
  executed. Do not read any 2xT4 TP/PP number into this repo until they
  have.
- Both Phase 4 local tests use `torch.multiprocessing.spawn`, not
  `torchrun`, for the same reason as Phase 2/3: this dev sandbox's
  `torchrun` elastic-launcher rendezvous resolves `socket.gethostname()`
  (ignoring `MASTER_ADDR`) and hangs indefinitely on a broken local DNS
  lookup, so `mp.spawn` was used instead for local verification. This is a
  launcher-level sandbox quirk, not a bug in `train_tp.py`/`train_pp.py`'s
  distributed logic.
- Phase 5 (profiling) can only prove the comm-vs-compute CLASSIFIER is
  correct on CPU/gloo, not report any real scaling-efficiency, comms-
  overhead-%, or memory-savings number -- those require actual multi-GPU
  throughput/memory (>1 real GPU), which this machine doesn't have.
  `notebooks/kaggle_profiling_2gpu.py` is written and ready but has not
  been executed; do not read any 2xT4 profiling number into this repo
  until it has.
- Phase 5's classifier uses `self_cpu_time_total` per op name (standard
  torch.profiler convention -- avoids double-counting nested parent/child
  calls on the same thread), summed per comm/compute category. One
  documented quirk found while investigating this (see
  `src/profiling.py`'s module docstring and `.probe/profiling_probe2.py`):
  gloo's own backend collective op (e.g. `gloo:all_reduce`) runs on a
  separate worker thread and reports `self_cpu_time_total == 0` for
  itself -- the real collective cost instead shows up as the *calling*
  thread's `c10d::allreduce_` self time (the blocking wait). The
  namespace-based classification rule (match on `c10d`/`gloo:`) correctly
  buckets both under "comm" regardless of which one carries the nonzero
  self time, but a naive keyword-only classifier that only checked, say,
  `gloo:all_reduce`'s OWN self time would have badly undercounted DDP's
  real comm cost.
- Because `profile_run.py` profiles an ENTIRE `train(args)` call (not just
  the per-step forward/backward/optimizer.step loop), the reported
  `comm_fraction` includes one-time collectives too -- e.g. DDP/FSDP/TP's
  constructor-time weight broadcast and their end-of-run cross-rank
  correctness `all_gather` -- not a "pure per-step steady-state" number.
  This is disclosed rather than silently averaged away; at `--steps 15`
  used for the local smoke test, these one-time costs are a
  non-negligible fraction of the total, which is part of why (along with
  small-model CPU noise) the reported `comm_fraction` numbers above should
  be read as "the classifier works and ranks strategies in a sane order",
  not as a precise per-step comms-overhead percentage.
- Peak CPU memory (`resource.getrusage(RUSAGE_SELF).ru_maxrss`) is a
  process-lifetime historical high-water mark, not resettable -- so
  `profile_run.py`'s "before"/"after" pair still reports two valid
  absolute peaks, not an isolated per-call delta the way the CUDA peak
  (which IS reset before each profiled call) does.
