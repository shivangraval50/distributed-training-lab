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

## Results
| Metric | Value |
| ------ | ----- |
| CPU smoke test (28K params, 300 steps, M2/8GB) | Runs correctly; loss drops 3.4565 -> 0.9396 over 300 steps (sanity check only, not a benchmark). Log: `logs/phase1_cpu_smoke_300steps.csv`. |
| MPS smoke test | Runs correctly (few steps). |
| Single T4 GPU (Kaggle) throughput/loss | TODO -- not yet measured. Script is ready: `notebooks/kaggle_single_gpu_baseline.py`. |
| DDP CPU/gloo correctness (2 ranks, 25 steps, 4.5K-param model) | Verified: ranks trained on disjoint shards (`[0:2520)` / `[2520:5040)`), per-rank loss differed throughout (e.g. final loss rank0=3.2988 vs rank1=3.2540), yet final weights were bit-identical across all ranks (`torch.equal` on every parameter tensor). Also verified with 3 ranks and over 60 steps. See `tests/test_train_ddp.py`. |
| DDP 2xT4 GPU (Kaggle) throughput/speedup | TODO -- not yet measured. Script is ready: `notebooks/kaggle_ddp_2gpu.py`. |
| FSDP / TP / scaling efficiency | TODO -- later phases. |

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
