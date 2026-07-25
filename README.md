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

## Results
| Metric | Value |
| ------ | ----- |
| CPU smoke test (28K params, 300 steps, M2/8GB) | Runs correctly; loss drops 3.46 -> 0.82 over 300 steps (sanity check only, not a benchmark). |
| MPS smoke test | Runs correctly (few steps). |
| Single T4 GPU (Kaggle) throughput/loss | TODO -- not yet measured. Script is ready: `notebooks/kaggle_single_gpu_baseline.py`. |
| DDP / FSDP / TP / scaling efficiency | TODO -- later phases. |

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
