"""Phase 1: single-device training baseline.

Trains a tiny char-level GPT (src/model.py) on a synthetic corpus
(src/data.py) for a fixed number of steps, logging loss and per-step timing
so later phases (DDP, FSDP, profiling) have a real single-device number to
compare against.

Device-agnostic: picks CUDA > MPS > CPU automatically, or honors --device.
Must run correctly (not just "not crash") on CPU, since that's the only
device available on the dev machine (macOS, no CUDA).

Usage (local smoke test, CPU, seconds):
    python train_baseline.py --steps 20 --batch-size 8 --block-size 32 \
        --d-model 32 --n-layer 2 --n-head 2 --log-every 5

Usage (real run, e.g. on Kaggle 2xT4 -- see notebooks/kaggle_single_gpu_baseline.py
for the ready-to-run remote script that pins to a single GPU):
    python train_baseline.py --steps 500 --batch-size 64 --block-size 128 \
        --d-model 256 --n-layer 4 --n-head 4

No performance numbers are hard-coded anywhere in this repo: throughput and
loss curves only exist once this script has actually been run on the target
device. As of writing, only CPU/MPS local smoke-test numbers exist; a real
GPU run on Kaggle is still TODO (see README.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from src.data import build_corpus, build_vocab, encode, get_batch
from src.model import TinyGPT, count_flops_per_token


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class StepRecord:
    step: int
    loss: float
    step_time_s: float
    tokens_per_sec: float


@dataclass
class RunResult:
    device: str
    num_params: int
    history: list[StepRecord] = field(default_factory=list)
    total_time_s: float = 0.0


def train(args: argparse.Namespace) -> RunResult:
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    corpus = build_corpus(repeats=max(50, args.batch_size * args.block_size // 10 + 50))
    stoi, _ = build_vocab(corpus)
    data = encode(corpus, stoi).to(device)

    model = TinyGPT(
        vocab_size=len(stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"[baseline] device={device} params={model.num_params():,} "
          f"vocab_size={len(stoi)} approx_flops/token={count_flops_per_token(model):,}")

    gen = torch.Generator().manual_seed(args.seed)
    result = RunResult(device=str(device), num_params=model.num_params())

    model.train()
    run_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, args.block_size, generator=gen)

        # Sync before/after timing on CUDA/MPS so step_time reflects actual
        # device compute, not just async kernel-launch overhead on the host.
        _sync(device)
        t0 = time.perf_counter()

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        _sync(device)
        step_time = time.perf_counter() - t0

        tokens = args.batch_size * args.block_size
        tps = tokens / step_time if step_time > 0 else float("inf")
        record = StepRecord(step=step, loss=loss.item(), step_time_s=step_time, tokens_per_sec=tps)
        result.history.append(record)

        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"[baseline] step {step:4d}/{args.steps} "
                  f"loss={record.loss:.4f} step_time={step_time * 1000:.1f}ms "
                  f"tok/s={tps:,.0f}")

    result.total_time_s = time.perf_counter() - run_start

    if args.out:
        _write_log(args.out, args, result)

    print(f"[baseline] done in {result.total_time_s:.2f}s "
          f"final_loss={result.history[-1].loss:.4f}")
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _write_log(out_path: str, args: argparse.Namespace, result: RunResult) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".json":
        payload = {
            "config": vars(args),
            "device": result.device,
            "num_params": result.num_params,
            "total_time_s": result.total_time_s,
            "history": [r.__dict__ for r in result.history],
        }
        path.write_text(json.dumps(payload, indent=2))
    else:
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "step_time_s", "tokens_per_sec"])
            for r in result.history:
                writer.writerow([r.step, r.loss, r.step_time_s, r.tokens_per_sec])
    print(f"[baseline] wrote log to {path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20, help="number of training steps")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--block-size", type=int, default=32, help="context length (tokens)")
    p.add_argument("--d-model", type=int, default=32)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument("--out", type=str, default=None, help="optional path to write per-step log (.csv or .json)")
    return p


def main() -> RunResult:
    args = build_arg_parser().parse_args()
    return train(args)


if __name__ == "__main__":
    main()
