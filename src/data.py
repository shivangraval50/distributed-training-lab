"""Tiny synthetic char-level dataset.

Deliberately offline and deterministic: no downloads, no external files, so
this runs identically on a laptop, in CI, and on Kaggle. The "corpus" is a
public-domain pangram repeated to a fixed length. It is NOT meant to produce
a useful language model -- it exists so later phases (DDP, FSDP, profiling)
have a real forward/backward/optimizer step to compare against, with a loss
that measurably decreases (the pattern is trivially learnable), rather than
random noise.
"""

from __future__ import annotations

import torch

_PANGRAM = "the quick brown fox jumps over the lazy dog. "


def build_corpus(repeats: int = 200) -> str:
    """Return a deterministic char corpus by repeating a pangram."""
    return _PANGRAM * repeats


def build_vocab(corpus: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(corpus: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in corpus], dtype=torch.long)


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of (input, target) next-char sequences.

    Requires len(data) > block_size so at least one valid start index exists.
    """
    if len(data) <= block_size:
        raise ValueError(
            f"corpus length {len(data)} must exceed block_size {block_size}"
        )
    max_start = len(data) - block_size - 1
    ix = torch.randint(0, max_start + 1, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y
