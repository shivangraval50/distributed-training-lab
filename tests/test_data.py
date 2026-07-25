"""Real tests for src/data.py (synthetic char corpus + batching)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_corpus, build_vocab, encode, get_batch  # noqa: E402


def test_build_corpus_deterministic_and_scales_with_repeats():
    c1 = build_corpus(repeats=5)
    c2 = build_corpus(repeats=5)
    assert c1 == c2
    assert len(c1) == 5 * len("the quick brown fox jumps over the lazy dog. ")
    assert len(build_corpus(repeats=10)) == 2 * len(c1)


def test_build_vocab_covers_all_chars_and_is_bijective():
    corpus = build_corpus(repeats=3)
    stoi, itos = build_vocab(corpus)
    assert set(stoi.keys()) == set(corpus)
    # stoi/itos must be exact inverses of each other.
    for ch, i in stoi.items():
        assert itos[i] == ch
    assert len(stoi) == len(itos) == len(set(corpus))


def test_encode_round_trips_through_vocab():
    corpus = build_corpus(repeats=2)
    stoi, itos = build_vocab(corpus)
    encoded = encode(corpus, stoi)
    assert encoded.dtype == torch.long
    assert encoded.shape == (len(corpus),)
    decoded = "".join(itos[i.item()] for i in encoded)
    assert decoded == corpus


def test_get_batch_shapes_and_shift_by_one():
    corpus = build_corpus(repeats=5)
    stoi, _ = build_vocab(corpus)
    data = encode(corpus, stoi)

    batch_size, block_size = 4, 12
    gen = torch.Generator().manual_seed(0)
    x, y = get_batch(data, batch_size, block_size, generator=gen)

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # y is x shifted by exactly one position into the underlying data stream:
    # for every row, y[:-1] == x[1:] would only hold within a single sampled
    # window if x and y were sourced from the same start+1 offset, which is
    # exactly how get_batch constructs them.
    for row in range(batch_size):
        assert torch.equal(y[row][:-1], x[row][1:])


def test_get_batch_indices_within_bounds():
    corpus = build_corpus(repeats=3)
    stoi, _ = build_vocab(corpus)
    data = encode(corpus, stoi)
    x, y = get_batch(data, batch_size=8, block_size=10)
    assert x.min() >= 0 and x.max() < len(stoi)
    assert y.min() >= 0 and y.max() < len(stoi)


def test_get_batch_raises_when_corpus_too_short():
    corpus = "abc"
    stoi, _ = build_vocab(corpus)
    data = encode(corpus, stoi)
    with pytest.raises(ValueError):
        get_batch(data, batch_size=2, block_size=10)


def test_get_batch_is_deterministic_with_seeded_generator():
    corpus = build_corpus(repeats=5)
    stoi, _ = build_vocab(corpus)
    data = encode(corpus, stoi)

    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    x1, y1 = get_batch(data, 4, 12, generator=gen_a)
    x2, y2 = get_batch(data, 4, 12, generator=gen_b)

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
