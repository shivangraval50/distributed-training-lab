"""Real tests for src/model.py (TinyGPT).

Exercises actual forward passes and parameter accounting, not placeholders.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import TinyGPT, count_flops_per_token  # noqa: E402


def _make_model(vocab_size=11, block_size=16, d_model=16, n_layer=2, n_head=2):
    return TinyGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
    )


def test_forward_output_shape():
    model = _make_model()
    batch, seq_len = 3, 10
    idx = torch.randint(0, 11, (batch, seq_len))
    logits = model(idx)
    assert logits.shape == (batch, seq_len, 11)


def test_forward_rejects_sequence_longer_than_block_size():
    model = _make_model(block_size=8)
    idx = torch.randint(0, 11, (2, 9))  # 9 > block_size=8
    with pytest.raises(AssertionError):
        model(idx)


def test_forward_accepts_sequence_at_exactly_block_size():
    model = _make_model(block_size=8)
    idx = torch.randint(0, 11, (2, 8))
    logits = model(idx)
    assert logits.shape == (2, 8, 11)


def test_num_params_matches_manual_sum():
    model = _make_model()
    manual_total = sum(p.numel() for p in model.parameters())
    assert model.num_params() == manual_total
    assert model.num_params() > 0


def test_num_params_scales_with_depth():
    small = _make_model(n_layer=1)
    big = _make_model(n_layer=4)
    assert big.num_params() > small.num_params()


def test_count_flops_per_token_excludes_embeddings_and_is_positive():
    model = _make_model()
    flops = count_flops_per_token(model)
    non_embed_params = (
        model.num_params()
        - model.tok_emb.weight.numel()
        - model.pos_emb.weight.numel()
    )
    assert flops == 2 * non_embed_params
    assert flops > 0
    # Sanity: embeddings contribute to num_params but not to the FLOPs proxy.
    assert flops < 2 * model.num_params()


def test_gradients_flow_on_backward():
    # A real learning-signal precondition: every parameter should receive a
    # gradient after backward(), not silently stay disconnected from the graph.
    model = _make_model()
    idx = torch.randint(0, 11, (2, 6))
    targets = torch.randint(0, 11, (2, 6))
    logits = model(idx)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached parameter {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_causal_mask_is_upper_triangular_inf():
    model = _make_model(block_size=5)
    mask = model.causal_mask
    assert mask.shape == (5, 5)
    # Diagonal and below must be unmasked (0), strictly above must be -inf.
    for i in range(5):
        for j in range(5):
            if j > i:
                assert mask[i, j] == float("-inf")
            else:
                assert mask[i, j] == 0
