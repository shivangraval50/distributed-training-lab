"""A deliberately small decoder-only transformer (char-level GPT).

Sized to run forward/backward on CPU in well under a second per step at the
default config. This is the model reused by the single-device baseline
(Phase 1) and, unmodified, by DDP/FSDP phases later -- the point of this repo
is comparing *how* the same model trains across parallelism strategies, not
the model itself.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

# Benign PyTorch-internal notice ("nested_tensor fast path disabled because
# norm_first=True") -- not actionable, not an error, just noisy in logs.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")


class TinyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 64,
        n_layer: int = 2,
        n_head: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((block_size, block_size), float("-inf")), diagonal=1),
            persistent=False,
        )

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        assert t <= self.block_size, f"sequence length {t} > block_size {self.block_size}"
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        mask = self.causal_mask[:t, :t]
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


def count_flops_per_token(model: TinyGPT) -> int:
    """Rough (non-exact) forward FLOPs/token estimate, ~2 * non-embedding params.

    Standard back-of-envelope used for transformer FLOP accounting (see e.g.
    Kaplan et al. 2020). Only used for logging/context, not a precise measurement.
    """
    non_embed_params = model.num_params() - model.tok_emb.weight.numel() - model.pos_emb.weight.numel()
    return 2 * non_embed_params
