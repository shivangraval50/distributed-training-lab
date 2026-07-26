"""Phase 4a: Tensor parallelism (TP) for a TinyGPT-shaped transformer.

--------------------------------------------------------------------------
Step 1 finding (empirical, not assumed -- see .probe/tp_probe*.py, run
locally on this machine: macOS/Apple Silicon, torch 2.13.0, no CUDA):

Does the REAL, modern `torch.distributed.tensor` (DTensor) +
`torch.distributed.tensor.parallel` API (`DeviceMesh`, `ColwiseParallel`,
`RowwiseParallel`, `parallelize_module`) work on CPU with `gloo`? YES, and it
is not just "doesn't crash" -- it is NUMERICALLY CORRECT:

  - `.probe/tp_probe.py`: `parallelize_module` with a `{"fc1": ColwiseParallel(),
    "fc2": RowwiseParallel()}` plan on a 2-layer MLP runs on a real 2-process
    `gloo` `DeviceMesh("cpu", ...)` without error.
  - `.probe/tp_probe2.py`: that same sharded MLP's output matches a
    non-parallel reference (identical weights, identical input) with
    max abs diff ~6e-8 (float32 noise) -- the real API's column/row split +
    implicit `all_reduce` genuinely reproduces the unsharded computation.

HOWEVER: it does NOT extend cleanly to TinyGPT's actual attention module,
`nn.MultiheadAttention` (used inside `nn.TransformerEncoderLayer`,
src/model.py). `parallelize_module`'s plan mechanism partitions NAMED
SUBMODULES that are `nn.Linear` or `nn.Embedding` -- but `MultiheadAttention`
packs Q/K/V into a single raw `nn.Parameter` (`in_proj_weight`, shape
`(3*d_model, d_model)`), not three separate `nn.Linear` submodules.
`.probe/tp_probe2.py` demonstrates this concretely: pointing a
`ColwiseParallel()` plan at the string `"in_proj_weight"` silently resolves
to nothing ("no submodule matching token ... skipping this plan entry" --
PyTorch warns and no-ops rather than erroring) because `in_proj_weight` is a
`Parameter`, not a `Module`. This is a genuine, real limitation of applying
this API to `nn.MultiheadAttention` specifically, not a CUDA/NCCL-only
assumption -- it would reproduce on GPU identically. This is exactly why
production TP frameworks (Megatron-LM, torchtitan, HF's TP support) define
their OWN attention modules with separate `wq`/`wk`/`wv`/`wo` `nn.Linear`
submodules instead of using `nn.MultiheadAttention` -- so this module does
the same thing.

DECISION: rather than hand-roll the collectives ourselves, this module
defines a `TPCausalSelfAttention` (separate `wq`/`wk`/`wv`/`wo` Linears,
numerically equivalent to `nn.MultiheadAttention` -- verified in
`tests/test_train_tp.py::test_tp_model_matches_reference_attention_math_unsharded`)
and a `TPMLP` (same `linear1`/`linear2` shape as
`nn.TransformerEncoderLayer`, renamed `fc1`/`fc2`), and shards THOSE with the
REAL `ColwiseParallel`/`RowwiseParallel` + `DeviceMesh` API -- i.e. Step 2's
"if the real API works, use it" path, applied at the granularity the real
API actually supports. The classic Megatron attention-TP pattern falls out
for free from the API's own defaults (verified via source inspection, see
`.probe/`):
  - `ColwiseParallel()` shards `wq`/`wk`/`wv`'s OUTPUT dim (`Shard(0)` on the
    weight) and returns a LOCAL (non-DTensor) tensor by default
    (`use_local_output=True`), i.e. each rank's projection output already
    contains only that rank's subset of attention heads -- no communication
    needed for Q/K/V.
  - `RowwiseParallel()` shards `wo`'s INPUT dim (`Shard(1)`) to match, and by
    default performs an implicit `all_reduce` (via DTensor's `Replicate()`
    output layout) then returns a local (already-summed) tensor -- exactly
    the "each rank computes its shard of the matmul, then combines via a
    real collective" mechanic requested for this phase, just performed by
    DTensor's machinery instead of a hand-written `dist.all_reduce` call.
  - The FFN (`fc1`/`fc2`) uses the identical Colwise-then-Rowwise pattern
    (Megatron's MLP TP): only ONE all_reduce per attention block and ONE per
    MLP block, not two, since the intermediate activation is already
    partitioned to match on both sides of the split.

Requires `n_head % world_size == 0` (each rank gets a whole, contiguous group
of attention heads) and `dim_feedforward % world_size == 0` -- a real,
documented constraint of head-parallel TP, not specific to this repo.

`tok_emb`, `pos_emb`, `ln_f`, `head` (vocab projection), and both LayerNorms
stay fully replicated (NOT sharded) on every rank -- a real simplification:
Megatron also supports vocab-parallel embeddings, which this repo does not
implement. Documented, not hidden -- see README Limitations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import TinyGPT


class TPCausalSelfAttention(nn.Module):
    """Megatron-style causal self-attention: separate wq/wk/wv/wo `nn.Linear`
    submodules (unlike `nn.MultiheadAttention`'s fused `in_proj_weight`) so
    the real `ColwiseParallel`/`RowwiseParallel` API can target them.

    Numerically equivalent to `nn.MultiheadAttention` given the same
    Q/K/V/out-proj weights (see module docstring + the unsharded-equivalence
    test) -- this is a faithful reimplementation, not a different attention
    variant.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=True)
        self.wk = nn.Linear(d_model, d_model, bias=True)
        self.wv = nn.Linear(d_model, d_model, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # After TP sharding, wq/wk/wv's output ("local_dim") is d_model /
        # world_size, i.e. only a subset of heads -- inferred from the
        # tensor's actual last-dim size (not a fixed self.n_head), so this
        # forward is correct whether or not `parallelize_module` has been
        # applied yet.
        local_dim = q.size(-1)
        n_head_local = local_dim // self.head_dim
        q = q.view(b, t, n_head_local, self.head_dim).transpose(1, 2)
        k = k.view(b, t, n_head_local, self.head_dim).transpose(1, 2)
        v = v.view(b, t, n_head_local, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, local_dim)
        return self.wo(y)  # RowwiseParallel: implicit all_reduce lands here


class TPMLP(nn.Module):
    """Same shape as TinyGPT's FFN (`linear1`/`linear2`, GELU), renamed
    `fc1`/`fc2` so the TP parallelize plan can target them by name."""

    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class TPBlock(nn.Module):
    """Same `norm_first` residual structure as
    `nn.TransformerEncoderLayer(norm_first=True)`:
        x = x + attn(norm1(x)); x = x + mlp(norm2(x))
    LayerNorms are NOT sharded (replicated, cheap, elementwise)."""

    def __init__(self, d_model: int, n_head: int, dim_feedforward: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TPCausalSelfAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = TPMLP(d_model, dim_feedforward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TPTinyGPT(nn.Module):
    """A TinyGPT-equivalent model built from TP-shardable blocks.

    Architecturally equivalent to `src.model.TinyGPT` (same embeddings,
    same per-block residual structure, same final norm + head) but with
    `TPBlock`s instead of `nn.TransformerEncoderLayer`s, so its
    attention/MLP Linear submodules can be targeted by the real
    `torch.distributed.tensor.parallel` API. See `load_from_tinygpt` for the
    weight-copy that makes an (unsharded) `TPTinyGPT` numerically match a
    reference `TinyGPT` on the same input.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 64,
        n_layer: int = 2,
        n_head: int = 2,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [TPBlock(d_model, n_head, 4 * d_model) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        assert t <= self.block_size
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def load_from_tinygpt(tp_model: TPTinyGPT, ref: TinyGPT) -> None:
    """Copy a reference `TinyGPT`'s weights into a `TPTinyGPT` of matching
    config, splitting `nn.MultiheadAttention`'s fused `in_proj_weight`
    (`(3*d_model, d_model)`, stacked [Wq; Wk; Wv]) into separate wq/wk/wv
    Linears. Requires both models to share `vocab_size`/`block_size`/
    `d_model`/`n_layer`/`n_head`.
    """
    with torch.no_grad():
        tp_model.tok_emb.weight.copy_(ref.tok_emb.weight)
        tp_model.pos_emb.weight.copy_(ref.pos_emb.weight)
        tp_model.ln_f.weight.copy_(ref.ln_f.weight)
        tp_model.ln_f.bias.copy_(ref.ln_f.bias)
        tp_model.head.weight.copy_(ref.head.weight)

        for tp_block, ref_layer in zip(tp_model.blocks, ref.blocks.layers):
            d_model = tp_block.attn.wq.in_features
            mha = ref_layer.self_attn
            tp_block.attn.wq.weight.copy_(mha.in_proj_weight[0:d_model])
            tp_block.attn.wk.weight.copy_(mha.in_proj_weight[d_model : 2 * d_model])
            tp_block.attn.wv.weight.copy_(mha.in_proj_weight[2 * d_model : 3 * d_model])
            tp_block.attn.wq.bias.copy_(mha.in_proj_bias[0:d_model])
            tp_block.attn.wk.bias.copy_(mha.in_proj_bias[d_model : 2 * d_model])
            tp_block.attn.wv.bias.copy_(mha.in_proj_bias[2 * d_model : 3 * d_model])
            tp_block.attn.wo.weight.copy_(mha.out_proj.weight)
            tp_block.attn.wo.bias.copy_(mha.out_proj.bias)

            tp_block.mlp.fc1.weight.copy_(ref_layer.linear1.weight)
            tp_block.mlp.fc1.bias.copy_(ref_layer.linear1.bias)
            tp_block.mlp.fc2.weight.copy_(ref_layer.linear2.weight)
            tp_block.mlp.fc2.bias.copy_(ref_layer.linear2.bias)

            tp_block.norm1.weight.copy_(ref_layer.norm1.weight)
            tp_block.norm1.bias.copy_(ref_layer.norm1.bias)
            tp_block.norm2.weight.copy_(ref_layer.norm2.weight)
            tp_block.norm2.bias.copy_(ref_layer.norm2.bias)


def build_tp_plan(n_layer: int) -> dict:
    """The real `torch.distributed.tensor.parallel` parallelize plan: for
    every block, Colwise-shard wq/wk/wv/fc1 (head-/hidden-dim-parallel
    output split, no communication) and Rowwise-shard wo/fc2 (implicit
    all_reduce on the way out) -- the standard Megatron attention+MLP TP
    plan, using only the library's own `ColwiseParallel`/`RowwiseParallel`
    (default layouts -- see module docstring for exactly what those default
    to and why they compose correctly here).
    """
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    plan = {}
    for i in range(n_layer):
        plan[f"blocks.{i}.attn.wq"] = ColwiseParallel()
        plan[f"blocks.{i}.attn.wk"] = ColwiseParallel()
        plan[f"blocks.{i}.attn.wv"] = ColwiseParallel()
        plan[f"blocks.{i}.attn.wo"] = RowwiseParallel()
        plan[f"blocks.{i}.mlp.fc1"] = ColwiseParallel()
        plan[f"blocks.{i}.mlp.fc2"] = RowwiseParallel()
    return plan
