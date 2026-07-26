"""Phase 4b: Pipeline parallelism (PP) stage-splitting for TinyGPT.

--------------------------------------------------------------------------
Step 1 finding (empirical -- see `.probe/pp_probe*.py`, run locally on this
machine: macOS/Apple Silicon, torch 2.13.0, no CUDA): does the real
`torch.distributed.pipelining` API (`PipelineStage`, `ScheduleGPipe`) work
on CPU with `gloo`? YES -- and unlike TP's DTensor path, this one needs NO
architecture rewrite at all, because PP splits the model at LAYER
boundaries (a sequence of whole `nn.Module`s), not inside a fused layer like
`nn.MultiheadAttention`.

  - `.probe/pp_probe.py`: a 2-stage `PipelineStage` + `ScheduleGPipe(
    n_microbatches=2)` runs without error on a real 2-process `gloo`
    process group (real `dist.send`/`dist.recv` of activations/gradients
    across the two OS processes, not simulated).
  - `.probe/pp_probe2.py`: the pipelined forward output is BIT-FOR-BIT
    identical (`max abs diff = 0.0`) to a non-pipelined, single-process
    reference forward through the same 2 blocks with the same weights --
    unlike TP, there is no reduction/associativity reordering here (PP is
    just sequential computation split across processes), so exact equality
    is the right bar, not "close."
  - `.probe/pp_probe3.py`: a full forward+loss+backward step through
    `ScheduleGPipe` produces per-rank `.grad` on each stage's own local
    parameters that match a single-process reference's gradients for the
    same submodule to ~1e-8 (float32 backward-pass noise, not exact, since
    backward re-derives gradients numerically through a different
    microbatch-chunked call graph).
  - `.probe/pp_probe4.py`: repeats the same check with TinyGPT's ACTUAL
    architecture (real token/position embeddings on stage 0, real
    `nn.TransformerEncoderLayer` blocks, real causal mask, integer token
    indices, cross-entropy loss on stage 1) -- forward logits match a
    single-process TinyGPT reference EXACTLY (`max diff = 0.0`), and
    gradients match to ~1e-8. So this module wraps the unmodified
    `TinyGPT` (src/model.py) by literally slicing its own submodules
    (`tok_emb`/`pos_emb`/`blocks.layers[:k]` for stage 0,
    `blocks.layers[k:]`/`ln_f`/`head` for the last stage) into per-stage
    `nn.Module`s -- NOT a rewritten model, unlike TP.

DECISION: use the real API directly (Step 2's "if the real API works, use
it" path) -- `train_pp.py` builds a `PipelineStage` per rank from one of the
`nn.Module`s below and runs `ScheduleGPipe` (GPipe schedule: all
microbatches' forwards, then all backwards).

HONESTY NOTE on overlap: `ScheduleGPipe`'s schedule is structured to allow
overlap (stage 0 can start microbatch k+1's forward while stage 1 works on
microbatch k), but on this machine (2 local OS processes on one CPU core
pool, `gloo` P2P send/recv) no wall-clock overlap is measured or claimed
here -- only the real multi-microbatch data flow and real cross-process
tensor communication are being verified. Actual pipeline-bubble reduction
from overlap is a throughput question that needs the Kaggle 2xT4 run (see
notebooks/kaggle_pp_2gpu.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model import TinyGPT


def stage_layer_ranges(n_layer: int, num_stages: int) -> list[tuple[int, int]]:
    """Contiguous, disjoint [start, end) layer-index ranges per stage --
    same "last stage absorbs the remainder" convention used by
    train_ddp.py's/train_fsdp.py's `make_rank_shard` for data shards,
    applied here to LAYERS instead of tokens."""
    if n_layer < num_stages:
        raise ValueError(f"n_layer={n_layer} must be >= num_stages={num_stages}")
    base = n_layer // num_stages
    ranges = []
    start = 0
    for s in range(num_stages):
        end = start + base if s < num_stages - 1 else n_layer
        ranges.append((start, end))
        start = end
    return ranges


class PPFirstStage(nn.Module):
    """Stage 0: token/position embeddings + this stage's slice of
    TinyGPT's transformer blocks. Shares the EXACT SAME parameter objects
    as the source `TinyGPT` (`model.tok_emb`, `model.blocks.layers[a:b]`,
    etc.), not copies -- so gradients computed here ARE the source model's
    gradients (verified in `.probe/pp_probe4.py` and
    `tests/test_train_pp.py`)."""

    def __init__(self, model: TinyGPT, layer_start: int, layer_end: int):
        super().__init__()
        self.tok_emb = model.tok_emb
        self.pos_emb = model.pos_emb
        self.drop = model.drop
        self.layers = nn.ModuleList(list(model.blocks.layers[layer_start:layer_end]))
        self.register_buffer("causal_mask", model.causal_mask, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return x


class PPMiddleStage(nn.Module):
    """An intermediate stage (only used when num_stages > 2): just this
    stage's slice of transformer blocks, operating on the activation
    received from the previous stage."""

    def __init__(self, model: TinyGPT, layer_start: int, layer_end: int):
        super().__init__()
        self.layers = nn.ModuleList(list(model.blocks.layers[layer_start:layer_end]))
        self.register_buffer("causal_mask", model.causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return x


class PPLastStage(nn.Module):
    """Final stage: this stage's slice of transformer blocks + the final
    LayerNorm + LM head, producing logits."""

    def __init__(self, model: TinyGPT, layer_start: int, layer_end: int):
        super().__init__()
        self.layers = nn.ModuleList(list(model.blocks.layers[layer_start:layer_end]))
        self.ln_f = model.ln_f
        self.head = model.head
        self.register_buffer("causal_mask", model.causal_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


class PPSingleStage(nn.Module):
    """Degenerate num_stages=1 case (see train_pp.py's module docstring
    "single process, degenerate world_size=1 sanity check"): ONE stage
    holds token/position embeddings + ALL transformer blocks + final
    LayerNorm + LM head -- i.e. the entire model, wired end to end so the
    stage's forward actually produces logits (not raw block output).
    Without this, `stage_index == 0` would take the `PPFirstStage` branch
    below (embeddings + blocks only, no `ln_f`/`head`) even when stage 0 is
    ALSO the last/only stage, silently producing a d_model-sized output
    instead of vocab_size-sized logits."""

    def __init__(self, model: TinyGPT, layer_start: int, layer_end: int):
        super().__init__()
        self.tok_emb = model.tok_emb
        self.pos_emb = model.pos_emb
        self.drop = model.drop
        self.layers = nn.ModuleList(list(model.blocks.layers[layer_start:layer_end]))
        self.ln_f = model.ln_f
        self.head = model.head
        self.register_buffer("causal_mask", model.causal_mask, persistent=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head(x)


def build_pp_stage_module(model: TinyGPT, stage_index: int, num_stages: int) -> nn.Module:
    """Return the `nn.Module` for `stage_index` of `num_stages`, slicing
    `model`'s own layers/submodules (no copy).

    `num_stages == 1` is checked FIRST and dispatched to `PPSingleStage`:
    stage 0 there is simultaneously the first AND last stage, and must own
    the final norm/LM head, which the plain `stage_index == 0` branch
    (`PPFirstStage`) does not include."""
    ranges = stage_layer_ranges(len(model.blocks.layers), num_stages)
    start, end = ranges[stage_index]
    if num_stages == 1:
        return PPSingleStage(model, start, end)
    if stage_index == 0:
        return PPFirstStage(model, start, end)
    if stage_index == num_stages - 1:
        return PPLastStage(model, start, end)
    return PPMiddleStage(model, start, end)
