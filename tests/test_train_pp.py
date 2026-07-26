"""Real correctness test for Phase 4b Pipeline Parallelism (train_pp.py).

Launches genuine multi-process `torch.distributed` runs on CPU with the real
`torch.distributed.pipelining` API (`PipelineStage` + `ScheduleGPipe`, real
`dist.send`/`dist.recv` of activations/gradients between OS processes -- see
`src/pipeline_model.py`'s module docstring for the Step 1 investigation) and
verifies:

1. Each stage's [layer_start, layer_end) range (`stage_layer_ranges`) is
   disjoint, contiguous, and covers [0, n_layer) across all ranks -- for 2
   and 3 ranks.
2. Each rank's LOCAL parameter count (`local_stage_numel`) matches an
   INDEPENDENTLY computed expectation derived from which submodules that
   stage owns (embeddings on the first stage, final norm/head on the last
   stage, its own layer slice everywhere) -- built fresh from `src.model.
   TinyGPT`, not by re-calling `src.pipeline_model.build_pp_stage_module`.
   Since PP shards LAYERS (no replication, unlike DDP/FSDP/TP), the sum of
   every rank's local_stage_numel across the whole world is also checked to
   equal the full model's parameter count exactly -- there is nothing
   replicated to double-count.
3. THE key correctness proof this phase's docstrings claim: after N real
   pipelined training steps split across 2 (and separately 3) processes,
   the trained per-stage weights match an INDEPENDENT single-process
   TinyGPT reference trained on the identical (same-seed, no broadcast
   needed -- PP ranks build from a shared fixed seed, not a broadcast-synced
   one, unlike DDP/FSDP/TP) initial weights/data/optimizer steps, with an
   asserted numerical tolerance (not just "did not crash").
4. The degenerate `world_size=1` case (a single rank simultaneously first
   and last stage, `PPSingleStage`) runs without crashing and its trained
   weights also match the same single-process reference construction.

Uses torch.multiprocessing.spawn, same reasoning as test_train_ddp.py/
test_train_fsdp.py/test_train_tp.py: functionally equivalent to torchrun
(same init_process_group/PipelineStage code path, real OS processes, real
gloo collectives) and avoids torchrun's elastic rendezvous hanging in this
sandbox (see those tests' docstrings for the exact failure mode observed).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]

from src.data import build_corpus, build_vocab, encode, get_batch  # noqa: E402
from src.model import TinyGPT  # noqa: E402


def _worker(rank: int, world_size: int, out_dir: str, argv: list[str]) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29721")

    sys.path.insert(0, str(REPO_ROOT))
    from train_pp import build_arg_parser, train  # noqa: E402

    args = build_arg_parser().parse_args(argv)
    train(args)


def _run_pp(tmp_path: Path, world_size: int = 2, master_port: str = "29721", **overrides) -> tuple[Path, dict]:
    out_dir = tmp_path / "pp_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MASTER_PORT"] = master_port

    defaults = dict(
        steps=15, batch_size=8, block_size=16, d_model=24, n_layer=2, n_head=4,
        microbatches=2, lr=3e-4, seed=0, log_every=1000, backend="gloo",
    )
    defaults.update(overrides)
    argv = [
        "--steps", str(defaults["steps"]),
        "--batch-size", str(defaults["batch_size"]),
        "--block-size", str(defaults["block_size"]),
        "--d-model", str(defaults["d_model"]),
        "--n-layer", str(defaults["n_layer"]),
        "--n-head", str(defaults["n_head"]),
        "--microbatches", str(defaults["microbatches"]),
        "--lr", str(defaults["lr"]),
        "--seed", str(defaults["seed"]),
        "--log-every", str(defaults["log_every"]),
        "--backend", defaults["backend"],
        "--out-dir", str(out_dir),
    ]
    mp.spawn(_worker, args=(world_size, str(out_dir), argv), nprocs=world_size, join=True)
    return out_dir, defaults


def _load_rank_artifacts(out_dir: Path, world_size: int) -> tuple[list[dict], list[dict]]:
    hists = [json.loads((out_dir / f"rank{r}_history.json").read_text()) for r in range(world_size)]
    states = [torch.load(out_dir / f"rank{r}_stage_state.pt", map_location="cpu") for r in range(world_size)]
    return hists, states


def _expected_stage_numel(
    vocab_size: int, block_size: int, d_model: int, n_layer: int, n_head: int,
    layer_start: int, layer_end: int, is_first: bool, is_last: bool,
) -> int:
    """Independently compute the expected LOCAL parameter count for a PP
    stage: built directly from a fresh `src.model.TinyGPT` (NOT by calling
    `src.pipeline_model.build_pp_stage_module`), reasoning purely from the
    architecture docstring's description of which submodules each stage
    owns -- first stage additionally owns tok_emb/pos_emb, last stage
    additionally owns ln_f/head, every stage owns its own contiguous slice
    of transformer blocks.
    """
    model = TinyGPT(vocab_size=vocab_size, block_size=block_size, d_model=d_model, n_layer=n_layer, n_head=n_head, dropout=0.0)
    total = sum(p.numel() for layer in model.blocks.layers[layer_start:layer_end] for p in layer.parameters())
    if is_first:
        total += model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    if is_last:
        total += sum(p.numel() for p in model.ln_f.parameters()) + model.head.weight.numel()
    return total


def _max_abs_diff_stage_vs_reference(
    stage_state: dict, reference: TinyGPT, layer_start: int, layer_end: int, has_embed: bool, has_head: bool,
) -> float:
    """Compare a saved PP stage's state dict against the corresponding
    submodules of an independently-trained single-process reference model,
    returning the max abs diff found across every compared tensor."""
    max_diff = 0.0
    with torch.no_grad():
        if has_embed:
            max_diff = max(max_diff, (stage_state["tok_emb.weight"] - reference.tok_emb.weight).abs().max().item())
            max_diff = max(max_diff, (stage_state["pos_emb.weight"] - reference.pos_emb.weight).abs().max().item())
        for local_i, global_i in enumerate(range(layer_start, layer_end)):
            ref_layer_sd = reference.blocks.layers[global_i].state_dict()
            for key, ref_t in ref_layer_sd.items():
                stage_t = stage_state[f"layers.{local_i}.{key}"]
                max_diff = max(max_diff, (stage_t - ref_t).abs().max().item())
        if has_head:
            max_diff = max(max_diff, (stage_state["ln_f.weight"] - reference.ln_f.weight).abs().max().item())
            max_diff = max(max_diff, (stage_state["ln_f.bias"] - reference.ln_f.bias).abs().max().item())
            max_diff = max(max_diff, (stage_state["head.weight"] - reference.head.weight).abs().max().item())
    return max_diff


def _train_single_process_reference(cfg: dict, steps: int) -> TinyGPT:
    """Reproduce train_pp.py's exact single-process computation: SAME fixed
    seed on every rank (no per-rank offset, unlike DDP/TP -- see train_pp.py's
    module docstring), SAME data-generator seed, one AdamW instance over the
    WHOLE model (equivalent to N per-stage AdamW instances over disjoint
    parameter sets, since Adam's update is computed independently per
    parameter with no cross-parameter coupling like global grad clipping)."""
    corpus = build_corpus(repeats=max(50, cfg["batch_size"] * cfg["block_size"] // 10 + 50))
    stoi, _ = build_vocab(corpus)
    full_data = encode(corpus, stoi)

    torch.manual_seed(cfg["seed"])
    reference = TinyGPT(
        vocab_size=len(stoi), block_size=cfg["block_size"], d_model=cfg["d_model"],
        n_layer=cfg["n_layer"], n_head=cfg["n_head"], dropout=0.0,
    )
    optimizer = torch.optim.AdamW(reference.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg["seed"] + 1)
    reference.train()
    for _ in range(steps):
        x, y = get_batch(full_data, cfg["batch_size"], cfg["block_size"], generator=gen)
        logits = reference(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return reference


def test_pp_layer_ranges_disjoint_contiguous_2rank(tmp_path):
    out_dir, cfg = _run_pp(tmp_path, world_size=2, master_port="29722", n_layer=4)
    hists, _ = _load_rank_artifacts(out_dir, 2)

    ranges = sorted((h["layer_start"], h["layer_end"]) for h in hists)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == cfg["n_layer"]
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1  # contiguous, no gap, no overlap
    # Even n_layer split across 2 ranks: exactly half the layers each.
    assert ranges[0][1] - ranges[0][0] == cfg["n_layer"] // 2
    assert ranges[1][1] - ranges[1][0] == cfg["n_layer"] // 2


def test_pp_layer_ranges_disjoint_contiguous_3rank(tmp_path):
    out_dir, cfg = _run_pp(tmp_path, world_size=3, master_port="29723", n_layer=4, d_model=24, n_head=4, steps=5)
    hists, _ = _load_rank_artifacts(out_dir, 3)

    ranges = sorted((h["layer_start"], h["layer_end"]) for h in hists)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == cfg["n_layer"]
    for (s0, e0), (s1, _e1) in zip(ranges, ranges[1:]):
        assert e0 == s1
    # "last stage absorbs remainder" convention: base = 4 // 3 = 1 layer/stage
    # for stages 0 and 1, stage 2 gets the remaining 2.
    assert [e - s for s, e in ranges] == [1, 1, 2]


def test_pp_local_stage_numel_matches_independent_expectation_2rank(tmp_path):
    out_dir, cfg = _run_pp(tmp_path, world_size=2, master_port="29724")
    hists, _ = _load_rank_artifacts(out_dir, 2)

    corpus = build_corpus(repeats=max(50, cfg["batch_size"] * cfg["block_size"] // 10 + 50))
    stoi, _ = build_vocab(corpus)
    full_params = hists[0]["full_params"]
    assert hists[1]["full_params"] == full_params

    total_local = 0
    for h in hists:
        is_first = h["rank"] == 0
        is_last = h["rank"] == h["world_size"] - 1
        expected = _expected_stage_numel(
            len(stoi), cfg["block_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"],
            h["layer_start"], h["layer_end"], is_first, is_last,
        )
        assert h["local_stage_numel"] < full_params  # genuine fraction, not the whole model
        assert h["local_stage_numel"] == expected
        total_local += h["local_stage_numel"]

    # PP shards LAYERS with no replication (unlike DDP/FSDP/TP): every
    # parameter is owned by EXACTLY one stage, so the sum across ranks must
    # equal the full model's parameter count exactly.
    assert total_local == full_params


def test_pp_local_stage_numel_matches_independent_expectation_3rank(tmp_path):
    out_dir, cfg = _run_pp(tmp_path, world_size=3, master_port="29725", n_layer=4, d_model=24, n_head=4, steps=5)
    hists, _ = _load_rank_artifacts(out_dir, 3)

    corpus = build_corpus(repeats=max(50, cfg["batch_size"] * cfg["block_size"] // 10 + 50))
    stoi, _ = build_vocab(corpus)
    full_params = hists[0]["full_params"]

    total_local = 0
    for h in hists:
        is_first = h["rank"] == 0
        is_last = h["rank"] == h["world_size"] - 1
        expected = _expected_stage_numel(
            len(stoi), cfg["block_size"], cfg["d_model"], cfg["n_layer"], cfg["n_head"],
            h["layer_start"], h["layer_end"], is_first, is_last,
        )
        assert h["local_stage_numel"] < full_params
        assert h["local_stage_numel"] == expected
        total_local += h["local_stage_numel"]
    assert total_local == full_params


def test_pp_final_weights_match_independent_single_process_reference_2rank(tmp_path):
    """THE key correctness proof: run real 2-process pipelined training,
    then independently (single process, no pipelining) reproduce the
    identical initial weights/data/optimizer steps and check the two match
    to a real, asserted numerical tolerance."""
    steps = 15
    out_dir, cfg = _run_pp(tmp_path, world_size=2, master_port="29726", steps=steps)
    hists, states = _load_rank_artifacts(out_dir, 2)

    reference = _train_single_process_reference(cfg, steps)

    max_diff = 0.0
    for h, state in zip(hists, states):
        is_first = h["rank"] == 0
        is_last = h["rank"] == h["world_size"] - 1
        diff = _max_abs_diff_stage_vs_reference(state, reference, h["layer_start"], h["layer_end"], is_first, is_last)
        max_diff = max(max_diff, diff)

    print(f"\n[test] 2-rank PP vs independent single-process reference after {steps} steps: "
          f"max abs weight diff = {max_diff:.3e}")

    # Per src/pipeline_model.py's docstring probes: forward is bit-for-bit
    # identical and single-step gradients match to ~1e-8/~1e-9 (no
    # all_reduce reordering like DDP/TP -- PP is pure sequential compute
    # split across processes). Over `steps` AdamW updates that per-step
    # noise floor compounds; observed in practice (deterministic, stable
    # across repeated runs) is ~5.7e-5 after 15 steps here. 2e-4 leaves
    # headroom above that real, measured floor while still being a
    # genuinely discriminating check (a real bug -- wrong layer boundary,
    # dropped send/recv, wrong microbatch grad scaling -- would show up as
    # O(0.1-1+) differences, not this).
    assert max_diff < 2e-4, f"PP weights diverged from single-process reference (max diff {max_diff:.3e})"


def test_pp_final_weights_match_independent_single_process_reference_3rank(tmp_path):
    steps = 10
    out_dir, cfg = _run_pp(
        tmp_path, world_size=3, master_port="29727", steps=steps,
        n_layer=4, d_model=24, n_head=4,
    )
    hists, states = _load_rank_artifacts(out_dir, 3)

    reference = _train_single_process_reference(cfg, steps)

    max_diff = 0.0
    for h, state in zip(hists, states):
        is_first = h["rank"] == 0
        is_last = h["rank"] == h["world_size"] - 1
        diff = _max_abs_diff_stage_vs_reference(state, reference, h["layer_start"], h["layer_end"], is_first, is_last)
        max_diff = max(max_diff, diff)

    print(f"\n[test] 3-rank PP vs independent single-process reference after {steps} steps: "
          f"max abs weight diff = {max_diff:.3e}")
    # Observed in practice (deterministic, stable across repeated runs):
    # ~6.3e-5 after 10 steps here -- same noise-floor reasoning as the
    # 2-rank test above.
    assert max_diff < 2e-4, f"PP weights diverged from single-process reference (max diff {max_diff:.3e})"


def test_pp_last_rank_loss_actually_drops(tmp_path):
    """Sanity check that the pipelined run is genuinely training, not a
    no-op -- only the last stage computes/reports a loss."""
    out_dir, cfg = _run_pp(tmp_path, world_size=2, master_port="29728", steps=25, log_every=1)
    hists, _ = _load_rank_artifacts(out_dir, 2)
    last = [h for h in hists if h["rank"] == h["world_size"] - 1][0]
    losses = [r["loss"] for r in last["history"]]
    assert all(v is not None for v in losses)
    assert losses[-1] < losses[0]

    first = [h for h in hists if h["rank"] == 0][0]
    first_losses = [r["loss"] for r in first["history"]]
    assert all(v is None for v in first_losses)  # only the last stage sees `target`/reports loss


def test_pp_world_size_one_degenerate_matches_reference(tmp_path):
    """The just-fixed world_size=1 degenerate case: rank 0 is simultaneously
    first AND last stage (`PPSingleStage`, whole model, no real cross-process
    pipelining). Verify it runs without crashing and produces weights
    matching the same single-process reference construction."""
    steps = 15
    out_dir, cfg = _run_pp(tmp_path, world_size=1, master_port="29729", steps=steps)
    hists, states = _load_rank_artifacts(out_dir, 1)

    h, state = hists[0], states[0]
    assert h["layer_start"] == 0
    assert h["layer_end"] == cfg["n_layer"]
    assert h["local_stage_numel"] == h["full_params"]  # single stage owns EVERYTHING

    losses = [r["loss"] for r in h["history"]]
    assert all(v is not None for v in losses)  # rank 0 is also the last stage here
    assert losses[-1] < losses[0]

    reference = _train_single_process_reference(cfg, steps)
    max_diff = _max_abs_diff_stage_vs_reference(state, reference, 0, cfg["n_layer"], has_embed=True, has_head=True)
    print(f"\n[test] world_size=1 degenerate PP vs independent single-process reference after {steps} steps: "
          f"max abs weight diff = {max_diff:.3e}")
    # Same noise-floor reasoning as the 2-rank test (identical config here,
    # so the same ~5.7e-5 is expected and observed).
    assert max_diff < 2e-4, f"world_size=1 PP weights diverged from single-process reference (max diff {max_diff:.3e})"
