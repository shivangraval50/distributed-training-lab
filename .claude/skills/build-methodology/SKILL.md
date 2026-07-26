---
name: build-methodology
description: The mandatory workflow and rules for building this project phase-by-phase. Follow it for every phase and every delegation.
---

# Build methodology

Build the project defined in this repo's PLAN.md, one phase at a time, in order.

## Per-phase loop
1. Implement the phase (builder).
2. Write REAL tests that exercise the code, and RUN them until green (tester).
3. Review for honesty + quality (reviewer): no fabricated results, tests genuinely ran,
   GPU phases stubbed not faked. Loop back to the builder on CHANGES-NEEDED.
4. Commit on branch `build/phase-work` with a clear message; check the phase's box in PLAN.md.

## Hard rules (never break)
- NEVER fabricate, guess, or hard-code any result, metric, accuracy, or latency. Numbers come
  ONLY from code that actually ran. Until then the README Results table stays "TODO".
- If a phase needs a GPU or an environment this machine lacks (model training, CUDA, multi-GPU,
  x86/AVX2/io_uring benchmarks): implement the code, add a LOCAL smoke test (a few steps on a
  tiny CPU model / tiny input), AND write a ready-to-run script in notebooks/ for Kaggle. Leave
  that phase UNCHECKED with a note "TODO: run on Kaggle". Do NOT mark it done. Do NOT invent numbers.
- Branch `build/phase-work` only. Never push to main. Never force-push.
- No secrets in the repo. Weights/datasets go to Hugging Face or notebook output, never git.
- Keep everything runnable on macOS / 8GB where possible.

## Definition of done (per phase)
Code runs; real tests pass (or, for GPU phases, a smoke test passes and a Kaggle notebook exists);
reviewer approved; README/PLAN reflect reality including limits; committed on the branch.
