# CLAUDE.md -- Distributed training; profile scaling on Kaggle's 2xT4.

## Context
One of five portfolio projects for ML/AI + quant roles. The hiring signal is DEPTH and
HONEST RIGOR, not breadth or scale. Assume an expert reads this code and grills me on it.
Thesis across all five: built on constrained hardware (M2, 8GB, no local GPU) -- "where does
every byte / microsecond / dollar actually go?" Efficiency over throwing hardware at problems.

## Hard rules (never break)
1. Honesty over impressiveness. Never fabricate benchmarks or results. State real scale
   ("30M params, not 30B"; "2 GPUs, not a cluster"). Anything not built/measured yet is TODO.
2. README = Problem -> Approach -> Results -> Limitations. Every README ends with an honest
   "Limitations / what's unrealistic" section. No marketing language.
3. Explain as you go: a line or two on what you're about to do and why, before non-trivial work.
4. Work on a branch, small clear commits (what + why). Smoke-test before committing.
5. NEVER push broken code. If you can't fix it cleanly, stop and leave a TODO.
6. No secrets in the repo. Weights/datasets go to the Hugging Face Hub or are gitignored.

## Environment
Local = macOS Apple Silicon, 8GB. Heavy GPU work (training, inference benchmarks) runs on
Kaggle (2xT4, 30h/wk) or Colab (T4), authored as scripts in notebooks/. Do NOT assume a local GPU.

## Layout
README.md - PLAN.md (phases as a checklist) - notebooks/ - tests/ - .github/workflows/ci.yml - .gitignore
Definition of done per phase: runs, has a test, README/PLAN reflect reality (incl. limits), committed on a branch.
