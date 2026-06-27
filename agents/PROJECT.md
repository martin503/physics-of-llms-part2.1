# Physics of Language Models: Part 2.1

This project aims to reproduce interp and training parts of paper "[Physics of Language Models: Part 2.1](https://physics.allen-zhu.com/part-2-grade-school-math/part-2-1)". To be precise, atm we want to focus on Section 4 and training the model needed to to the probing. [Here](https://arxiv.org/abs/2407.20311) is the paper and [here](https://github.com/facebookresearch/iGSM) is the codebase for data gen.

## 1. General Idea

### The paper in one paragraph
*Physics of Language Models: Part 2.1 — Grade-School Math and the Hidden Reasoning Process*
([arXiv:2407.20311](https://arxiv.org/abs/2407.20311); Ye, Xu, Li, Allen-Zhu) asks a deceptively
simple question: **when an LM solves a math word problem, is it actually reasoning — or just
pattern-matching / memorising?** To study this cleanly the authors build **iGSM**, a *fully
synthetic* grade-school-math dataset generated from random dependency graphs with a
hierarchical vocabulary and **mod-23 arithmetic**. The problem space is astronomically large, so
there is zero data contamination and no chance of memorising specific problems — any correct
generalisation must come from learned reasoning. They then train small GPT-2 models from scratch
on iGSM and **probe their internal states** to reverse-engineer the hidden "reasoning process".

### What we reproduce (and what we don't)
- ✅ **Training setup** (paper Appendix F / Part 2.2 App. D): a from-scratch **GPT2-12-12**
  (12 layers × 12 heads, hidden 768, FFN 3072, ~125M params) **with RoPE**, trained on
  **iGSM-med** (`max_op=15, max_edge=20, perm_level=5, detail_level=0`), context 768; AdamW
  betas (0.9, 0.98), lr 2e-3, cosine decay → 0.01× with 1000-step warmup, weight decay 0.05,
  global batch 512, bf16, **no loss masking**; token layout
  `[222] problem [223] solution [224] answer [50256]`. The *pipeline* is done & CPU-validated
  (see code map); only the environment gates remain (checkout the `iGSM/` submodule, `uv lock`,
  run on the 2×3090).
- 🔄 **Section 4 — probing the hidden reasoning process** (the actual interp target, not yet
  implemented). The paper introduces **V-probes** (a linear probe on the output **plus** a
  low-rank modification of the token embeddings) and reads out, at every solution step, the
  quantities the model is *supposed* to be tracking internally:
  - `value(A)` — numeric value of parameter A (0–22, or 23 = unknown);
  - `can_next(A)` / `nece_next(A)` — A can / must be computed next (predecessors done / it's needed);
  - `nece(A)` — A is needed for the answer; `dep(A, B)` — A (recursively) depends on B; `known(A)`.
  The key control: applying the *same* probe to a **randomly-initialised** transformer shows
  little uplift, isolating what training actually contributed. This requires writing the probes
  + an iGSM solution **parser** (to label probe targets per step) from scratch — the paper's
  probe code isn't released.

> Dataset generation is **not** reimplemented — we reuse the authors'
> [facebookresearch/iGSM](https://github.com/facebookresearch/iGSM) generator, vendored as a git
> submodule at `iGSM/` (it has no `__init__.py` → put on `sys.path`; needs `networkx`).

## 2. Bug Fixes / Lessons Learned

TBD

## 3. Test Scenarios / Edge Cases

TBD
