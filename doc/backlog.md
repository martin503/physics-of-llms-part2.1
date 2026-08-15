
| size | feature |
|:----:|---------|
| small | save full random model for probe to be more code independent? |
| medium | DONE (highest priority) successfully train v-probe on dep(A, B) task |
| small | generate regular continuations on probe prompts to justify need for training probes. |
| small | DECIDED AGAINST for probe testing (see project_log 2026-07-18: probes are supervised classification, inspect is built for generation evals; custom `report-dep` covers sample inspection). Inspect still makes sense later for behavioral evals of the LM itself (iGSM answer accuracy). |
| medium | DONE for dep(A, B): `test` + `report-dep` commands (interactive graph report, see probe_guide "Testing a trained probe"). Still open for nece. |
| medium | DONE per-problem query cap (Appendix E): `--max-queries` (default 10) samples uniformly without replacement inside the `QUERY_BUILDERS`. `dep` splits the budget evenly per class, so a positive-poor problem gives 3+3 not 3+7 -- a deliberate deviation from the paper, which draws pairs uniformly; `--unbalanced` is that paper-literal rule. Batch composition fixed too: shuffling indices before the stable length-sort took distinct problems per batch of 8 from 5.12 to 7.41 at identical padding (`scripts/debug/batch_composition.py`, 300 problems at the cap; the effect is far larger uncapped). |
| medium | retrain the dep random control: the run started 2026-07-18 19:05 predates the seeded random-init fix, so its `probe.pt` cannot be re-paired with its transformer for test-time evaluation (training-time val metrics remain valid). |
| medium | DONE uniform difficulty: `--uniform-difficulty` pins each problem's op via `IdGen(op=...)`, cycling `1..max_op` by global problem index so every prefix is balanced too. Costs 1.84x per problem; 0 of 222k queries hit MAX_SEQ_LEN even at op 23. `find-seeds` / `--seeds-file` retired in favour of it. |
| small | DONE probe training epochs: `--epochs` now defaults to 1 (one pass, no duplicate queries) in both `run.py` and `train_vprobe`. |
| medium | remove the linear-probe pipeline: `probe.py`, `extract.py`, the `extract`/`train` CLI commands, and `tests/test_probe.py`. It was added on a misunderstanding of what the probes are for; the V-probe is the real thing. The probe_guide sections are already gone, so those two CLI commands are currently undocumented. Removing the code also retires the last legitimate uses of "row" in a probe context (see d22756c). |
| small | move test files to mirror the source tree (probe tests into their own subdirectory). |
| small | DONE (d22756c) rename the probe's `row` concept to `query`: `VProbeRow` -> `ProbeQuery`, and the metadata keys `rows_version`/`n_rows` -> `queries_version`/`n_queries` (`QUERIES_VERSION` = 3). |
| small | `.gitmodules` points the iGSM submodule at `org-16943930@github.com:facebookresearch/iGSM.git`, an SSH alias that does not resolve, so `git submodule update --init iGSM` always fails. Use the public HTTPS URL. Locally (2026-08-07) the checkout was repaired by hand: a clone had landed in `iGSM/iGSM/` instead of `iGSM/`, so `iGSM/` is now a plain clone at the pinned `a1ed1d0`, not a registered submodule. |
| small | DONE probe_guide.md's module table listed `queries.py`; the file has been `build_queries.py` since 1053143. |
| small | `tests/test_gpt.py` DDP tests fail on this Windows box: `makeDeviceForHostname(): unsupported gloo device` (torch cannot bind the hostname). Skip them on Windows, or mark them as requiring a working gloo backend. |
| small | `vprobe`'s online path (`build_vprobe_queries`, no `--data`) inherits `max_queries=10` with no CLI flag to override it. Add `--max-queries` to the `vprobe` command, or drop the online path when the linear-probe pipeline goes. |
| small | bare `uv run pytest` cannot import `src` (cwd is not on `sys.path`); `uv run python -m pytest` works. Fix with a `pythonpath = ["."]` in the pytest config. |
---
_postponed:_
|||
|:----:|---------|
| small | ~integrate probe training with WandB to track progress, see visual learning behavior and keep a record of past runs.~ |
| large | Add statistical analysis to gain more trustworthy results. (First for dep(A, B)) |
| large | add remaining probes from paper, documented in [[paper_summary.md]]</br> Look at the paper to understand correct input format and probe location. |
| medium | re-run the nece V-probe (+ random control) on the fixed HF model — all pre-fix probe results are invalid (garbage `inv_freq` before the persistent-buffer fix). |
| small | linear-probe dep baseline: cache the end-of-problem-description read position in `extract.py` (currently nece-only). |
| small | delete or re-save the old local checkpoint `final_models/gpt2-rope-igsm/final` (loads with garbage `inv_freq`; `load_lm` now rejects it loudly). |
| small | reconcile branches: `eval` still has the `inv_freq` bug; the fix lives on `qknorm` (4957741) and `probes`. Consider merging so no branch silently loads corrupted models. |
| small | `--uniform-difficulty` has no *minimum* op: it spreads over `1..max_op`, so an ops-20-23-only set cannot be expressed. Needs a `--op-range lo,hi` (the `op_spec` machinery in `src/data/igsm.py` already does this) for the release plan of ops 1-23. |