
| size | feature |
|:----:|---------|
| small | save full random model for probe to be more code independent? |
| medium | DONE (highest priority) successfully train v-probe on dep(A, B) task |
| small | generate regular continuations on probe prompts to justify need for training probes. |
| small | DECIDED AGAINST for probe testing (see project_log 2026-07-18: probes are supervised classification, inspect is built for generation evals; custom `report-dep` covers sample inspection). Inspect still makes sense later for behavioral evals of the LM itself (iGSM answer accuracy). |
| medium | DONE for dep(A, B): `test` + `report-dep` commands (interactive graph report, see probe_guide "Testing a trained probe"). Still open for nece. |
| medium | NEXT COMMIT match the paper's probing-query sampling (Appendix E): it draws parameters uniformly at random and keeps **at most 10** queries per problem and task, sampled without replacement. We keep every candidate parameter (`nece`, 12-72 per problem) and every positive pair plus matched negatives (`dep`). Queries within a problem are highly correlated, so a problem with 70 queries counts 70x toward the loss. Also affects query ordering: `dep` emits positives before negatives, and training length-buckets rather than shuffles queries, so a batch can be near-single-problem and near-single-label. |
| medium | retrain the dep random control: the run started 2026-07-18 19:05 predates the seeded random-init fix, so its `probe.pt` cannot be re-paired with its transformer for test-time evaluation (training-time val metrics remain valid). |
| medium | NEXT COMMIT uniform difficulty. Probe *training* data: fully shuffled, uniform difficulty distribution. Probe *eval* data: uniform difficulty (shuffling irrelevant). Likely route: thread `op`/`op_spec` into `regenerate_problem` instead of `find_showcase_seeds`' rejection sampling. Also fixes the `--seeds-file` path emitting seeds grouped by op (`data.py:178`), which stratifies shards by difficulty. |
| small | NEXT COMMIT probe training epochs: train for one epoch with no duplicate queries (`--epochs` currently defaults to 3). |
| medium | remove the linear-probe pipeline: `probe.py`, `extract.py`, the `extract`/`train` CLI commands, `tests/test_probe.py`, and the probe_guide sections. It was added on a misunderstanding of what the probes are for; the V-probe is the real thing. Removing it also retires the last legitimate uses of "row" in a probe context (see d22756c). |
| small | move test files to mirror the source tree (probe tests into their own subdirectory). |
| small | DONE (d22756c) rename the probe's `row` concept to `query`: `VProbeRow` -> `ProbeQuery`, and the metadata keys `rows_version`/`n_rows` -> `queries_version`/`n_queries` (`QUERIES_VERSION` = 3). |
| small | `.gitmodules` points the iGSM submodule at `org-16943930@github.com:facebookresearch/iGSM.git`, an SSH alias that does not resolve, so `git submodule update --init iGSM` always fails. Use the public HTTPS URL. Locally (2026-08-07) the checkout was repaired by hand: a clone had landed in `iGSM/iGSM/` instead of `iGSM/`, so `iGSM/` is now a plain clone at the pinned `a1ed1d0`, not a registered submodule. |
| small | probe_guide.md's module table still lists `queries.py`; the file has been `build_queries.py` since 1053143. |
| small | `tests/test_gpt.py` DDP tests fail on this Windows box: `makeDeviceForHostname(): unsupported gloo device` (torch cannot bind the hostname). Skip them on Windows, or mark them as requiring a working gloo backend. |
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
