
| size | feature |
|:----:|---------|
| small | save full random model for probe to be more code independent? |
| medium | DONE (highest priority) successfully train v-probe on dep(A, B) task |
| small | generate regular continuations on probe prompts to justify need for training probes. |
| small | DECIDED AGAINST for probe testing (see project_log 2026-07-18: probes are supervised classification, inspect is built for generation evals; custom `report-dep` covers sample inspection). Inspect still makes sense later for behavioral evals of the LM itself (iGSM answer accuracy). |
| medium | DONE for dep(A, B): `test` + `report-dep` commands (interactive graph report, see probe_guide "Testing a trained probe"). Still open for nece. |
| medium | NEXT COMMIT match the paper's probing-query sampling (Appendix E): it draws parameters uniformly at random and keeps **at most 10** queries per problem and task, sampled without replacement. We keep every candidate parameter (`nece`, 12-72 per problem) and every positive pair plus matched negatives (`dep`). Queries within a problem are highly correlated, so a problem with 70 queries counts 70x toward the loss. Also affects query ordering: `dep` emits positives before negatives, and training length-buckets rather than shuffles queries, so a batch can be near-single-problem and near-single-label.</br>**Wired up, not implemented:** `--max-queries` reaches `queries_for_problem(max_queries=...)`, which raises. Remaining work: sample inside the `QUERY_BUILDERS` (dep must subsample positives and negatives *jointly* to stay balanced), plus batch composition below. The cap does **not** apply under `--dep-all-pairs` (enforced in `data.py`) -- the graph report needs the full matrix.</br>**Do not simply delete the length bucketing.** Measured on 300 dep problems (41.6k queries, lengths 46-548, within-problem spread ~7 tokens): distinct problems per batch of 8 = 1.41 today, 3.80 if the indices are shuffled *before* the sort, 7.84 with no bucketing -- but no bucketing costs 35% padding waste (3738 vs 2413 tokens/batch). The clustering comes from `sorted()` being stable: equal-length queries keep seed order, so one problem's queries stay adjacent. Shuffling before the sort is free (identical padding) and the cap compounds it (~14x fewer same-problem queries per length band). |
| medium | retrain the dep random control: the run started 2026-07-18 19:05 predates the seeded random-init fix, so its `probe.pt` cannot be re-paired with its transformer for test-time evaluation (training-time val metrics remain valid). |
| medium | NEXT COMMIT uniform difficulty. Probe *training* data: fully shuffled, uniform difficulty distribution. Probe *eval* data: uniform difficulty (shuffling irrelevant).</br>**CLI only:** `--uniform-difficulty` reaches `generate_queries_to_dir`, which raises; nothing else is wired. Each problem needs a target op threaded into iGSM's generator so it is *built* with that many steps -- not rejection sampling. Shard composition is irrelevant to training: `load_vprobe_queries` concatenates every shard before splitting or batching. |
| small | `find_showcase_seeds` has no *minimum* op: `--max-op 23` fills buckets 1-23 and keeps them all, so an ops-20-23-only showcase set cannot be expressed. Needs a `--min-op` (or an op range) alongside the release plan of ops 1-23 with 5-10 seeds each. |
| small | DONE probe training epochs: `--epochs` now defaults to 1 (one pass, no duplicate queries) in both `run.py` and `train_vprobe`. |
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
