
| size | feature |
|:----:|---------|
| small | generate regular continuations on probe prompts to justify need for training probes. |
| medium | (partially DONE) for dep(A, B): `test` + `report-dep` commands (interactive graph report, see probe_guide "Testing a trained probe"). nece is now covered for test + per-op eval by `scripts/probe/eval_fig7a.py` (Figure 7a row); only the interactive graph report remains dep-only. |
| medium | Test linear probes: same or similar queries (possibly without special tokens), train only linear probes. |
| small | move test files to mirror the source tree (probe tests into their own subdirectory). |
| small | a query over `MAX_SEQ_LEN` should drop its whole problem, not just itself (`build_queries.py:135` and `:207` `continue` per query). A partly-dropped problem still reaches the report and draws a dependency graph with edges silently missing. `gen-data` also throws the count away (`data.py:232` unpacks it as `_skipped`) and records nothing in `metadata.json`, so the loss would be invisible. Currently moot: 0 of 222k queries hit the cap even at op 23. |
| small | the "~85-90% negative" figure for the natural pair distribution is stale — measured `positive_frac` is 0.200, so ~80%. Still wrong in `evaluate.py:6`, `report_dep.py:27`, and `run.py:194` (the latter is `--help` text users see). `project_log.md:218` keeps it as a dated record. |
| small | `vprobe`'s online path (`build_vprobe_queries`, no `--data`) inherits `max_queries=10` with no CLI flag to override it. Add `--max-queries` to the `vprobe` command, or drop the online path. |
---
_postponed:_
|||
|:----:|---------|
| large | Add statistical analysis to gain more trustworthy results. (First for dep(A, B)) |
| large | add remaining probes from paper, documented in [[paper_summary.md]]</br> Look at the paper to understand correct input format and probe location. |
| medium | re-run the nece V-probe (+ random control) on the fixed HF model |
| small | linear-probe dep baseline: cache the end-of-problem-description read position in `extract.py` (currently nece-only). |
| small | `--uniform-difficulty` has no *minimum* op: it spreads over `1..max_op`, so an ops-20-23-only set cannot be expressed. Needs a `--op-range lo,hi` (the `op_spec` machinery in `src/data/igsm.py` already does this) for the release plan of ops 1-23. The eval side is now covered differently: pinned-op shards (`op=` in `generate_queries_to_dir`, `scripts/probe/gen_eval_shards.py`) put each op in its own dataset. |