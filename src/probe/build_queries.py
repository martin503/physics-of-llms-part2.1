"""V-probe input construction: one iGSM problem seed -> the probe's labelled token sequences.

What distinguishes V-probing (paper section 4.1) from the linear probe in `probe.py` is that
the *queried parameter(s) go into the input*, so one hidden state can answer many questions.
This module owns those inputs -- the layouts, the read positions, and which questions get
asked -- and nothing about the model that consumes them (`vprobe.py`) or the training loop
(`vprobe_train.py`):

    nece(A):    [EOS] problem question [START] desc(A) [END]
    dep(A, B):  [EOS] problem          [START] desc(A) [MID] desc(B) [END]

The head always reads the final token, so the two tasks' differing read positions (paper
Figure 13) are expressed purely by what each layout keeps: `nece` ends at the end of the
*question*, `dep` drops the question and ends at the end of the *problem description*.

Deliberately torch-free: queries are plain lists of token ids, so the multiprocess generation
in `src.probe.data` spawns workers that never import torch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from src.data.igsm import EOS, ensure_igsm_submodule
from src.probe.labels import ProbeProblem, problem_desc_end_index, regenerate_problem

logger = logging.getLogger(__name__)

# Byte-fallback ids in the same never-in-ASCII dead zone as 222/223/224 (verified: 0 occurrences
# in 300 iGSM-med test problems). [MID] separates the two parameter descriptions in dep(A, B).
START, MID, END = 225, 227, 226

MAX_SEQ_LEN = 1024  # queries longer than this are dropped (iGSM-med problems are far shorter)


@dataclass
class ProbeQuery:
    """One probe query. `input_ids` ends with the injected parameter block: [START] desc(A) [END]
    for nece, or [START] desc(A) [MID] desc(B) [END] for dep. The head reads the final token.

    `param_a`/`param_b` index the queried A and B into `ProbeProblem.all_param` (B is -1 for
    nece queries), so predictions map back onto the dependency graph by regenerating the
    problem. `n_op` is the source problem's reasoning-step count, carried so a dataset's
    difficulty mix can be read off it directly. Training ignores all three.
    """

    input_ids: list[int]
    label: int
    group: int  # problem seed; split train/val by this, never by query
    param_a: int = -1
    param_b: int = -1
    n_op: int = -1


def queries_for_problem(
    seed: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    med_cfg: dict[str, Any] | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    dep_all_pairs: bool = False,
    max_queries: int | None = None,
) -> tuple[list[ProbeQuery], int]:
    """Build the V-probe queries for one regenerated problem; return `(queries, n_skipped)`.

    Targets share the candidate parameters (`labels.named_params`) and differ only in their
    builder (:data:`QUERY_BUILDERS`), which owns the input layout and read position.
    `dep_all_pairs` keeps every ordered off-diagonal pair instead of the balanced subsample
    -- the natural distribution used for testing; one-parameter targets ignore it.

    `max_queries` (per-problem cap) is accepted but not implemented yet; it raises.

    `(seed, split, med_cfg, target, max_queries)` fixes the result, including `dep`'s seeded
    negative sampling, so the multiprocess generation in `src.probe.data` stays byte-identical
    however the seeds are split across workers.
    """
    if target not in QUERY_BUILDERS:
        raise ValueError(f'target must be one of {TARGETS}, got {target!r}')
    if max_queries is not None:
        raise NotImplementedError(
            'per-problem query capping (paper Appendix E: <=10, uniform without replacement) is '
            'not implemented yet. Sample inside the QUERY_BUILDERS -- dep must subsample '
            'positives and negatives jointly to stay balanced -- not by truncating the result.'
        )
    pp = regenerate_problem(seed, split=split, med_cfg=med_cfg)
    return QUERY_BUILDERS[target](pp, seed, max_seq_len, dep_all_pairs)


def _nece_queries_for_problem(
    pp: ProbeProblem, seed: int, max_seq_len: int, all_pairs: bool = False
) -> tuple[list[ProbeQuery], int]:
    """`nece(A)` queries: one per candidate parameter, read at the end of the question.

    Reached through `queries_for_problem(target='nece')`, which supplies `pp`.

    The prefix is the full problem+question (everything before the [223] solution marker),
    preceded by EOS -- matching pretraining, where every problem is preceded by the previous
    one's EOS. The parameter description comes from iGSM's own `Problem.get_param` (the same
    "each X's Y" phrasing used in problem sentences) and is wrapped in [START]/[END].
    `all_pairs` is a pairwise-target option and does nothing for a one-parameter query.
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    prefix = [EOS, *pp.token_id[: pp.sol_bos_index]]  # [EOS] [222] problem+question
    queries: list[ProbeQuery] = []
    skipped = 0
    for p_idx, param in enumerate(pp.all_param):
        desc = tokenizer.encode(' ' + pp.problem.get_param(param))
        input_ids = [*prefix, START, *desc, END]
        if len(input_ids) > max_seq_len:
            skipped += 1  # guardrail: one pathological query must not set the batch's memory
            continue
        queries.append(
            ProbeQuery(
                input_ids=input_ids, label=int(pp.nece[p_idx]), group=seed, param_a=p_idx,
                n_op=pp.problem.n_op,
            )
        )
    return queries, skipped


def _dep_queries_for_problem(
    pp: ProbeProblem, seed: int, max_seq_len: int, all_pairs: bool = False
) -> tuple[list[ProbeQuery], int]:
    """`dep(A, B)` queries: balanced (A, B) pairs, read at the end of the problem description.

    Reached through `queries_for_problem(target='dep')`, which supplies `pp`.

    Input layout (paper Figure 13b / Appendix B): the question is dropped and the two parameter
    descriptions are injected in one [START]..[END] block separated by [MID]::

        [EOS] [222] <problem description> [START] desc(A) [MID] desc(B) [END]
                                                                        ^ read the head here

    Label = `dep(A, B)` = "does A (recursively) depend on B" = `pp.dep()[a, b]`.

    Pair selection (the `--target dep` universe): the full dep matrix is `n_param x n_param`, so we
    keep **all** positive pairs and sample an **equal** number of negatives (both off-diagonal),
    giving a per-problem-balanced dataset. Sampling is seeded from the problem `seed`, so offline
    generation stays deterministic. Unlike `nece`, no `--balance-classes` is needed:
    the classes are ~50/50 by construction (the paper's natural dep distribution is ~83% negative).

    `all_pairs=True` skips the subsampling and keeps every ordered off-diagonal pair -- the
    natural (heavily negative) distribution. Use it for *test* datasets, where predictions are
    mapped back onto the full dependency graph; training on it would need class re-weighting.
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    prefix = [EOS, *pp.token_id[: problem_desc_end_index(pp) + 1]]  # [EOS] [222] problem description
    dep = pp.dep()  # (n_param, n_param); dep[a, b] == 1 iff param a depends on param b
    n = len(pp.all_param)
    off_diag = ~np.eye(n, dtype=bool)  # a param never "depends on itself" (dep diagonal is 0)
    pos = np.argwhere((dep == 1) & off_diag)
    neg = np.argwhere((dep == 0) & off_diag)

    if not all_pairs:
        rng = np.random.default_rng(seed)  # per-problem: keeps offline generation reproducible
        n_neg = min(len(pos), len(neg))  # balance 1:1; keep all positives, subsample negatives
        if len(neg) > n_neg:
            neg = neg[rng.choice(len(neg), size=n_neg, replace=False)]

    # get_param is not free, so encode each parameter's description once (n_param is small)
    desc_ids = [tokenizer.encode(' ' + pp.problem.get_param(param)) for param in pp.all_param]

    queries: list[ProbeQuery] = []
    skipped = 0
    for pairs, label in ((pos, 1), (neg, 0)):
        for a, b in pairs:
            input_ids = [*prefix, START, *desc_ids[a], MID, *desc_ids[b], END]
            if len(input_ids) > max_seq_len:
                skipped += 1  # guardrail: one pathological query must not set the batch's memory
                continue
            queries.append(
                ProbeQuery(
                    input_ids=input_ids, label=label, group=seed,
                    param_a=int(a), param_b=int(b), n_op=pp.problem.n_op,
                )
            )
    return queries, skipped


# Probe tasks, each mapped to the builder turning one problem into its queries. Builders share
# the signature `(pp, seed, max_seq_len, all_pairs)`, so dispatch needs no per-target
# branching and each owns its own read position. A new probe is one entry plus its builder.
QUERY_BUILDERS: dict[str, Callable[[ProbeProblem, int, int, bool], tuple[list[ProbeQuery], int]]] = {
    'nece': _nece_queries_for_problem,
    'dep': _dep_queries_for_problem,
}
TARGETS = tuple(QUERY_BUILDERS)


def build_vprobe_queries(
    n_problems: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    seed_start: int = 0,
    dep_all_pairs: bool = False,
) -> list[ProbeQuery]:
    """Build V-probe queries over `n_problems` problems, in-process (see `queries_for_problem`).

    Convenient for small runs; for the query counts that actually avoid overfitting,
    generate offline with `python -m src.probe.run gen-data` and pass `--data` instead.
    """
    queries: list[ProbeQuery] = []
    skipped = 0
    for seed in tqdm(range(seed_start, seed_start + n_problems), desc='build queries', unit='prob'):
        problem_queries, n_skipped = queries_for_problem(
            seed, target=target, split=split, dep_all_pairs=dep_all_pairs
        )
        queries.extend(problem_queries)
        skipped += n_skipped
    if skipped:
        logger.info('dropped %d queries longer than MAX_SEQ_LEN=%d', skipped, MAX_SEQ_LEN)
    lens = [len(q.input_ids) for q in queries]
    logger.info(
        '%d queries, token length: median %d, max %d', len(queries), int(np.median(lens)), max(lens)
    )
    return queries
