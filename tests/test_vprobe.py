"""Tests for the V-probe's *pure* helpers (src.probe.vprobe, src.probe.vprobe_train).

``train_vprobe``/``VProbe`` need the frozen LM in the loop, so they're integration-shaped and
not unit-tested here. But three helpers are plain array/list logic with no model dependency,
and two of them encode correctness guarantees the code comments lean on heavily:

* ``_pad_batch``      -- the hidden state is read at ``end_index``; if that index is wrong, every
                         logit is read off the wrong token and the probe silently trains on noise.
* ``_split_by_group`` -- the same anti-leakage guarantee as ``probe.py``'s group split, restated
                         for the V-probe's query objects.
* ``_length_bucketed_batches`` -- a memory/speed optimisation; must still cover every query
                         exactly, and must not fill a batch from one problem.

The per-problem query cap (``build_queries``) is checked against real iGSM problems -- the
candidate set it samples from is the dependency graph -- so those tests are marked ``slow`` and
kept to a handful of low-``max_op`` problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.igsm import IGSM_MED
from src.probe.build_queries import (
    _STREAM_DEP,
    _STREAM_DEP_UNBALANCED,
    _STREAM_NECE,
    ProbeQuery,
    _sample,
    queries_for_problem,
)
from src.probe.labels import regenerate_problem
from src.probe.vprobe import EOS, _pad_batch
from src.probe.vprobe_train import _length_bucketed_batches, _split_by_group

MED_SMALL = {**IGSM_MED, 'max_op': 4}  # small graphs: the sampling rules are what is under test


def _dep_queries(seed, **kwargs):
    queries, _skipped = queries_for_problem(
        seed, target='dep', split='test', med_cfg=MED_SMALL, **kwargs
    )
    return queries


def test_pad_batch_end_index_points_at_last_real_token():
    """``end_index`` must be the [END] token (the last real id), never a pad position.

    The probe reads ``last_hidden_state[end_index]``; off-by-one here reads a padded EOS.
    """
    queries = [ProbeQuery([1, 2, 3, 4], label=1, group=0), ProbeQuery([5, 6], label=0, group=0)]
    ids, mask, end, labels = _pad_batch(queries, 'cpu')
    assert end.tolist() == [3, 1]  # n-1 for each query's own length
    # the token at each end_index is the query's actual last token, not the EOS pad
    assert ids[0, end[0]].item() == 4
    assert ids[1, end[1]].item() == 6


def test_pad_batch_pads_shorter_rows_with_eos_and_zeroes_mask():
    """Short queries are right-padded with EOS and their pad positions masked out."""
    queries = [ProbeQuery([1, 2, 3], label=1, group=0), ProbeQuery([9], label=0, group=1)]
    ids, mask, _end, _labels = _pad_batch(queries, 'cpu')
    assert ids.shape == (2, 3)  # padded to the longest query
    assert ids[1].tolist() == [9, EOS, EOS]
    assert mask[1].tolist() == [1, 0, 0]  # only the real token is attended
    assert mask[0].tolist() == [1, 1, 1]


def test_pad_batch_preserves_labels():
    queries = [ProbeQuery([1], label=1, group=0), ProbeQuery([2, 3], label=0, group=0)]
    _ids, _mask, _end, labels = _pad_batch(queries, 'cpu')
    assert labels.tolist() == [1, 0]


def test_split_by_group_puts_no_group_in_both_splits():
    """No problem may appear in both train and val -- the anti-leakage invariant."""
    queries = [ProbeQuery([1], label=i % 2, group=i // 5) for i in range(50)]  # 10 groups, 5 queries each
    train, val = _split_by_group(queries, val_frac=0.3, seed=0)
    train_groups = {q.group for q in train}
    val_groups = {q.group for q in val}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups | val_groups == set(range(10))  # every group landed somewhere
    assert len(train) + len(val) == len(queries)  # no query dropped or duplicated


def test_split_by_group_always_keeps_at_least_one_val_group():
    """``max(1, ...)`` guarantees a non-empty val split even at tiny val_frac."""
    queries = [ProbeQuery([1], label=0, group=g) for g in range(4)]
    _train, val = _split_by_group(queries, val_frac=0.01, seed=0)
    assert len(val) >= 1


def test_length_bucketed_batches_cover_every_row_once():
    """Bucketing reorders queries for memory locality but must not drop or duplicate any."""
    queries = [ProbeQuery(list(range(np.random.default_rng(i).integers(1, 40))), label=0, group=0)
               for i in range(37)]
    indices = np.arange(len(queries))
    rng = np.random.default_rng(0)
    batches = _length_bucketed_batches(queries, indices, batch_size=8, rng=rng)
    flat = [q for batch in batches for q in batch]
    assert len(flat) == len(queries)  # every query present exactly once
    assert all(len(b) <= 8 for b in batches)  # size respected
    assert sum(len(b) for b in batches) == len(queries)


def test_length_bucketed_batches_group_similar_lengths():
    """Within a batch, lengths should be close together (that's the whole point).

    Four loosely separated length bands, none internally uniform: a batch may not span two
    bands, but equal lengths within a batch are not what is being asserted.
    """
    lengths = [10, 12, 13, 11, 20, 23, 21, 22, 30, 33, 31, 29, 40, 42, 41, 39]
    queries = [ProbeQuery(list(range(n)), label=0, group=0) for n in lengths]
    batches = _length_bucketed_batches(
        queries, np.arange(len(queries)), batch_size=4, rng=np.random.default_rng(0)
    )
    for batch in batches:
        lens = [len(q.input_ids) for q in batch]
        assert max(lens) - min(lens) <= 5  # one band; the dataset as a whole spans 33


def test_length_bucketed_batches_mix_problems():
    """A batch must not be one problem's queries: a stable length-sort keeps them adjacent.

    Queries from the same problem differ by a few tokens, so sorting by length alone groups them.
    Shuffling first breaks the ties randomly instead.

    The fixture is sized like real data -- `--max-queries` per problem, many problems -- because
    the clumping scales with how many same-length queries one problem holds. At 40 problems the
    two orderings score alike and the assertion passes either way.
    """
    rng = np.random.default_rng(0)
    queries = [  # 300 problems x the 10-query cap, each problem's lengths spread over ~8 tokens
        ProbeQuery(list(range(base + int(rng.integers(0, 8)))), label=0, group=problem)
        for problem, base in enumerate(rng.integers(46, 100, size=300))
        for _ in range(10)
    ]
    indices = np.arange(len(queries))
    batches = _length_bucketed_batches(
        queries, indices, batch_size=8, rng=np.random.default_rng(1)
    )
    distinct = np.mean([len({q.group for q in batch}) for batch in batches])
    assert distinct > 6.5  # ~7.4 here; a stable sort over the same fixture gives ~5.1


def test_sampling_streams_are_independent():
    """One problem seed must not hand the same draw to every target and mode."""
    pool = np.arange(50)
    draws = {
        stream: tuple(_sample(pool, 8, np.random.default_rng([7, stream])).tolist())
        for stream in (_STREAM_NECE, _STREAM_DEP, _STREAM_DEP_UNBALANCED)
    }
    assert len(set(draws.values())) == 3


@pytest.mark.slow
def test_max_queries_caps_each_problem():
    """Both targets respect the cap; ``max_queries=None`` keeps the whole candidate set."""
    for seed in range(3):
        n_param = len(regenerate_problem(seed, split='test', med_cfg=MED_SMALL).all_param)
        capped, _ = queries_for_problem(
            seed, target='nece', split='test', med_cfg=MED_SMALL, max_queries=6
        )
        uncapped, _ = queries_for_problem(seed, target='nece', split='test', med_cfg=MED_SMALL)
        assert len(capped) == min(6, n_param)
        assert len(uncapped) == n_param  # one query per candidate parameter
        assert len(_dep_queries(seed, max_queries=6)) == 6
        assert len(_dep_queries(seed)) > 6


@pytest.mark.slow
def test_capped_queries_are_deterministic():
    """A problem's queries depend on its seed alone, so worker chunking cannot change them."""
    seeds = [0, 1, 2]

    def build(order):
        return {s: [(q.input_ids, q.label) for q in _dep_queries(s, max_queries=6)] for s in order}

    assert build(seeds) == build(reversed(seeds))


@pytest.mark.slow
def test_dep_balanced_stays_one_to_one_under_the_cap():
    """Half the budget per class -- and when positives run out, the budget goes unfilled."""
    for seed in range(3):
        queries = _dep_queries(seed, max_queries=10)
        assert len(queries) == 10
        assert sum(q.label for q in queries) == 5
    poor = _dep_queries(3, max_queries=1000)  # far more budget than the problem has positives
    assert len(poor) < 1000
    assert sum(q.label for q in poor) * 2 == len(poor)


@pytest.mark.slow
def test_dep_unbalanced_draws_from_every_pair():
    """The paper's rule keeps the natural positive rate instead of forcing 1:1."""
    labels = [
        q.label for seed in range(20) for q in _dep_queries(seed, max_queries=10, unbalanced=True)
    ]
    assert len(labels) == 200
    assert 0.05 < np.mean(labels) < 0.4


@pytest.mark.slow
def test_dep_all_pairs_ignores_the_cap():
    """The graph report needs the full matrix, so ``dep_all_pairs`` outranks ``max_queries``."""
    n_param = len(regenerate_problem(0, split='test', med_cfg=MED_SMALL).all_param)
    queries = _dep_queries(0, dep_all_pairs=True, max_queries=2)
    assert {(q.param_a, q.param_b) for q in queries} == {
        (a, b) for a in range(n_param) for b in range(n_param) if a != b
    }
