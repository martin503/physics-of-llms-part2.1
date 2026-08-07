"""Tests for the V-probe's *pure* helpers (src.probe.vprobe, src.probe.vprobe_train).

``train_vprobe``/``VProbe`` need the frozen LM in the loop, so they're integration-shaped and
not unit-tested here. But three helpers are plain array/list logic with no model dependency,
and two of them encode correctness guarantees the code comments lean on heavily:

* ``_pad_batch``      -- the hidden state is read at ``end_index``; if that index is wrong, every
                         logit is read off the wrong token and the probe silently trains on noise.
* ``_split_by_group`` -- the same anti-leakage guarantee as ``probe.py``'s group split, restated
                         for the V-probe's query objects.
* ``_length_bucketed_batches`` -- a memory/speed optimisation; must still cover every query exactly.
"""

from __future__ import annotations

import numpy as np

from src.probe.build_queries import ProbeQuery
from src.probe.vprobe import EOS, _pad_batch
from src.probe.vprobe_train import _length_bucketed_batches, _split_by_group


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
    batches = _length_bucketed_batches(queries, indices, batch_size=8)
    flat = [q for batch in batches for q in batch]
    assert len(flat) == len(queries)  # every query present exactly once
    assert all(len(b) <= 8 for b in batches)  # size respected
    assert sum(len(b) for b in batches) == len(queries)


def test_length_bucketed_batches_group_similar_lengths():
    """Within a batch, lengths should be close together (that's the whole point)."""
    lengths = [1, 1, 1, 1, 50, 50, 50, 50]
    queries = [ProbeQuery(list(range(n)), label=0, group=0) for n in lengths]
    batches = _length_bucketed_batches(queries, np.arange(len(queries)), batch_size=4)
    # each length-4 batch should be internally homogeneous, not a 1-and-50 mix
    for batch in batches:
        lens = [len(q.input_ids) for q in batch]
        assert max(lens) - min(lens) == 0
