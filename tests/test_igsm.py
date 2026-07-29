"""Tests for iGSM data generation (``src/data/igsm.py``) and difficulty recovery.

Three properties matter and would silently corrupt training if they broke:

* **no duplicate problems across workers** -- the only thing preventing it is the seed-stride
  scheme ``seed + b*stride + j`` in ``_generate_batch_in_pool``; two units sharing a seed emit
  identical iGSM problems. Tested deterministically with a fake pool (no iGSM) and for real.
* **packing** -- covered in ``test_pack.py``; here we check the end-to-end packed path on real
  data is well-formed.
* **difficulty spread** -- generating enough data must yield a couple of each op-count 1..15
  (regression for the old per-shard-op-lock bug). Difficulty is recoverable exactly as the number
  of ``;`` tokens (id 26) in the ``[223]..[224]`` region.

Fast tests (no ``slow`` mark) run in ``make fast-test``; the four real-generation tests are marked
``slow`` (``make test-slow``) and assume the iGSM submodule is checked out.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.data.igsm import (
    ANS_BOS,
    DEFAULT_CONTEXT_LENGTH,
    EOS,
    IGSM_MED,
    PROB_BOS,
    SOL_BOS,
    _count_shard_rows,
    _done_batches,
    _fine_unit,
    _generate_batch_in_pool,
    _normalize_op_spec,
    _shard_path,
    _unit_counts,
    _write_shard_atomic,
    generate_to_dir,
)
from visualizations.dataset_difficulty import (
    SEMI,
    _flatten_shard,
    expected_min_pmf,
    recover_op_counts,
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def synth_problem(op: int, uid: int) -> list[int]:
    """Synthetic iGSM-layout stream planting ``op`` as the difficulty.

    Layout ``[222] body [223] ;*op [224] ans [50256]``: the unique body token (``1000 + uid``)
    makes streams dedup-able and ``op`` semicolons (id 26) make ``recover_op_counts`` read the
    planted difficulty. Avoids the reserved ids elsewhere.
    """
    return [PROB_BOS, 1000 + uid, SOL_BOS, *[SEMI] * op, ANS_BOS, 7, EOS]


def _load_all(out: Path | str) -> list[list[int]]:
    """Read every ``input_ids`` row across all shards in ``out`` (raw or packed)."""
    streams: list[list[int]] = []
    for shard in sorted(Path(out).glob('batch_*.parquet')):
        streams.extend(pq.read_table(shard)['input_ids'].to_pylist())
    return streams


def _recover_ops(out: Path | str) -> Counter:
    """Recover per-problem op-counts from a packed dataset dir (flatten-first per shard)."""
    counts: Counter = Counter()
    for shard in sorted(Path(out).glob('batch_*.parquet')):
        flat, _ = _flatten_shard(str(shard))
        ops, _ = recover_op_counts(flat)
        counts.update(int(o) for o in ops)
    return counts


class _FakeFuture:
    def result(self) -> list[list[int]]:
        return []


class _FakePool:
    """Stand-in for ProcessPoolExecutor: records what ``_generate_chunk`` would receive.

    The real pool runs iGSM; this just captures ``(args, kwargs)`` of each ``submit`` so we can
    assert on the per-unit seeds and op assignments without generating anything.
    """

    def __init__(self) -> None:
        self.submits: list[tuple[tuple, dict]] = []

    def submit(self, fn, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        self.submits.append((args, kwargs))
        return _FakeFuture()


def _drive_batches(batch_size, workers, n_batches, seed, op_spec):  # noqa: ANN001
    """Mimic ``generate_to_dir``'s seed/op math across ``n_batches`` using a fake pool.

    Returns ``(stride, submits)`` where ``submits`` records each unit's submitted args/kwargs.
    """
    unit = _fine_unit(batch_size, workers)
    stride = len(_unit_counts(batch_size, unit))
    pool = _FakePool()
    bins = list(range(16))  # unused: the fake pool never calls _generate_chunk
    for b in range(n_batches):
        _generate_batch_in_pool(
            pool,
            batch_size,
            workers,
            seed + b * stride,
            bins,
            IGSM_MED,
            unit=unit,
            op_spec=op_spec,
            unit_start_index=b * stride,
        )
    return stride, pool.submits


# --------------------------------------------------------------------------- #
# unit / op-spec math (pure logic, iGSM-free)
# --------------------------------------------------------------------------- #


def test_unit_counts_split():
    assert _unit_counts(10, 3) == [3, 3, 3, 1]
    assert _unit_counts(9, 3) == [3, 3, 3]
    assert _unit_counts(0, 3) == []
    assert _unit_counts(5, 10) == [5]  # unit > num_problems -> one smaller chunk


def test_fine_unit_targets_and_caps():
    assert _fine_unit(0, 4) == 1  # degenerate -> 1
    assert _fine_unit(100, 1) == 100  # workers<=1 -> whole batch as one unit
    assert _fine_unit(100, 4) == 3  # target 32 units -> 100//32 = 3
    assert _fine_unit(10**9, 8) <= 256  # capped
    assert _fine_unit(1, 8) >= 1  # floored at 1


def test_normalize_op_spec_variants():
    assert _normalize_op_spec(None) is None
    assert _normalize_op_spec(5) == [5]
    assert _normalize_op_spec((1, 3)) == [1, 2, 3]
    assert _normalize_op_spec([2, 4, 6]) == [2, 4, 6]
    for bad in (True, False):
        with pytest.raises(TypeError):
            _normalize_op_spec(bad)
    with pytest.raises(ValueError):
        _normalize_op_spec([])
    with pytest.raises(ValueError):
        _normalize_op_spec([0, 5])  # non-positive
    with pytest.raises(ValueError):
        _normalize_op_spec([1.0])  # non-int
    with pytest.raises(ValueError):
        _normalize_op_spec((1, 2, 3))  # range must be (lo, hi)


# --------------------------------------------------------------------------- #
# shard I/O (parquet, atomic, resumable -- iGSM-free)
# --------------------------------------------------------------------------- #


def test_shard_path(tmp_path):
    assert _shard_path(tmp_path, 0).name == 'batch_000000.parquet'
    assert _shard_path(tmp_path, 42).name == 'batch_000042.parquet'


def test_shard_atomic_round_trip(tmp_path):
    rows = [[1, 2, 3], [4, 5, 6]]
    path = tmp_path / 'batch_000000.parquet'
    _write_shard_atomic(rows, path, require_uniform=True)
    assert path.is_file()
    assert not (tmp_path / '.batch_000000.parquet.partial').exists()  # temp cleaned up
    assert pq.read_table(path)['input_ids'].to_pylist() == rows


def test_done_batches_ignores_partials(tmp_path):
    """Resume correctness: a leftover ``.partial`` (interrupted write) never counts as done."""
    _write_shard_atomic([[1]], tmp_path / 'batch_000000.parquet', require_uniform=False)
    _write_shard_atomic([[3]], tmp_path / 'batch_000002.parquet', require_uniform=False)
    (tmp_path / '.batch_000001.parquet.partial').write_bytes(b'')  # simulated interrupted write
    assert _done_batches(tmp_path, num_batches=3) == {0, 2}


def test_count_shard_rows(tmp_path):
    _write_shard_atomic([[1], [2], [3]], tmp_path / 'batch_000000.parquet', require_uniform=False)
    _write_shard_atomic([[4], [5]], tmp_path / 'batch_000001.parquet', require_uniform=False)
    shards = sorted(tmp_path.glob('batch_*.parquet'))
    assert _count_shard_rows(shards) == 5


# --------------------------------------------------------------------------- #
# the seed-stride anti-duplicate rule (deterministic, iGSM-free)
# --------------------------------------------------------------------------- #


def test_per_unit_seeds_globally_unique():
    """Every pool unit, across all batches, gets a globally-unique seed (``seed + b*stride + j``).

    Two units sharing a seed would emit identical iGSM problems -- this is the load-bearing rule
    that makes multi-worker generation duplicate-free. Drives ``_generate_batch_in_pool`` with a
    fake pool so no iGSM runs.
    """
    seed, n_batches = 0, 3
    stride, submits = _drive_batches(
        batch_size=240, workers=4, n_batches=n_batches, seed=seed, op_spec=(1, 15)
    )
    seeds = [args[1] for args, _ in submits]
    assert len(seeds) == len(set(seeds))  # no two units share a seed
    expected = sorted(seed + i for i in range(n_batches * stride))
    assert sorted(seeds) == expected  # exactly seed + b*stride + j
    op_list = _normalize_op_spec((1, 15))
    for gi, (_, kw) in enumerate(submits):  # op assigned by round-robin
        assert kw['op'] == op_list[gi % len(op_list)]


def test_per_unit_op_none_without_spec():
    """With no op_spec, every unit is assigned op=None (natural per-problem op draw)."""
    _, submits = _drive_batches(240, 4, 2, 0, op_spec=None)
    assert submits and all(kw['op'] is None for _, kw in submits)


# --------------------------------------------------------------------------- #
# generate_to_dir plumbing via an injected generator (needs iGSM bins only; fast)
# --------------------------------------------------------------------------- #


def test_generate_to_dir_batch_seeds_and_no_dupes(tmp_path, monkeypatch):
    """The batch base-seed spacing (``seed + b*stride``) is distinct, and the written raw streams
    carry no duplicates end-to-end. Uses the injectable ``problem_generator`` seam. ``get_bins`` is
    stubbed (its result is unused when ``problem_generator`` is set) so this needs no iGSM."""
    monkeypatch.setattr('src.data.igsm.get_bins', lambda split: [0])
    seen: list[int] = []

    def fake(num_problems, seed, **_kw):  # noqa: ANN001, ANN002, ANN003
        seen.append(seed)
        # per-problem-unique streams: batch seed in the body + per-problem index
        return [
            [PROB_BOS, 1000 + (seed % 40000), i, SOL_BOS, ANS_BOS, EOS]
            for i in range(num_problems)
        ]

    n, batch_size, workers, seed = 30, 10, 3, 0
    generate_to_dir(
        tmp_path,
        n,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        pack=False,
        problem_generator=fake,
    )
    stride = len(_unit_counts(batch_size, _fine_unit(batch_size, workers)))
    assert seen == [seed + b * stride for b in range(math.ceil(n / batch_size))]
    streams = _load_all(tmp_path)
    assert len(streams) == n
    assert len(set(map(tuple, streams))) == n  # no duplicate problems


def test_generate_to_dir_resumes(tmp_path, monkeypatch):
    """A second run with completed shards resumes and regenerates nothing. ``get_bins`` is stubbed
    so this needs no iGSM."""
    monkeypatch.setattr('src.data.igsm.get_bins', lambda split: [0])
    calls = {'n': 0}

    def fake(num_problems, seed, **_kw):  # noqa: ANN001, ANN002, ANN003
        calls['n'] += 1
        return [[PROB_BOS, seed % 40000, i, SOL_BOS, ANS_BOS, EOS] for i in range(num_problems)]

    kw = {'batch_size': 10, 'workers': 2, 'seed': 0, 'pack': False, 'problem_generator': fake}
    generate_to_dir(tmp_path, 20, **kw)
    assert calls['n'] == 2  # 2 batches generated
    calls['n'] = 0
    generate_to_dir(tmp_path, 20, **kw)
    assert calls['n'] == 0  # both shards done -> resumed


# --------------------------------------------------------------------------- #
# difficulty recovery instrument (synthetic, fast)
# --------------------------------------------------------------------------- #


def test_recover_op_counts_synthetic():
    """``recover_op_counts`` reads back exactly the op-counts planted in synthetic problems
    (it's the measurement instrument the distribution tests rely on)."""
    problems = [synth_problem(op, uid) for op, uid in [(1, 0), (5, 1), (15, 2), (3, 3)]]
    flat = np.array([t for p in problems for t in p], dtype=np.int64)
    ops, _ = recover_op_counts(flat)
    assert sorted(int(o) for o in ops) == [1, 3, 5, 15]


def test_expected_min_pmf_sanity():
    """The paper target pmf is a valid, strictly-decreasing distribution over op 1..15."""
    pmf = expected_min_pmf(15)
    assert pmf.shape == (15,)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert all(pmf[i] > pmf[i + 1] for i in range(14))
    assert pmf[0] > pmf[-1]


# --------------------------------------------------------------------------- #
# slow: real iGSM generation (assumes the submodule is checked out)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_no_duplicates_and_deterministic(tmp_path):
    """Real generation across multiple workers/batches produces no duplicate problems, is
    reproducible under the same config, and changes with the seed."""
    n, batch_size, workers = 48, 16, 4
    generate_to_dir(
        tmp_path / 's0a', n, batch_size=batch_size, workers=workers, seed=0, pack=False
    )
    streams0a = _load_all(tmp_path / 's0a')
    assert len(streams0a) == n
    assert len(set(map(tuple, streams0a))) == n  # no duplicates

    generate_to_dir(
        tmp_path / 's0b', n, batch_size=batch_size, workers=workers, seed=0, pack=False
    )
    assert sorted(map(tuple, streams0a)) == sorted(
        map(tuple, _load_all(tmp_path / 's0b'))
    )  # deterministic

    generate_to_dir(tmp_path / 's1', n, batch_size=batch_size, workers=workers, seed=1, pack=False)
    assert sorted(map(tuple, _load_all(tmp_path / 's1'))) != sorted(
        map(tuple, streams0a)
    )  # seed-sensitive


@pytest.mark.slow
def test_real_packed_windows_well_formed(tmp_path):
    """The end-to-end packed path on real data yields exactly-context-length, in-vocab windows
    from which at least one well-formed problem is recoverable."""
    generate_to_dir(
        tmp_path, 48, workers=4, seed=0, pack=True, context_length=DEFAULT_CONTEXT_LENGTH
    )
    wins = _load_all(tmp_path)
    assert wins
    assert all(len(w) == DEFAULT_CONTEXT_LENGTH for w in wins)
    assert all(0 <= t < 50257 for w in wins for t in w)
    ops = _recover_ops(tmp_path)
    assert sum(ops.values()) >= 1
    assert all(1 <= o <= 15 for o in ops)


@pytest.mark.slow
def test_distribution_forced_uniform_ops(tmp_path):
    """Pinning op_spec to 1..15 yields a couple of each difficulty (regression for the
    per-shard-op-lock bug where a whole shard collapsed to one op)."""
    n = 300
    generate_to_dir(
        tmp_path, n, batch_size=n, workers=4, seed=0, pack=True, op_spec=list(range(1, 16))
    )
    ops = _recover_ops(tmp_path)
    assert set(ops) <= set(range(1, 16))  # pinning honored
    for o in range(1, 16):
        assert ops[o] >= 2, f'op {o} too rare: {ops[o]}'  # a couple of each difficulty


@pytest.mark.slow
def test_distribution_natural_all_ops_present(tmp_path):
    """Under the natural distribution (op_spec=None), generating enough data yields every op
    1..15 (op 15 is ~1/225). The slowest test in the suite."""
    n = 3000
    generate_to_dir(tmp_path, n, batch_size=n, workers=8, seed=0, pack=True, op_spec=None)
    ops = _recover_ops(tmp_path)
    for o in range(1, 16):
        assert ops[o] >= 2, f'op {o} too rare: {ops[o]}'
    assert ops[1] > ops[15]  # shape: easiest >> hardest
