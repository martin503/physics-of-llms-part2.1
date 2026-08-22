"""Tests for offline V-probe dataset generation (``src.probe.data``).

``run.py`` and ``evaluate.py`` import this module lazily, from inside function bodies, so
nothing else in the suite loads it: a name that has moved in ``src.data.igsm`` stays
invisible until someone runs ``gen-data`` for real. The fast tests here make that loud.

The slow tests cover what an import check cannot -- that a pooled run writes shards the
readers can read, and that splitting the seeds differently across workers yields the same
queries.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.data.igsm import EOS, _shard_path
from src.probe.data import SHARD_GLOB, generate_queries_to_dir, load_vprobe_queries
from src.probe.labels import regenerate_problem

# Small enough that op pinning stays cheap: 4 op buckets instead of the default 15.
MED_4 = dict(max_op=4, max_edge=20, perm_level=5, detail_level=0)


def test_module_imports():
    """``src.probe.data`` must load on its own -- the check no other test performs.

    Deliberately trivial: it fails when a name this module imports from ``src.data.igsm``
    moves, which nothing else catches short of running generation.
    """
    import src.probe.data  # noqa: F401


def test_shard_glob_matches_shard_path():
    """The pattern used to *find* shards must match the names used to *write* them.

    Both derive from ``SHARD_PREFIX``/``SHARD_SUFFIX``. Should they diverge, generation
    writes shards its own stats pass cannot see -- zero rows, no error.
    """
    assert fnmatch.fnmatch(_shard_path(Path('d'), 0).name, SHARD_GLOB)


# --------------------------------------------------------------------------- #
# Real generation runs
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize('target', ['nece', 'dep'])
def test_generation_writes_shards_the_readers_can_read(tmp_path, target):
    """End-to-end: a pooled run writes shards, metadata, and queries that load back.

    Exercises every piece ``src.data.igsm`` supplies (shard naming, the work split) plus
    the process pool, none of which an import check touches.
    """
    out = tmp_path / target
    meta = generate_queries_to_dir(out, 3, target=target, split='test', workers=2,
                                    problems_per_shard=2)

    assert meta['n_shards'] == 2  # 3 problems at 2 per shard
    assert sorted(out.glob(SHARD_GLOB)) == [_shard_path(out, 0), _shard_path(out, 1)]
    assert meta == json.loads((out / 'metadata.json').read_text(encoding='utf-8'))

    on_disk = sum(pq.read_table(str(p)).num_rows for p in sorted(out.glob(SHARD_GLOB)))
    queries, loaded_meta = load_vprobe_queries(out)
    assert meta['n_queries'] == on_disk == len(queries) > 0
    assert loaded_meta == meta
    assert {q.group for q in queries} == {0, 1, 2}  # one group per seed, none lost


@pytest.mark.slow
def test_each_shard_holds_mixed_difficulties(tmp_path):
    """No shard may hold a single difficulty, so reading one is not reading one op count.

    Sequential seeds draw `n_op` independently per problem, so the mix comes for free --
    what this pins is that nothing downstream groups seeds by difficulty on the way into a
    shard. Difficulty is read from the dataset's own `n_op` column rather than regenerated.
    """
    out = tmp_path / 'ds'
    meta = generate_queries_to_dir(out, 8, target='nece', split='test', workers=4,
                                    problems_per_shard=4)

    for b in range(meta['n_shards']):
        table = pq.read_table(str(_shard_path(out, b)), columns=['group', 'n_op'])
        by_seed = dict(zip(table.column('group').to_pylist(),
                           table.column('n_op').to_pylist(), strict=True))
        assert min(by_seed.values()) >= 1  # a real op count, not the -1 default
        assert len(set(by_seed.values())) > 1, f'shard {b} holds one difficulty: {by_seed}'


@pytest.mark.slow
def test_uniform_difficulty_balances_every_prefix(tmp_path):
    """Ops cycle per problem, so any *prefix* of the dataset is difficulty-balanced.

    `report-dep` embeds the first N problems, so a merely dataset-wide equal share is not
    enough. Op counts are read from the `n_op` column, which therefore also pins that the
    recorded step count equals the op the problem was built at.
    """
    out = tmp_path / 'uniform'
    generate_queries_to_dir(out, 8, target='nece', split='test', workers=2,
                            problems_per_shard=3, med_cfg=MED_4, max_queries=None,
                            uniform_difficulty=True)

    queries, meta = load_vprobe_queries(out)
    assert meta['uniform_difficulty'] is True
    by_seed: dict[int, int] = {}
    for q in queries:
        by_seed.setdefault(q.group, q.n_op)
        assert q.n_op == by_seed[q.group]  # one difficulty per problem
    ops = list(by_seed.values())  # dataset order
    assert ops == [1, 2, 3, 4, 1, 2, 3, 4]
    for k in range(1, len(ops) + 1):
        counts = [ops[:k].count(op) for op in (1, 2, 3, 4)]
        assert max(counts) - min(counts) <= 1, f'prefix of {k} problems is skewed: {ops[:k]}'


@pytest.mark.slow
def test_recorded_op_reproduces_the_dataset_problem(tmp_path):
    """Regenerating with the recorded op gives back the problem the queries were built from.

    `report_dep` redraws each problem from `(seed, op)` to label its graph; without the op it
    gets a *different* problem and the graph stops matching the predictions.
    """
    out = tmp_path / 'uniform'
    generate_queries_to_dir(out, 4, target='nece', split='test', workers=2,
                            problems_per_shard=4, med_cfg=MED_4, max_queries=None,
                            uniform_difficulty=True)

    queries, _meta = load_vprobe_queries(out)
    first: dict[int, object] = {}
    for q in queries:
        first.setdefault(q.group, q)
    differs = []
    for seed, q in first.items():
        pinned = regenerate_problem(seed, split='test', med_cfg=MED_4, op=q.n_op)
        prefix = [EOS, *pinned.token_id[: pinned.sol_bos_index]]  # nece layout, before [START]
        assert q.input_ids[: len(prefix)] == prefix, f'seed {seed} regenerates differently'
        unpinned = regenerate_problem(seed, split='test', med_cfg=MED_4)
        differs.append(unpinned.token_id != pinned.token_id)
    assert any(differs), 'op pinning changed no problem -- the test cannot detect a missing op'


@pytest.mark.slow
@pytest.mark.parametrize('target, unbalanced', [('nece', False), ('dep', True)])
def test_pinned_op_builds_one_difficulty(tmp_path, target, unbalanced):
    """`op` pins every problem to that step count -- the per-op eval shards of Figure 7a.

    A pinned shard must hold exactly one difficulty (the shard *is* one figure column), record
    its `op` in metadata so downstream tools can replay the request, and -- on dep with the
    unbalanced sampling the eval shards use -- keep the natural class mix instead of the 1:1
    balance of training data.
    """
    out = tmp_path / 'pinned'
    meta = generate_queries_to_dir(out, 4, target=target, split='test', workers=2,
                                   problems_per_shard=4, med_cfg=MED_4, max_queries=10,
                                   unbalanced=unbalanced, op=3)

    queries, loaded_meta = load_vprobe_queries(out)
    assert meta['op'] == 3
    assert meta['uniform_difficulty'] is False
    assert loaded_meta['op'] == 3
    assert {q.n_op for q in queries} == {3}
    if target == 'dep':
        rate = sum(q.label for q in queries) / len(queries)
        assert 0.0 < rate < 0.45, f'unbalanced dep should stay near the natural mix, got {rate}'


def test_pinned_op_and_uniform_difficulty_are_exclusive(tmp_path):
    """The two difficulty controls answer the same question; passing both must fail loudly."""
    with pytest.raises(ValueError, match='not both'):
        generate_queries_to_dir(tmp_path / 'x', 1, uniform_difficulty=True, op=3, workers=1)


def test_pinned_op_outside_max_op_is_refused(tmp_path):
    """iGSM would silently raise its own cap; metadata must not claim a max_op problems exceed."""
    with pytest.raises(ValueError, match='max_op'):
        generate_queries_to_dir(tmp_path / 'x', 1, med_cfg=MED_4, op=5, workers=1)


@pytest.mark.slow
def test_worker_count_does_not_change_rows(tmp_path):
    """Splitting the same seeds across more workers must produce identical queries.

    Work distribution may decide *who* builds a query, never *which* queries exist or in what
    order -- the property the whole shard/unit design rests on, and the one that would
    break silently. Run on ``dep`` because it is the only target whose queries depend on
    randomness (seeded negative sampling); ``nece`` is deterministic regardless.
    """
    kwargs = dict(target='dep', split='test', problems_per_shard=2)
    generate_queries_to_dir(tmp_path / 'w1', 3, workers=1, **kwargs)
    generate_queries_to_dir(tmp_path / 'w3', 3, workers=3, **kwargs)

    serial, _ = load_vprobe_queries(tmp_path / 'w1')
    parallel, _ = load_vprobe_queries(tmp_path / 'w3')
    assert [(q.input_ids, q.label, q.group, q.param_a, q.param_b) for q in serial] == [
        (q.input_ids, q.label, q.group, q.param_a, q.param_b) for q in parallel
    ]
