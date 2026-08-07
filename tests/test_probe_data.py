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

from src.data.igsm import _shard_path
from src.probe.data import SHARD_GLOB, generate_queries_to_dir, load_vprobe_queries


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
