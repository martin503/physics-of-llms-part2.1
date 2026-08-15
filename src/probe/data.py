"""Offline V-probe query datasets: multiprocess generation into parquet shards + metadata.

Building V-probe queries regenerates every iGSM problem (graph, labels, tokens) (CPU-bound
work). This module generates queries offline with a process pool (mirroring `src.data.igsm`)
and writes them as parquet shards, so training can draw on tens of thousands of problems.

Each dataset directory contains:

    batch_000000.parquet ...   columns: input_ids (list<int64>), label, group,
                               param_a, param_b, n_op
    metadata.json              how the queries were made (target/split/seeds/config/commits)

Provenance note: V-probe queries are token sequences + graph labels from the iGSM generator
-- the only model-adjacent piece is the GPT-2 tokenizer, so the queries themselves are
model-independent. `model_path` is still recorded in metadata.json (when given) to tie a
dataset to the model it was generated for. Cached activations (`extract.py`) are the
genuinely model-tied artifact; those embed the model in their own pipeline.

Sharding is shared with `src.data.igsm` (same shard naming, same atomic-write-then-rename
protocol): shard `b` covers seeds `[seed_start + b*problems_per_shard, ...)`, which bounds
peak memory and lets readers stream. Generation is not resumable -- it is cheap enough to
redo. Each problem is fully determined by its seed and requested op, so any number of
workers produces the same queries in the same order.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.igsm import (
    IGSM_MED,
    IGSM_REPO_ROOT,
    REPO_ROOT,
    SHARD_PREFIX,
    SHARD_SUFFIX,
    _fine_unit,
    _shard_path,
    _unit_counts,
    ensure_igsm_submodule,
)

SHARD_GLOB = f'{SHARD_PREFIX}*{SHARD_SUFFIX}'

METADATA_NAME = 'metadata.json'
# Bumped whenever the queries built from a given seed change, so `metadata.json` records which
# rule produced a dataset (older queries cannot be told apart by their contents).
#   2: candidates restricted to parameters the problem text names (`labels.named_params`),
#      which also renumbers param_a/param_b; queries carry the source problem's `n_op`
#   3: metadata keys renamed row -> query (`rows_version`/`n_rows` -> `queries_version`/
#      `n_queries`); the queries themselves are unchanged from 2
#   4: per-problem query cap (`max_queries_per_problem`, `unbalanced`) and uniform difficulty
#      (`uniform_difficulty`); `identity` records each setting
QUERIES_VERSION = 4


def git_commit(cwd: Path) -> str | None:
    """Current HEAD commit of the repo at `cwd`, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _queries_for_seeds(
    seeds: list[int],
    ops: list[int | None],
    target: str,
    split: str,
    med_cfg: dict[str, Any],
    max_seq_len: int,
    dep_all_pairs: bool,
    max_queries: int | None = None,
    unbalanced: bool = False,
) -> tuple[list[list[int]], list[int], list[int], list[int], list[int], list[int], int]:
    """Worker: build the queries of one problem per seed, as plain columns (picklable).

    Returns the columns `(input_ids, labels, groups, param_a, param_b, n_op)`, one entry each
    per query, plus the number of over-long queries dropped. `ops[i]` asks iGSM to build
    `seeds[i]` with that many reasoning steps; None lets iGSM choose, which favours low counts.
    """
    from src.probe.build_queries import queries_for_problem

    input_ids: list[list[int]] = []
    labels: list[int] = []
    groups: list[int] = []
    param_a: list[int] = []
    param_b: list[int] = []
    n_op: list[int] = []
    skipped = 0
    for seed, op in zip(seeds, ops, strict=True):
        queries, n_skipped = queries_for_problem(
            seed, target=target, split=split, med_cfg=med_cfg, max_seq_len=max_seq_len,
            dep_all_pairs=dep_all_pairs, max_queries=max_queries, unbalanced=unbalanced, op=op,
        )
        skipped += n_skipped
        for q in queries:
            input_ids.append(q.input_ids)
            labels.append(q.label)
            groups.append(q.group)
            param_a.append(q.param_a)
            param_b.append(q.param_b)
            n_op.append(q.n_op)
    return input_ids, labels, groups, param_a, param_b, n_op, skipped


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write parquet via a hidden `.partial` temp file, then atomically rename (see igsm.py)."""
    tmp = path.with_name(f'.{path.name}.partial')
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)


def generate_queries_to_dir(
    out: Path | str,
    n_problems: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    seed_start: int = 0,
    workers: int = 8,
    problems_per_shard: int = 1_000,
    med_cfg: dict[str, Any] | None = None,
    model_path: str | None = None,
    overwrite: bool = False,
    dep_all_pairs: bool = False,
    max_queries: int | None = None,
    unbalanced: bool = False,
    uniform_difficulty: bool = False,
) -> dict[str, Any]:
    """Generate V-probe queries for `n_problems` seeds into `out`; return the metadata dict.

    Refuses if `out` already holds a finished dataset, unless `overwrite`.

    `max_queries` caps each problem's contribution (ignored under `dep_all_pairs`).
    `unbalanced` drops `dep`'s 1:1 class balance, sampling random pairs instead.
    `uniform_difficulty` cycles the requested step count over `1..med_cfg['max_op']` by problem
    index, so any *prefix* of the dataset is difficulty-balanced too (`report-dep` embeds the
    first N problems). All three go into `metadata.json`.
    """
    from src.probe.build_queries import MAX_SEQ_LEN

    out = Path(out)
    med_cfg = IGSM_MED if med_cfg is None else med_cfg
    if max_queries is not None and dep_all_pairs:
        # An all-pairs dataset exists to be mapped back onto the dependency graph; a capped
        # one would render graphs with most edges simply missing.
        max_queries = None
    identity = {
        'target': target,
        'split': split,
        'seed_start': seed_start,
        'n_problems': n_problems,
        'med_cfg': med_cfg,
        'max_seq_len': MAX_SEQ_LEN,
        'dep_all_pairs': dep_all_pairs,
        'max_queries_per_problem': max_queries,
        'unbalanced': unbalanced,
        'uniform_difficulty': uniform_difficulty,
        'queries_version': QUERIES_VERSION,
    }

    if overwrite and out.exists():
        shutil.rmtree(out)
    meta_path = out / METADATA_NAME
    # `metadata.json` is written only once generation finishes, so this refuses on complete
    # datasets; an interrupted run leaves shards but no metadata and is simply redone.
    if meta_path.exists():
        raise ValueError(
            f'{out} already holds a dataset; pass --overwrite or choose another --out'
        )
    out.mkdir(parents=True, exist_ok=True)
    for partial in out.glob('*.partial'):
        partial.unlink()

    num_shards = math.ceil(n_problems / problems_per_shard) if n_problems > 0 else 0
    all_seeds = list(range(seed_start, seed_start + n_problems))
    # Round-robin per problem, so every prefix of the dataset holds an
    # equal share of each op count (i.e. sorted by op: 1 2 3 1 2 3 ...)
    ops = list(range(1, med_cfg.get('max_op', 15) + 1))
    all_ops: list[int | None] = (
        [ops[i % len(ops)] for i in range(n_problems)] if uniform_difficulty
        else [None] * n_problems
    )
    pbar = tqdm(
        total=n_problems, desc=f'gen vprobe-{target} queries', unit='prob', dynamic_ncols=True
    )
    with ProcessPoolExecutor(max_workers=workers, initializer=ensure_igsm_submodule) as pool:
        for b in range(num_shards):
            shard_seeds = all_seeds[b * problems_per_shard : (b + 1) * problems_per_shard]
            shard_ops = all_ops[b * problems_per_shard : (b + 1) * problems_per_shard]
            # Contiguous seed sub-slices; each seed is independent. Many small units rather
            # than one per worker, so the pool rebalances iGSM's heavy-tailed generation
            # (see `_unit_counts`). Units are submitted and collected in order, so queries land
            # in seed order whatever the split.
            futures = []
            cursor = 0
            unit = _fine_unit(len(shard_seeds), workers)
            for count in _unit_counts(len(shard_seeds), unit):
                futures.append(
                    pool.submit(
                        _queries_for_seeds, shard_seeds[cursor : cursor + count],
                        shard_ops[cursor : cursor + count],
                        target, split, med_cfg, MAX_SEQ_LEN, dep_all_pairs, max_queries,
                        unbalanced,
                    )
                )
                cursor += count
            ids_col: list[list[int]] = []
            label_col: list[int] = []
            group_col: list[int] = []
            pa_col: list[int] = []
            pb_col: list[int] = []
            op_col: list[int] = []
            for f in futures:
                ids, labels, groups, params_a, params_b, n_ops, _skipped = f.result()
                ids_col.extend(ids)
                label_col.extend(labels)
                group_col.extend(groups)
                pa_col.extend(params_a)
                pb_col.extend(params_b)
                op_col.extend(n_ops)
            table = pa.table(
                {
                    'input_ids': ids_col, 'label': label_col, 'group': group_col,
                    'param_a': pa_col, 'param_b': pb_col, 'n_op': op_col,
                }
            )
            _write_table_atomic(table, _shard_path(out, b))
            pbar.update(len(shard_seeds))
    pbar.close()

    # dataset-wide stats read back from the shards themselves
    shards = sorted(out.glob(SHARD_GLOB))
    n_queries = 0
    n_positive = 0
    for shard in shards:
        labels = pq.read_table(str(shard), columns=['label']).column('label')
        n_queries += len(labels)
        n_positive += pc.sum(labels).as_py()

    metadata: dict[str, Any] = {
        'kind': 'vprobe_rows',
        **identity,
        'problems_per_shard': problems_per_shard,
        'n_shards': len(shards),
        'n_queries': n_queries,
        'positive_frac': n_positive / n_queries if n_queries else 0.0,
        'model_path': model_path,  # provenance only; queries are model-independent (see docstring)
        'tokenizer': 'gpt2 (iGSM; probe markers START=225 MID=227 END=226)',
        'repo_commit': git_commit(REPO_ROOT),
        'igsm_commit': git_commit(IGSM_REPO_ROOT),
        'command': ' '.join(sys.argv),
        'created': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(
        f'{n_problems} problems -> {n_queries} queries ({metadata["positive_frac"]:.3f} positive) '
        f'in {len(shards)} shards at {out}'
    )
    return metadata


def load_vprobe_queries(path: Path | str) -> tuple[list, dict[str, Any]]:
    """Load an offline query dataset; return `(queries, metadata)`.

    Query order is deterministic (sorted shards, insertion order within each), so a fixed
    seed reproduces the same train/val group split across runs.
    """
    from src.probe.build_queries import ProbeQuery

    path = Path(path)
    shards = sorted(path.glob(SHARD_GLOB))
    if not shards:
        raise FileNotFoundError(f'no parquet shards at {path} -- run `gen-data` first')
    meta_path = path / METADATA_NAME
    metadata = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}

    queries: list[ProbeQuery] = []
    for shard in shards:
        table = pq.read_table(str(shard))
        n = table.num_rows
        ids_col = table.column('input_ids').to_pylist()
        label_col = table.column('label').to_pylist()
        group_col = table.column('group').to_pylist()
        # provenance columns: datasets written before they existed load as -1
        names = table.column_names
        pa_col = table.column('param_a').to_pylist() if 'param_a' in names else [-1] * n
        pb_col = table.column('param_b').to_pylist() if 'param_b' in names else [-1] * n
        op_col = table.column('n_op').to_pylist() if 'n_op' in names else [-1] * n
        queries.extend(
            ProbeQuery(input_ids=ids, label=label, group=group, param_a=a, param_b=b, n_op=op)
            for ids, label, group, a, b, op in zip(
                ids_col, label_col, group_col, pa_col, pb_col, op_col, strict=True
            )
        )
    return queries, metadata
