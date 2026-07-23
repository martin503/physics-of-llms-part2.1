"""Offline V-probe row datasets: multiprocess generation into parquet shards + metadata.

Building V-probe rows regenerates every iGSM problem (graph, labels, tokens) -- CPU-bound
work that caps online runs at a few hundred problems and makes the probe overfit. This
module generates rows *offline* with a process pool (mirroring `src.data.igsm`) and writes
them as resumable parquet shards, so training can draw on tens of thousands of problems.

Each dataset directory contains:

    batch_000000.parquet ...   columns: input_ids (list<int64>), label, group
    metadata.json              how the rows were made (target/split/seeds/config/commits)

Provenance note: V-probe rows are token sequences + graph labels from the iGSM generator
-- the only model-adjacent piece is the GPT-2 tokenizer, so the rows themselves are
model-independent. `model_path` is still recorded in metadata.json (when given) to tie a
dataset to the model it was generated *for*. Cached activations (`extract.py`) are the
genuinely model-tied artifact; those embed the model in their own pipeline.

Sharding/resume machinery is shared with `src.data.igsm` (same shard naming, same
atomic-write-then-rename protocol), so an interrupted generation resumes from the last
complete shard. Shard `b` covers seeds `[seed_start + b*problems_per_shard, ...)`, and
each problem is fully determined by its seed -- so parallel and resumed runs produce
byte-identical datasets.
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
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from src.data.igsm import (
    IGSM_MED,
    IGSM_REPO_ROOT,
    REPO_ROOT,
    _done_batches,
    _shard_path,
    _worker_counts,
    ensure_igsm_submodule,
)

METADATA_NAME = 'metadata.json'
# Fields that make two datasets different data (not just differently-run generation).
# Datasets written before a key existed compare against its default (`_IDENTITY_DEFAULTS`).
_IDENTITY_KEYS = (
    'target', 'split', 'seed_start', 'n_problems', 'med_cfg', 'max_seq_len', 'dep_all_pairs',
    'seed_list',
)
_IDENTITY_DEFAULTS = {'dep_all_pairs': False, 'seed_list': None}


def git_commit(cwd: Path) -> str | None:
    """Current HEAD commit of the repo at `cwd`, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _rows_for_seeds(
    seeds: list[int],
    target: str,
    split: str,
    med_cfg: dict[str, Any],
    max_seq_len: int,
    dep_all_pairs: bool,
) -> tuple[list[list[int]], list[int], list[int], list[int], list[int], int]:
    """Worker: build rows for the given seeds as plain columns (picklable)."""
    from src.probe.vprobe import rows_for_problem

    input_ids: list[list[int]] = []
    labels: list[int] = []
    groups: list[int] = []
    param_a: list[int] = []
    param_b: list[int] = []
    skipped = 0
    for seed in seeds:
        rows, n_skipped = rows_for_problem(
            seed, target=target, split=split, med_cfg=med_cfg, max_seq_len=max_seq_len,
            dep_all_pairs=dep_all_pairs,
        )
        skipped += n_skipped
        for r in rows:
            input_ids.append(r.input_ids)
            labels.append(r.label)
            groups.append(r.group)
            param_a.append(r.param_a)
            param_b.append(r.param_b)
    return input_ids, labels, groups, param_a, param_b, skipped


def _probe_seed_op(seed: int, split: str, med_cfg: dict[str, Any]) -> tuple[int, int, int]:
    """Worker: regenerate problem `seed`; return `(seed, n_op, n_param)`."""
    from src.probe.labels import regenerate_problem

    pp = regenerate_problem(seed, split=split, med_cfg=med_cfg)
    return seed, int(pp.problem.n_op), len(pp.all_param)


def find_showcase_seeds(
    out: Path | str,
    *,
    per_op: int = 3,
    split: str = 'test',
    scan_seed: int = 0,
    seed_lo: int = 1_000_000,
    max_scan: int = 2_000,
    workers: int = 8,
    med_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan scattered (non-sequential) seeds until `per_op` problems exist per op count.

    Candidate seeds are drawn uniformly from `[seed_lo, 2**31)` by an RNG keyed on
    `scan_seed` -- deterministic, but the accepted seeds look nothing like a range (and
    `seed_lo` keeps them disjoint from the sequential training ranges near 0). Problem
    difficulty is iGSM's own `n_op` (operation count; iGSM-med caps at `max_op=15`). High
    ops are rare (~1-2% of problems), so the scan runs `workers` regenerations in parallel
    and stops when every op bucket is full or `max_scan` candidates were tried -- unfilled
    buckets are reported, not fatal.

    Writes `{split, scan_seed, per_op, per_op_seeds: {op: [seed, ...]}, seeds: [...]}` as
    JSON to `out` (input for `gen-data --seeds-file`) and returns it.
    """
    rng = np.random.default_rng(scan_seed)
    candidates = [int(s) for s in rng.integers(seed_lo, 2**31 - 1, size=max_scan)]
    med_cfg = IGSM_MED if med_cfg is None else med_cfg
    wanted_ops = set(range(1, med_cfg.get('max_op', 15) + 1))
    buckets: dict[int, list[int]] = {}
    n_scanned = 0
    chunk = max(workers * 4, 16)
    with ProcessPoolExecutor(max_workers=workers, initializer=ensure_igsm_submodule) as pool:
        with tqdm(total=max_scan, desc='scan seeds', unit='prob') as pbar:
            for lo in range(0, len(candidates), chunk):
                batch = candidates[lo : lo + chunk]
                results = pool.map(
                    _probe_seed_op, batch, repeat(split, len(batch)), repeat(med_cfg, len(batch))
                )
                for seed, n_op, _n_param in results:
                    n_scanned += 1
                    pbar.update(1)
                    bucket = buckets.setdefault(n_op, [])
                    if len(bucket) < per_op:
                        bucket.append(seed)
                if all(len(buckets.get(op, [])) >= per_op for op in wanted_ops):
                    break

    unfilled = {op: len(b) for op, b in sorted(buckets.items()) if len(b) < per_op}
    if unfilled:
        print(f'buckets not filled to {per_op} after {n_scanned} scans: {unfilled}')
    per_op_seeds = {str(op): buckets[op] for op in sorted(buckets)}
    result = {
        'kind': 'showcase_seeds',
        'split': split,
        'med_cfg': med_cfg,
        'scan_seed': scan_seed,
        'per_op': per_op,
        'n_scanned': n_scanned,
        'per_op_seeds': per_op_seeds,
        'seeds': [s for seeds in per_op_seeds.values() for s in seeds],
        'created': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    ops = ', '.join(f'{op}:{len(b)}' for op, b in sorted(buckets.items()))
    print(f'{sum(len(b) for b in buckets.values())} seeds across ops [{ops}] -> {out}')
    return result


def _write_table_atomic(table: pa.Table, path: Path) -> None:
    """Write parquet via a hidden `.partial` temp file, then atomically rename (see igsm.py)."""
    tmp = path.with_name(f'.{path.name}.partial')
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)


def generate_rows_to_dir(
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
    seed_list: list[int] | None = None,
) -> dict[str, Any]:
    """Generate V-probe rows for `n_problems` seeds into `out`; return the metadata dict.

    Resumable: complete shards are skipped on re-run. If `out` already holds a dataset with
    *different* identity parameters (target/split/seeds/config), refuses unless `overwrite`.

    `seed_list` overrides the sequential `[seed_start, seed_start + n_problems)` range with
    explicit problem seeds (e.g. the scattered showcase seeds from `find_showcase_seeds`);
    shard `b` then covers `seed_list[b*problems_per_shard : (b+1)*problems_per_shard]`.
    """
    from src.probe.vprobe import MAX_SEQ_LEN

    out = Path(out)
    med_cfg = IGSM_MED if med_cfg is None else med_cfg
    if seed_list is not None:
        n_problems = len(seed_list)
    identity = {
        'target': target,
        'split': split,
        'seed_start': seed_start,
        'n_problems': n_problems,
        'med_cfg': med_cfg,
        'max_seq_len': MAX_SEQ_LEN,
        'dep_all_pairs': dep_all_pairs,
        'seed_list': seed_list,
    }

    if overwrite and out.exists():
        shutil.rmtree(out)
    meta_path = out / METADATA_NAME
    if meta_path.exists():
        existing = json.loads(meta_path.read_text(encoding='utf-8'))
        mismatched = {
            k: (existing.get(k, _IDENTITY_DEFAULTS.get(k)), identity[k])
            for k in _IDENTITY_KEYS
            if existing.get(k, _IDENTITY_DEFAULTS.get(k)) != identity[k]
        }
        if mismatched:
            raise ValueError(
                f'{out} already holds a different dataset (mismatch: {mismatched}); '
                'pass --overwrite or choose another --out'
            )
    out.mkdir(parents=True, exist_ok=True)
    for partial in out.glob('*.partial'):
        partial.unlink()

    num_shards = math.ceil(n_problems / problems_per_shard) if n_problems > 0 else 0
    done = _done_batches(out, num_shards)
    if done:
        print(f'Resuming: {len(done)}/{num_shards} shards already complete.')

    all_seeds = (
        list(seed_list) if seed_list is not None
        else list(range(seed_start, seed_start + n_problems))
    )
    pbar = tqdm(
        total=n_problems, desc=f'gen vprobe-{target} rows', unit='prob', dynamic_ncols=True
    )
    with ProcessPoolExecutor(max_workers=workers, initializer=ensure_igsm_submodule) as pool:
        for b in range(num_shards):
            shard_seeds = all_seeds[b * problems_per_shard : (b + 1) * problems_per_shard]
            if b in done:
                pbar.update(len(shard_seeds))
                continue
            # contiguous seed sub-slices across workers; each seed is independent
            futures = []
            cursor = 0
            for count in _worker_counts(len(shard_seeds), workers):
                futures.append(
                    pool.submit(
                        _rows_for_seeds, shard_seeds[cursor : cursor + count],
                        target, split, med_cfg, MAX_SEQ_LEN, dep_all_pairs,
                    )
                )
                cursor += count
            ids_col: list[list[int]] = []
            label_col: list[int] = []
            group_col: list[int] = []
            pa_col: list[int] = []
            pb_col: list[int] = []
            for f in futures:
                ids, labels, groups, params_a, params_b, _skipped = f.result()
                ids_col.extend(ids)
                label_col.extend(labels)
                group_col.extend(groups)
                pa_col.extend(params_a)
                pb_col.extend(params_b)
            table = pa.table(
                {
                    'input_ids': ids_col, 'label': label_col, 'group': group_col,
                    'param_a': pa_col, 'param_b': pb_col,
                }
            )
            _write_table_atomic(table, _shard_path(out, b))
            pbar.update(len(shard_seeds))
    pbar.close()

    # dataset-wide stats from the shards themselves (correct even on resumed runs)
    shards = sorted(out.glob('batch_*.parquet'))
    n_rows = 0
    n_positive = 0
    for shard in shards:
        labels = pq.read_table(str(shard), columns=['label']).column('label')
        n_rows += len(labels)
        n_positive += pc.sum(labels).as_py()

    metadata: dict[str, Any] = {
        'kind': 'vprobe_rows',
        **identity,
        'problems_per_shard': problems_per_shard,
        'n_shards': len(shards),
        'n_rows': n_rows,
        'positive_frac': n_positive / n_rows if n_rows else 0.0,
        'model_path': model_path,  # provenance only; rows are model-independent (see docstring)
        'tokenizer': 'gpt2 (iGSM; probe markers START=225 MID=227 END=226)',
        'repo_commit': git_commit(REPO_ROOT),
        'igsm_commit': git_commit(IGSM_REPO_ROOT),
        'command': ' '.join(sys.argv),
        'created': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(
        f'{n_problems} problems -> {n_rows} rows ({metadata["positive_frac"]:.3f} positive) '
        f'in {len(shards)} shards at {out}'
    )
    return metadata


def load_vprobe_rows(path: Path | str) -> tuple[list, dict[str, Any]]:
    """Load an offline row dataset; return `(rows, metadata)`.

    Row order is deterministic (sorted shards, insertion order within each), so a fixed
    seed reproduces the same train/val group split across runs.
    """
    from src.probe.vprobe import VProbeRow

    path = Path(path)
    shards = sorted(path.glob('batch_*.parquet'))
    if not shards:
        raise FileNotFoundError(f'no parquet shards at {path} -- run `gen-data` first')
    meta_path = path / METADATA_NAME
    metadata = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}

    rows: list[VProbeRow] = []
    for shard in shards:
        table = pq.read_table(str(shard))
        n = table.num_rows
        ids_col = table.column('input_ids').to_pylist()
        label_col = table.column('label').to_pylist()
        group_col = table.column('group').to_pylist()
        # pair-identity columns: datasets written before they existed load as -1
        names = table.column_names
        pa_col = table.column('param_a').to_pylist() if 'param_a' in names else [-1] * n
        pb_col = table.column('param_b').to_pylist() if 'param_b' in names else [-1] * n
        rows.extend(
            VProbeRow(input_ids=ids, label=label, group=group, param_a=a, param_b=b)
            for ids, label, group, a, b in zip(
                ids_col, label_col, group_col, pa_col, pb_col, strict=True
            )
        )
    return rows, metadata
