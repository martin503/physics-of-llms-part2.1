#!/usr/bin/env python3
"""Report rows (= packed windows) per parquet shard, and the DDP imbalance they imply.

Accelerate shards the stream at *file* granularity, so a rank's window count is the sum of
its files' row counts. Equal file counts therefore only mean equal data if the files are
equal-sized -- and unequal window counts across ranks mean one rank runs dry first, which
in DDP is a hang at the allreduce rather than a clean stop.

Reads parquet metadata only (no data pages), except one row per shard to recover the window
length. Usage::

    uv run python scripts/debug/shard_row_counts.py <data_dir> [world_size] [num_workers]

``world_size``/``num_workers`` default to the train.sbatch config (3 ranks, 4 workers).
"""

import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.igsm import SHARD_PREFIX, SHARD_SUFFIX  # noqa: E402

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/igsm_train_120M')
world = int(sys.argv[2]) if len(sys.argv) > 2 else 3
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4

files = sorted(data_dir.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
if not files:
    raise SystemExit(f'no {SHARD_PREFIX}*{SHARD_SUFFIX} shards at {data_dir}')

counts = [pq.read_metadata(str(f)).num_rows for f in files]
total = sum(counts)
first_row = next(pq.ParquetFile(str(files[0])).iter_batches(batch_size=1))
window_len = len(first_row.column('input_ids')[0])

print(f'{data_dir}: {len(files)} shards, {total:,} rows, window length {window_len}')
sizes = sorted(set(counts))
if len(sizes) == 1:
    print(f'all shards uniform at {sizes[0]:,} rows')
else:
    print(f'{len(sizes)} distinct shard sizes: min {min(counts):,}  max {max(counts):,}  '
          f'mean {total / len(counts):,.1f}')
    odd = [(f.name, c) for f, c in zip(files, counts, strict=True) if c != max(counts)]
    print(f'{len(odd)} shard(s) below max; first few: {odd[:5]}')

# Accelerate assigns files round-robin: rank r takes files[r::world]. Workers sub-shard the
# same way inside a rank, which does not change the rank's total.
print(f'\nfile-level split across {world} rank(s):')
per_rank = [sum(counts[r::world]) for r in range(world)]
for r, n in enumerate(per_rank):
    print(f'  rank {r}: {len(counts[r::world])} files, {n:,} rows')
spread = max(per_rank) - min(per_rank)
print(f'  spread {spread:,} rows ({spread / max(per_rank):.3%} of the largest rank)')
print(f'  a 1-epoch run stops at the shortest rank: {min(per_rank) * world:,}/{total:,} rows '
      f'({min(per_rank) * world / total:.4%}) before ranks desynchronise')

# Each rank sub-shards its files across its workers; a worker with zero files yields nothing.
print(f'\nworker sub-split ({workers} worker(s) per rank):')
for r in range(world):
    rank_files = counts[r::world]
    empty = sum(1 for w in range(workers) if not rank_files[w::workers])
    per_worker = [sum(rank_files[w::workers]) for w in range(workers)]
    flag = f'  <-- {empty} worker(s) get no files' if empty else ''
    print(f'  rank {r}: {[f"{n:,}" for n in per_worker]}{flag}')
