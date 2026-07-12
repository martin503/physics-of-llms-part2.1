"""Merge parquet shards into fewer, larger, row-shuffled shards.

Reads the ``batch_*.parquet`` shards written by :mod:`src.data.igsm` (or
:mod:`src.data.pack`), groups them ``n`` at a time, shuffles every row across the
``n`` shards together, and writes one output shard per group -- yielding
``ceil(num_shards / n)`` shards. The output keeps the same ``batch_*.parquet``
layout (and all original columns) that :func:`src.data.igsm.load_igsm_dataset`
and training consume, so the merged dir is a drop-in replacement for the input
dir (e.g. pass it straight to training as ``--data-dir``).

Groups are disjoint and only ``n`` shards are held in memory at once, so the
merge stays cheap regardless of total dataset size; the per-group row shuffle
re-randomizes which windows share a shard without a full in-memory shuffle.

Usage::

    uv run python -m src.data.merge --in data/igsm_train --out data/igsm_train_merged \\
        --n 4 --seed 0
    # then train on it:
    uv run python -m src.train.gpt --data-dir data/igsm_train_merged ...
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

from src.data.igsm import SHARD_PREFIX, SHARD_SUFFIX


def merge_shards(input_dir: Path, output_dir: Path, n: int, seed: int = 0) -> tuple[int, int]:
    """Group ``input_dir``'s shards ``n`` at a time, row-shuffle each group, and write one
    output shard per group to ``output_dir``; return ``(in_shards, out_shards)``.

    Args:
        input_dir: Dir of ``batch_*.parquet`` shards to merge.
        output_dir: Output dir (created if missing); written as ``batch_NNNNNN.parquet``.
        n: Input shards merged into each output shard (last group may be smaller; ``>= 1``).
        seed: Seed for the per-group row shuffle.

    Returns:
        ``(number of input shards, number of output shards)``.
    """
    if n <= 0:
        raise ValueError(f'n must be a positive integer, got {n}')
    files = sorted(input_dir.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
    if not files:
        raise FileNotFoundError(f'no {SHARD_PREFIX}*{SHARD_SUFFIX} shards in {input_dir}')

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    n_out = 0
    for start in tqdm(range(0, len(files), n)):
        group = files[start : start + n]
        table = pa.concat_tables([pq.read_table(f) for f in group])
        order = list(range(table.num_rows))
        rng.shuffle(order)
        table = table.take(order)  # reorder rows in place; preserves all columns/types
        pq.write_table(table, str(output_dir / f'{SHARD_PREFIX}{n_out:06d}{SHARD_SUFFIX}'))
        n_out += 1
    return len(files), n_out


app = typer.Typer(add_completion=False, help='Merge parquet shards into fewer, shuffled shards.')


@app.command()
def merge(
    n: Annotated[
        int, typer.Option('--n', help='Input shards merged into each output shard (>= 1).')
    ],
    input_dir: Annotated[
        Path, typer.Option('--in', help='Dir of batch_*.parquet shards to merge.')
    ] = Path('data/igsm_train'),
    output_dir: Annotated[
        Path, typer.Option('--out', help='Output dir for the merged batch_*.parquet shards.')
    ] = Path('data/igsm_train_merged'),
    seed: Annotated[
        int, typer.Option('--seed', help='Seed for the per-group row shuffle.')
    ] = 0,
) -> None:
    """Group shards ``n`` at a time, row-shuffle each group into one output shard."""
    in_count, out_count = merge_shards(input_dir, output_dir, n, seed)
    typer.echo(
        f'  {in_count} shards -> {out_count} shards (groups of {n}, seed={seed}) at {output_dir}'
    )


if __name__ == '__main__':
    app()
