"""Pack eval datasets into fixed-length training windows.

Reads the one-row-per-problem parquet files written by ``src/data/eval.py`` (their
``gold_token_id`` column is the full ``[222] problem [223] solution [224] answer
[50256]`` training sequence), shuffles all problems, concatenates them, and chunks
into fixed ``--ctx`` windows -- the paper's "concatenate + right-truncate" packing.
Output is ``batch_NNNNNN.parquet`` shards with a single ``input_ids`` column, in the
same format ``load_igsm_dataset`` / ``src/train/gpt.py`` consume, so the packed dir
can be passed straight to training as ``--data-dir``.

Usage::

    uv run python -m src.data.pack --in data/igsm_eval --out data/igsm_eval_pack \\
        --ctx 768 --seed 0
    # then train on it:
    uv run python -m src.train.gpt --data-dir data/igsm_eval_pack ...
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import pyarrow.parquet as pq
import typer

from src.data.igsm import pack_sequences

SHARD_PREFIX = 'batch_'
SHARD_SUFFIX = '.parquet'


def _load_streams(input_dir: Path, column: str, pattern: str) -> list[list[int]]:
    """Read ``column`` from every ``pattern`` parquet in ``input_dir`` into a list of streams."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no {pattern!r} files in {input_dir}')
    streams: list[list[int]] = []
    for f in files:
        streams.extend(pq.read_table(f).column(column).to_pylist())
    return streams


def _write_shards(windows: list[list[int]], out: Path, shard_size: int) -> int:
    """Write uniform-length ``windows`` to ``batch_*.parquet`` shards; return the shard count."""
    assert windows, 'cannot write empty pack (no windows)'
    assert all(len(w) == len(windows[0]) for w in windows), 'non-uniform window lengths'
    out.mkdir(parents=True, exist_ok=True)
    n_shards = max(1, (len(windows) + shard_size - 1) // shard_size)
    for i in range(n_shards):
        chunk = windows[i * shard_size : (i + 1) * shard_size]
        path = out / f'{SHARD_PREFIX}{i:06d}{SHARD_SUFFIX}'
        pq.write_table(pa.table({'input_ids': chunk}), str(path))
    return n_shards


app = typer.Typer(add_completion=False, help='Pack eval data into fixed-length training windows.')


@app.command()
def pack(
    input_dir: Annotated[
        Path, typer.Option('--in', help='Dir with eval parquet files (from src.data.eval).')
    ] = Path('data/igsm_eval'),
    output_dir: Annotated[
        Path, typer.Option('--out', help='Output dir for batch_*.parquet shards.')
    ] = Path('data/igsm_eval_pack'),
    context_length: Annotated[int, typer.Option('--ctx', help='Packed window length.')] = 768,
    seed: Annotated[
        int, typer.Option('--seed', help='Shuffle seed (problems are shuffled before packing).')
    ] = 0,
    column: Annotated[
        str,
        typer.Option(
            '--column',
            help='Column to pack. gold_token_id = full train seq; prompt_ids = problem only.',
        ),
    ] = 'gold_token_id',
    include: Annotated[
        str,
        typer.Option('--include', help='Filename glob to select slices (default all *.parquet).'),
    ] = '*.parquet',
    shard_size: Annotated[
        int, typer.Option('--shard-size', help='Windows per output shard.')
    ] = 100_000,
) -> None:
    """Shuffle eval problems and pack them into fixed-length training windows."""
    streams = _load_streams(input_dir, column, include)
    lengths = [len(s) for s in streams]
    random.Random(seed).shuffle(streams)
    windows = pack_sequences(streams, context_length=context_length)
    if not windows:
        typer.echo(
            f'  {len(streams)} problems but {sum(lengths)} total tokens < ctx {context_length}; '
            f'nothing to pack. Lower --ctx.'
        )
        return
    n_shards = _write_shards(windows, output_dir, shard_size)
    typer.echo(
        f'  {len(streams)} problems (len min/median/max = '
        f'{min(lengths)}/{sorted(lengths)[len(lengths) // 2]}/{max(lengths)})'
    )
    typer.echo(
        f'  -> shuffled (seed={seed}) -> {len(windows)} windows of len {context_length} '
        f'in {n_shards} shard(s) at {output_dir}'
    )


if __name__ == '__main__':
    app()
