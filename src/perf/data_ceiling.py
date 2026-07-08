"""Data pipeline throughput ceiling (CPU-only, no model).

Measures how fast each data source can deliver **packed windows**: random tensors (DataLoader IPC
upper bound with ~zero production cost), real streaming parquet + TRL pack, real in-memory + TRL
pack, and TRL pack-only. Sweeps ``num_workers``. CPU-only -- no CUDA, so workers fork safely.

The decision: compare the pipeline's producible ``windows/s * ctx`` (tokens/s) against
``gpu_ceiling``'s consumed tokens/s. If production < consumption, the GPU is data-starved by
construction; if production >> consumption, the bottleneck is elsewhere (and ``e2e_bench`` will
show it as idle GPU between kernels).

Examples::

    # quick smoke (1s per cell, no workers):
    uv run python -m src.perf.data_ceiling --source all --seconds 1 --num-workers 0

    # full sweep on the real raw store:
    uv run python -m src.perf.data_ceiling --source all --num-workers 0,1,2,4 --ctx 12288
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Annotated

import pyarrow as pa
import typer
from datasets import IterableDataset
from torch.utils.data import DataLoader
from trl import pack_dataset
from trl.data_utils import _pack_bfd  # type: ignore[attr-defined]

from src.data.igsm import VOCAB_SIZE, load_igsm_dataset, load_igsm_stream

app = typer.Typer(add_completion=False, help='Data pipeline throughput ceiling (CPU-only).')


def _passthrough_collate(batch: list) -> list:
    """No stacking -- throughput only cares that windows were produced (seq_lengths is var-len)."""
    return batch


def _rand_dataset(ctx: int, vocab: int) -> IterableDataset:
    """Infinite random ctx-length windows (~zero production cost) -- DataLoader IPC ceiling."""

    def gen(c: int, v: int):  # noqa: ANN202 -- datasets from_generator wants a plain generator
        while True:
            yield {'input_ids': [random.randrange(v) for _ in range(c)]}

    return IterableDataset.from_generator(gen, gen_kwargs={'c': ctx, 'v': vocab})


def _measure(
    dataset, num_workers: int, batch_size: int, seconds: float, warmup_batches: int = 3
) -> tuple[float, int]:
    """Iterate ``dataset`` through a DataLoader for ``seconds`` -> (windows/sec, count)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_passthrough_collate,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    it = iter(loader)
    for _ in range(warmup_batches):  # absorb worker spawn / first-read latency
        next(it)
    count = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        next(it)
        count += batch_size
    elapsed = time.perf_counter() - t0
    return count / elapsed, count


def _measure_pack_only(data_dir: Path, ctx: int, batch_problems: int, seconds: float) -> float:
    """Time TRL ``_pack_bfd`` directly on raw problems (no DataLoader) -> ms/problem."""
    raw = load_igsm_dataset(data_dir)
    n = min(batch_problems, len(raw))
    rows = raw.select(range(n))['input_ids']
    table = pa.table({'input_ids': rows})
    # warmup
    _pack_bfd(table, seq_length=ctx)
    count = 0
    t0 = time.perf_counter()
    deadline = t0 + seconds
    while time.perf_counter() < deadline:
        _pack_bfd(table, seq_length=ctx)
        count += n
    elapsed = time.perf_counter() - t0
    return elapsed / count * 1000 if count else float('nan')


@app.command()
def main(
    data_dir: Annotated[
        Path, typer.Option('--data-dir', help='Raw iGSM problems (unpacked).')
    ] = Path('data/igsm_raw'),
    ctx: Annotated[int, typer.Option('--ctx', help='Packed window length.')] = 12288,
    vocab_size: Annotated[int, typer.Option('--vocab-size')] = VOCAB_SIZE,
    source: Annotated[
        str,
        typer.Option('--source', help='Comma list: random,stream,inmem,pack-only  (or "all").'),
    ] = 'all',
    num_workers: Annotated[
        str, typer.Option('--num-workers', help='Comma list of worker counts, e.g. 0,1,2,4.')
    ] = '0,1,2,4',
    batch_size: Annotated[
        int, typer.Option('--batch-size', help='Windows per DataLoader batch.')
    ] = 16,
    seconds: Annotated[
        float, typer.Option('--seconds', help='Measurement window per cell.')
    ] = 10.0,
    pack_batch_problems: Annotated[
        int,
        typer.Option(
            '--pack-batch-problems', help='Raw problems per _pack_bfd call (pack-only source).'
        ),
    ] = 1000,
) -> None:
    """Measure windows/s + ms/problem across data sources x num_workers."""
    sources = ['random', 'stream', 'inmem', 'pack-only'] if source == 'all' else source.split(',')
    workers_list = [int(w) for w in num_workers.split(',')]
    typer.echo(
        f'[data_ceiling] ctx={ctx} batch_size={batch_size} seconds={seconds} '
        f'sources={sources} workers={workers_list}'
    )

    for src in sources:
        typer.echo(f'\n--- source={src} ---')
        if src == 'pack-only':
            ms = _measure_pack_only(data_dir, ctx, pack_batch_problems, max(seconds, 1.0))
            typer.echo(
                f'  pack-only: {ms:.3f} ms/problem  ({1000 / ms:.0f} problems/s/_pack_bfd call)'
            )
            continue

        # Build the dataset once per source.
        if src == 'random':
            dataset = _rand_dataset(ctx, vocab_size)
        elif src == 'stream':
            dataset = pack_dataset(load_igsm_stream(data_dir), seq_length=ctx, strategy='bfd')
        elif src == 'inmem':
            t0 = time.perf_counter()
            dataset = pack_dataset(load_igsm_dataset(data_dir), seq_length=ctx, strategy='bfd')
            # Force eager materialization (map-style .map is lazy until iterated); measure it.
            _ = list(dataset.take(1))
            pack_time = time.perf_counter() - t0
            typer.echo(f'  (inmem eager-pack of store: {pack_time:.1f}s one-time)')
        else:  # pragma: no cover
            raise ValueError(f'unknown source {src!r}')

        for w in workers_list:
            try:
                wps, count = _measure(dataset, w, batch_size, seconds)
            except Exception as e:  # noqa: BLE001 -- report and continue to next cell
                typer.echo(f'  workers={w}: FAILED ({e})')
                continue
            tps = wps * ctx
            typer.echo(
                f'  workers={w}: {wps:,.0f} windows/s  {tps:,.0f} tokens/s  '
                f'({count} windows in {seconds:.0f}s)'
            )

    typer.echo(
        '\ninterpret: if max windows/s * ctx (tokens/s) < gpu_ceiling consumed tokens/s, '
        'the GPU is data-starved. random vs stream/inmem shows the data-production cost.'
    )


if __name__ == '__main__':
    app()
