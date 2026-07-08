"""iGSM data generation for the Physics-of-LMs Part 2.1 reproduction.

Wraps the official iGSM generator (git submodule at ``iGSM/``, not pip-installable) to
produce tokenized, **packed** training data matching the paper:

* problems use the med config (``max_op=15, max_edge=20, perm_level=5, detail_level=0``);
* each problem is tokenized by iGSM's GPT-2 tokenizer into
  ``[222] problem [223] solution [224] answer [50256]``;
* problems are concatenated and chunked into fixed ``context_length`` (768) windows
  (the paper's "concatenate + right-truncate" packing; no loss masking);
* the train/test split follows iGSM's hash bins (``train_bin=[0..15]``, ``test_bin=[16..22]``).

Generation is **streaming and resumable**: problems are produced in batches, each batch
written atomically to its own parquet shard (``batch_NNNNNN.parquet``). If generation is
interrupted, re-running resumes from the last complete shard -- only the in-flight batch is
re-done. The parquet-shard layout is also Hub-friendly (see ``push``).

CLI::

    uv run python -m src.data.igsm sanity
    uv run python -m src.data.igsm generate --split train --num-problems 200000 \\
        --batch-size 5000 --workers 8 --out data/igsm_train
    # raw (unpacked) problems for online TRL packing during training (see src.train.gpt_pack):
    uv run python -m src.data.igsm generate --raw --split train --num-problems 200000 \\
        --batch-size 20000 --workers 8 --out data/igsm_raw
    uv run python -m src.data.igsm push --data-dir data/igsm_train --repo-id user/igsm-med
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
import pyarrow.parquet as pq
import typer
from tqdm import tqdm

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
IGSM_REPO_ROOT = REPO_ROOT / 'iGSM'
IGSM_ENTRYPOINT = IGSM_REPO_ROOT / 'data_gen' / 'pretrain' / 'id_gen.py'

# Special tokens used by iGSM's token layout (GPT-2 BPE, vocab 50257).
PROB_BOS = 222  # start of problem
SOL_BOS = 223  # transition problem -> solution
ANS_BOS = 224  # start of answer
EOS = 50256  # GPT-2 EOS (stop token)

VOCAB_SIZE = 50257
MOD = 23
DEFAULT_CONTEXT_LENGTH = 768
DEFAULT_BATCH_SIZE = 5_000
SHARD_PREFIX = 'batch_'
SHARD_SUFFIX = '.parquet'

# iGSM "med" config (paper formatting: perm_level=5 full shuffle, detail_level=0).
IGSM_MED: dict[str, Any] = dict(max_op=15, max_edge=20, perm_level=5, detail_level=0)

_IGSM_READY = False


def ensure_igsm_submodule() -> Path:
    """Ensure the iGSM submodule is checked out and importable; return its root path.

    Adds the iGSM repo root to ``sys.path`` (it has no ``__init__.py``) exactly once.
    Attempts ``git submodule update --init iGSM`` if missing, else raises a clear error.
    """
    global _IGSM_READY
    if _IGSM_READY:
        return IGSM_REPO_ROOT
    if not IGSM_ENTRYPOINT.is_file():
        try:
            subprocess.run(
                ['git', 'submodule', 'update', '--init', 'iGSM'],
                check=True,
                cwd=REPO_ROOT,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                f'iGSM submodule not checked out at {IGSM_REPO_ROOT}. Run '
                '`git submodule update --init iGSM` (or clone '
                'https://github.com/facebookresearch/iGSM.git into ./iGSM). '
                f'Underlying error: {e}'
            ) from e
        if not IGSM_ENTRYPOINT.is_file():
            raise RuntimeError(f'iGSM still missing {IGSM_ENTRYPOINT} after submodule init.')
    root = str(IGSM_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    _IGSM_READY = True
    return IGSM_REPO_ROOT


def get_bins(split: str) -> list[int]:
    """Return the list of allowed hash bins for ``split`` ('train' or 'test')."""
    ensure_igsm_submodule()
    from const.params import test_bin, train_bin  # type: ignore[import-not-found]

    bins = train_bin if split == 'train' else test_bin
    return list(bins)


def _generate_chunk(
    num_problems: int, seed: int, bins: list[int], med_cfg: dict[str, Any]
) -> list[list[int]]:
    """Generate ``num_problems`` token streams in one process (worker for the pool)."""
    ensure_igsm_submodule()
    from data_gen.pretrain.id_gen import IdGen  # type: ignore[import-not-found]
    from tools.tools import fix_seed  # type: ignore[import-not-found]

    fix_seed(seed)  # sets random + numpy seeds (NOT torch); per-worker seed -> distinct problems
    gen = IdGen(**med_cfg)  # gen_prob mutates the instance, so one generator per unit
    streams: list[list[int]] = []
    for _ in range(num_problems):
        gen.gen_prob(bins, p_format='pq')  # type: ignore[attr-defined]
        token_id = list(gen.token_id)  # type: ignore[attr-defined]
        assert token_id, 'gen_prob produced an empty token_id'
        assert token_id[0] == PROB_BOS and token_id[-1] == EOS, 'unexpected iGSM token layout'
        streams.append(token_id)
    return streams


def _unit_counts(num_problems: int, unit: int) -> list[int]:
    """Split ``num_problems`` into many small chunks of size ``unit`` (last may be smaller).

    Produces many fine-grained units so a :class:`~concurrent.futures.ProcessPoolExecutor` can
    **dynamically rebalance** work across processes. This matters because iGSM-med per-problem
    generation time is heavy-tailed (~22 retries/problem; stragglers ~7x the median): one big
    chunk per worker means a single unlucky chunk stalls the whole batch (barrier on ``max``),
    collapsing parallel scaling and making throughput batch-size/noise-sensitive. Many small
    units let fast workers absorb the slack, so wall time tracks the *average* rather than the
    *max*.
    """
    if num_problems <= 0:
        return []
    unit = max(1, unit)
    n_full, rem = divmod(num_problems, unit)
    return [unit] * n_full + ([rem] if rem else [])


def _fine_unit(num_problems: int, workers: int) -> int:
    """Pick a fine chunk size targeting ~8 rebalancing units per worker (capped to 1..256).

    8 units/worker is enough for the pool to smooth out heavy-tailed stragglers without paying
    excessive per-unit overhead (each unit spins up its own ``IdGen``). The 256 cap keeps even
    very large batches finely grained (e.g. 20k problems -> ~79 units).
    """
    if num_problems <= 0:
        return 1
    if workers <= 1:
        return num_problems
    target_units = workers * 8
    unit = num_problems // target_units
    return max(1, min(256, unit or 1))


def generate_problems(
    *,
    num_problems: int,
    split: str = 'train',
    workers: int = 8,
    seed: int = 0,
    med_cfg: dict[str, Any] | None = None,
) -> list[list[int]]:
    """Generate ``num_problems`` tokenized iGSM problems for ``split`` across ``workers``.

    Spawns its own (short-lived) process pool -- convenient for one-shot use. For resumable
    batched generation use :func:`generate_to_dir`, which keeps a single persistent pool.

    Each unit ``j`` is seeded ``seed + j`` (iGSM uses module-level RNG state, so distinct
    process-level seeds are required to avoid duplicate problems). Work is split into many
    fine units (see :func:`_fine_unit`) so the pool rebalances heavy-tailed stragglers; with
    ``workers <= 1`` generation runs inline in a single ``IdGen``.
    """
    med_cfg = IGSM_MED if med_cfg is None else med_cfg
    bins = get_bins(split)
    if workers <= 1 or num_problems <= 0:
        return _generate_chunk(max(num_problems, 0), seed, bins, med_cfg)

    unit = _fine_unit(num_problems, workers)
    counts = _unit_counts(num_problems, unit)
    tasks = [(c, seed + j, bins, med_cfg) for j, c in enumerate(counts)]

    streams: list[list[int]] = []
    with ProcessPoolExecutor(max_workers=workers, initializer=ensure_igsm_submodule) as pool:
        for chunk in pool.map(_generate_chunk, *zip(*tasks, strict=True)):
            streams.extend(chunk)
    return streams


def _generate_batch_in_pool(
    pool: ProcessPoolExecutor,
    num_problems: int,
    workers: int,
    base_seed: int,
    bins: list[int],
    med_cfg: dict[str, Any],
    *,
    unit: int | None = None,
) -> list[list[int]]:
    """Generate one batch's problems on a *persistent* pool (workers reuse the IdGen import).

    Work is split into many fine units (``unit`` problems each; auto-sized via
    :func:`_fine_unit` when omitted) and submitted at once so the pool **dynamically
    rebalances** across its ``workers`` processes. Each unit ``j`` is seeded ``base_seed + j``;
    the caller must pass ``base_seed`` values spaced by at least the number of units per batch
    (see the stride computed in :func:`generate_to_dir`) so seeds are globally unique and
    duplicate-free across batches.
    """
    if num_problems <= 0:
        return []
    if unit is None:
        unit = _fine_unit(num_problems, workers)
    counts = _unit_counts(num_problems, unit)
    futures = [
        pool.submit(_generate_chunk, c, base_seed + j, bins, med_cfg) for j, c in enumerate(counts)
    ]
    streams: list[list[int]] = []
    for f in futures:
        streams.extend(f.result())
    return streams


def pack_sequences(
    token_streams: Sequence[list[int]], context_length: int = DEFAULT_CONTEXT_LENGTH
) -> list[list[int]]:
    """Concatenate all token streams and chunk into fixed ``context_length`` windows.

    Mirrors the paper: problems are concatenated and the stream is read off in fixed
    windows, dropping the partial tail. EOS (50256) of one problem is followed directly by
    the [222] of the next inside a window -- the model learns the packing boundary.

    Note: when applied per-batch (see :func:`generate_to_dir`), each batch's partial tail is
    dropped independently -- a negligible difference vs. global packing for large batches.
    """
    assert context_length > 0
    flat: list[int] = []
    for stream in token_streams:
        flat.extend(stream)
    n = (len(flat) // context_length) * context_length
    return [flat[i : i + context_length] for i in range(0, n, context_length)]


# --------------------------------------------------------------------------- #
# Shard I/O (parquet, atomic, resumable, Hub-friendly)
# --------------------------------------------------------------------------- #


def _shard_path(out: Path, batch_idx: int) -> Path:
    return out / f'{SHARD_PREFIX}{batch_idx:06d}{SHARD_SUFFIX}'


def _write_shard_atomic(rows: list[list[int]], path: Path, require_uniform: bool = True) -> None:
    """Write ``rows`` to ``path`` as parquet via a temp file, then atomically rename.

    ``rows`` are either packed windows (uniform length, ``require_uniform=True``) or raw
    problems (variable length, ``require_uniform=False``). A partial write (interrupt) leaves a
    hidden ``.<name>.partial`` file that is never matched by :func:`_done_batches`, so resume
    correctly re-does the interrupted batch.
    """
    assert rows, 'cannot write an empty shard'
    if require_uniform:
        row_len = len(rows[0])
        assert all(len(r) == row_len for r in rows), 'non-uniform window lengths'
    tmp = path.with_name(f'.{path.name}.partial')
    table = pa.table({'input_ids': rows})  # list<int64> column
    pq.write_table(table, str(tmp))
    os.replace(tmp, path)  # atomic on the same filesystem


def _done_batches(out: Path, num_batches: int) -> set[int]:
    """Return the set of fully-written batch indices in ``[0, num_batches)``."""
    done: set[int] = set()
    for p in out.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'):
        try:
            idx = int(p.stem.removeprefix(SHARD_PREFIX))
        except ValueError:
            continue
        if idx < num_batches:
            done.add(idx)
    return done


def _count_shard_rows(shards: Sequence[Path]) -> int:
    """Sum row counts across shards (parquet metadata only -- does not read data)."""
    total = 0
    for shard in shards:
        total += pq.read_metadata(str(shard)).num_rows
    return total


def generate_dataset(windows: list[list[int]]) -> Dataset:
    """Wrap packed windows into an HF ``Dataset`` with a single ``input_ids`` column."""
    from datasets import Dataset

    assert all(len(w) == len(windows[0]) and w for w in windows), 'non-uniform/non-empty windows'
    return Dataset.from_dict({'input_ids': windows})


def load_igsm_dataset(path: Path | str) -> Dataset:
    """Load a sharded (parquet) iGSM dataset (or a legacy ``save_to_disk`` dir).

    Reads parquet shards directly via pyarrow and wraps them in an HF ``Dataset`` -- this
    avoids the datasets cache (no re-write on every load) and keeps loading zero-copy.
    """
    from datasets import Dataset, load_from_disk

    path = Path(path)
    shards = sorted(path.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
    if shards:
        tables = [pq.read_table(str(s)) for s in shards]
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return Dataset(table)
    if (path / 'dataset_info.json').exists():
        return load_from_disk(str(path))
    raise FileNotFoundError(f'No parquet shards or HF dataset found at {path}')


def load_igsm_stream(path: Path | str) -> IterableDataset:
    """Stream a sharded (parquet) iGSM dataset lazily as a HF ``IterableDataset``.

    Unlike :func:`load_igsm_dataset` (map-style, all-in-memory), this streams the
    ``batch_*.parquet`` shards -- suited to online TRL packing of **raw** problems during
    training (see :mod:`src.train.gpt_pack`). HF shards the stream by *file* across DDP ranks
    and DataLoader workers, so generate enough shards for your setup
    (``num_problems / batch_size >= world_size * num_workers``) or some workers will idle.

    Yields ``{'input_ids': [...]}`` rows (one raw problem each, or one packed window each --
    depending on how the dir was generated).
    """
    from datasets import load_dataset

    path = Path(path)
    shards = sorted(path.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
    if not shards:
        raise FileNotFoundError(f'No parquet shards found at {path}')
    return load_dataset(
        'parquet',
        data_files={'train': [str(s) for s in shards]},
        split='train',
        streaming=True,
    )


# --------------------------------------------------------------------------- #
# Resumable batched generation
# --------------------------------------------------------------------------- #


def generate_to_dir(
    out: Path | str,
    num_problems: int,
    *,
    split: str = 'train',
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 8,
    seed: int = 0,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    med_cfg: dict[str, Any] | None = None,
    overwrite: bool = False,
    pack: bool = True,
    problem_generator: Callable[..., list[list[int]]] | None = None,
) -> dict[str, Any]:
    """Generate ``num_problems`` iGSM problems, writing them as resumable parquet shards.

    Args:
        out: Output directory (created if missing).
        num_problems: Total problems to generate.
        split: 'train' or 'test' (hash-bin selection).
        batch_size: Problems per batch == one parquet shard. Smaller -> finer progress bar,
            smaller interrupt loss, and (for ``pack=False``) more files for better streaming
            parallelism; larger -> fewer files.
        workers: Parallel generation processes (one persistent pool for all batches).
        seed: Base RNG seed. Batch ``b`` unit ``j`` uses ``seed + b*stride + j`` (globally
            unique -> resumable and duplicate-free), where ``stride`` = units per batch
            (``ceil(batch_size / unit)``).
        context_length: Packed window length. Only used when ``pack=True`` (ignored otherwise).
        med_cfg: iGSM-med config (defaults to :data:`IGSM_MED`).
        overwrite: If True, delete ``out`` first and start from scratch.
        pack: If True (default), pack problems into fixed ``context_length`` windows (one row
            per window). If False, write **raw** problems (one variable-length row per problem)
            for online packing at train time (see :func:`load_igsm_stream`).
        problem_generator: Injectable generator (defaults to the real iGSM one) for testing.

    Returns:
        A stats dict (problems, batches, shards, rows, pack, ...).
    """
    out = Path(out)
    med_cfg = IGSM_MED if med_cfg is None else med_cfg

    if overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    # Drop any leftover partial writes from a previous interrupted run.
    for partial in out.glob('*.partial'):
        partial.unlink()

    num_batches = math.ceil(num_problems / batch_size) if num_problems > 0 else 0
    done = _done_batches(out, num_batches)
    resumed = len(done)
    if resumed:
        typer.echo(f'Resuming: {resumed}/{num_batches} batches already complete.')

    # Fine-grained work units so the pool rebalances heavy-tailed generation stragglers
    # (see _unit_counts / _fine_unit). `stride` = units per full batch, used to keep per-unit
    # seeds globally unique across batches (seed + b*stride + j).
    unit = _fine_unit(batch_size, workers)
    stride = len(_unit_counts(batch_size, unit))

    bins = get_bins(split)
    pbar = tqdm(total=num_problems, desc=f'gen iGSM-{split}', unit='prob', dynamic_ncols=True)

    def gen(num: int, base_seed: int) -> list[list[int]]:
        if problem_generator is not None:
            return problem_generator(
                num_problems=num, split=split, workers=workers, seed=base_seed, med_cfg=med_cfg
            )
        return _generate_batch_in_pool(pool, num, workers, base_seed, bins, med_cfg, unit=unit)

    pool: ProcessPoolExecutor | None = None
    if problem_generator is None:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=ensure_igsm_submodule)
    try:
        for b in range(num_batches):
            batch_count = min(batch_size, num_problems - b * batch_size)
            if b in done:
                pbar.update(batch_count)
                continue
            streams = gen(batch_count, seed + b * stride)
            if pack:
                rows = pack_sequences(streams, context_length=context_length)
            else:
                rows = streams  # raw problems (variable length); packed online at train time
            if rows:
                _write_shard_atomic(rows, _shard_path(out, b), require_uniform=pack)
            pbar.update(batch_count)
    finally:
        pbar.close()
        if pool is not None:
            pool.shutdown(wait=True)

    shards = sorted(out.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
    total_rows = _count_shard_rows(shards)
    stats = {
        'out': str(out),
        'split': split,
        'num_problems': num_problems,
        'num_batches': num_batches,
        'shards': len(shards),
        'rows': total_rows,
        'pack': pack,
        'context_length': context_length if pack else None,
    }
    if pack:
        typer.echo(
            f'  {num_problems} problems -> {len(shards)} shards, {total_rows} packed windows '
            f'of length {context_length} at {out}'
        )
    else:
        typer.echo(
            f'  {num_problems} problems -> {len(shards)} shards, {total_rows} raw problems at '
            f'{out} (pack online at train time)'
        )
    typer.echo(
        '  note: paper trains 100k steps x batch 512 ~= 51M windows; this finite dataset is '
        'cycled over epochs for the working version.'
    )
    return stats


app = typer.Typer(add_completion=False, help='iGSM data generation for Part 2.1 repro.')


@app.command()
def generate(
    split: Annotated[
        str, typer.Option(help="'train' (bins 0-15) or 'test' (bins 16-22).")
    ] = 'train',
    num_problems: Annotated[
        int, typer.Option('--num-problems', help='Number of iGSM problems to generate.')
    ] = 200_000,
    context_length: Annotated[
        int, typer.Option('--context-length', help='Packed window length (only with --pack).')
    ] = DEFAULT_CONTEXT_LENGTH,
    batch_size: Annotated[
        int,
        typer.Option(
            '--batch-size',
            help='Problems per shard. With --raw, more shards = better streaming parallelism.',
        ),
    ] = DEFAULT_BATCH_SIZE,
    workers: Annotated[int, typer.Option('--workers', help='Parallel generation processes.')] = 8,
    seed: Annotated[
        int, typer.Option('--seed', help='Base RNG seed (batch b worker i uses seed+b*workers+i).')
    ] = 0,
    out: Annotated[Path, typer.Option('--out', help='Output dataset directory.')] = Path(
        'data/igsm_train'
    ),
    overwrite: Annotated[
        bool, typer.Option('--overwrite', help='Delete the output dir first and start fresh.')
    ] = False,
    raw: Annotated[
        bool,
        typer.Option(
            '--raw/--pack',
            help='--raw: write unpacked problems (one row each) for online packing at train '
            'time. --pack (default): pack into fixed context_length windows.',
        ),
    ] = False,
) -> None:
    """Generate an iGSM-med dataset (resumable, batched, with a progress bar)."""
    generate_to_dir(
        out,
        num_problems,
        split=split,
        batch_size=batch_size,
        workers=workers,
        seed=seed,
        context_length=context_length,
        overwrite=overwrite,
        pack=not raw,
    )


@app.command(name='sanity')
def sanity(
    n: Annotated[int, typer.Option(help='Number of problems to validate.')] = 10,
) -> None:
    """Validate the iGSM generator produces well-formed token sequences."""
    streams = generate_problems(num_problems=n, split='test', workers=1, seed=0)
    assert len(streams) == n, f'expected {n} problems, got {len(streams)}'
    for s in streams:
        assert s[0] == PROB_BOS, 'must start with [222]'
        assert s[-1] == EOS, 'must end with [50256]'
        assert SOL_BOS in s and ANS_BOS in s, 'must contain [223] and [224]'
        assert all(0 <= t < VOCAB_SIZE for t in s), f'token out of vocab range: {max(s)}'
    typer.echo(f'sanity OK: {n} problems, lengths {[len(s) for s in streams]}')


@app.command(name='push')
def push(
    repo_id: Annotated[
        str, typer.Option('--repo-id', help='HF Hub dataset repo, e.g. user/igsm-med.')
    ],
    data_dir: Annotated[
        Path, typer.Option('--data-dir', help='Sharded dataset directory.')
    ] = Path('data/igsm_train'),
    private: Annotated[bool, typer.Option('--private/--public', help='Repo visibility.')] = True,
    token: Annotated[
        str | None, typer.Option('--token', help='HF token (defaults to cached login).')
    ] = None,
) -> None:
    """Push a sharded iGSM dataset to the HuggingFace Hub (as parquet)."""
    dataset = load_igsm_dataset(data_dir)
    typer.echo(f'Pushing {dataset.num_rows} rows from {data_dir} to {repo_id}...')
    dataset.push_to_hub(repo_id, private=private, token=token)
    typer.echo(f'Done. Load later with: load_dataset("{repo_id}")')


if __name__ == '__main__':
    app()
