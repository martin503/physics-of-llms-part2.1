"""Data-pipeline tests for ``src/train/gpt.py`` (streaming DDP path).

Covers the multi-GPU data properties of ``build_streaming_dataset``:

* (fast, in-process) the stream is handed every shard, DataLoader workers within a rank
  stream disjoint rows, and -- driving accelerate's real ``prepare_data_loader`` -- the DDP
  ranks together read every window exactly once.
* (slow, 2-rank CPU DDP) over 100 optimizer steps no training batch is seen by both ranks,
  under both ``dispatch_batches`` settings.

``test_b3``/``test_b4`` guard the double-shard bug without and with dataloader workers;
``test_b5`` pins the old behaviour that caused it. ``build_streaming_dataset``
used to slice ``files[rank::world]`` itself *and* accelerate sharded that again, so the ranks
jointly trained on 1/world of the data while every disjointness invariant still held.

A tiny fake packed-parquet dataset is produced once under ``data/_test_gpt/`` (already
covered by ``/data`` in ``.gitignore``) and reused across runs unless its params change.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from accelerate.data_loader import prepare_data_loader
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from src.data.igsm import (
    SHARD_PREFIX,
    SHARD_SUFFIX,
    VOCAB_SIZE,
    _shard_path,
    _write_shard_atomic,
)
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope
from src.train.gpt import build_streaming_dataset, make_collator, stream_epochs_available

# Window length the streaming collator asserts (gpt.py derives it from --context-length;
# small here purely for test speed). Fake shards are packed to exactly this length.
CTX = 64
# >= 4 * world_size so accelerate stays on the clean (non-padding) file-level shard path
# (data_loader.py: the `datasets.shard()` branch needs files-per-rank >= world_size).
N_SHARDS = 32
WINDOWS_PER_SHARD = 64  # -> 2048 windows; per-rank effective ~512 (>> 200 consumed in 100 steps)
_META = {'n_shards': N_SHARDS, 'wps': WINDOWS_PER_SHARD, 'ctx': CTX}
_DATA_DIR = Path('data/_test_gpt')


@pytest.fixture(scope='session')
def fake_data() -> Path:
    """Return a dir of fake packed shards, producing it once unless already present.

    Windows are unique by global index ``g`` so batch/window fingerprints never collide;
    tokens stay in ``[0, VOCAB_SIZE)`` and every window has length ``CTX``. A ``_test_meta.json``
    sentinel makes a stale dataset (params changed) regenerate.
    """
    d = _DATA_DIR
    d.mkdir(parents=True, exist_ok=True)
    shards = sorted(d.glob('batch_*.parquet'))
    meta = d / '_test_meta.json'
    if shards and meta.exists() and json.loads(meta.read_text()) == _META:
        return d
    for f in shards:  # stale / param-mismatched -> regenerate
        f.unlink()
    for s in range(N_SHARDS):
        rows = [
            [((s * WINDOWS_PER_SHARD + r) * CTX + t) % VOCAB_SIZE for t in range(CTX)]
            for r in range(WINDOWS_PER_SHARD)
        ]
        _write_shard_atomic(rows, _shard_path(d, s), require_uniform=True)
    meta.write_text(json.dumps(_META))
    return d


def _identity_collate(batch):
    """Keep raw parquet rows (plain Python lists) so a window's tokens stay comparable."""
    return batch


def test_b1_stream_is_not_pre_split(monkeypatch, fake_data):
    """The dataset keeps every shard; the rank split is accelerate's job, done once.

    ``n_shards`` is what accelerate's ``dataset.shard()`` divides, so a reintroduced
    ``files[rank::world]`` pre-slice shows up here immediately.
    """
    for rank in (0, 1, 2):
        monkeypatch.setenv('RANK', str(rank))
        monkeypatch.setenv('WORLD_SIZE', '3')
        ds, all_files = build_streaming_dataset(fake_data, seed=0, shuffle_buffer=0)
        assert len(all_files) == N_SHARDS
        assert ds.n_shards == N_SHARDS, (
            f'rank {rank} sees {ds.n_shards}/{N_SHARDS} shards: the stream was pre-split by '
            f'rank and accelerate will now split it a second time'
        )


def test_b2_workers_no_duplication(fake_data):
    """DataLoader workers within a rank stream disjoint rows -- no row appears twice.

    Workers gather the raw parquet rows with an identity collate, which is all that's needed
    to exercise HF's per-worker file sharding (``IterableDataset._iter_pytorch``) without
    pinning the test to any particular collation.
    """
    ds, all_files = build_streaming_dataset(fake_data, seed=0, shuffle_buffer=0)
    dl = DataLoader(ds, batch_size=4, num_workers=4, collate_fn=_identity_collate)
    seen = [tuple(row['input_ids']) for batch in dl for row in batch]
    expected = len(all_files) * WINDOWS_PER_SHARD  # 32 files * 64 windows = 2048 rows
    assert len(seen) == expected, f'expected {expected} rows, got {len(seen)} (loss/gain)'
    assert len(seen) == len(set(seen)), 'a row was streamed by more than one worker'


def _rank_epoch_windows(
    data_dir: Path, world: int, rank: int, *, num_workers: int = 0
) -> list[tuple[int, ...]]:
    """Windows ``rank`` trains on in one epoch, through accelerate's real dataloader wrapping.

    Passing ``num_processes``/``process_index`` explicitly needs no process group and no env
    vars; ``dispatch_batches=False`` mirrors the ``accelerator_config`` that ``train()`` pins.
    """
    ds, _ = build_streaming_dataset(data_dir, seed=0, shuffle_buffer=0)
    dl = DataLoader(ds, batch_size=2, num_workers=num_workers, collate_fn=_identity_collate)
    dl = prepare_data_loader(
        dl,
        num_processes=world,
        process_index=rank,
        put_on_device=False,
        dispatch_batches=False,
        split_batches=False,
    )
    return [tuple(row['input_ids']) for batch in dl for row in batch]


def test_b3_ddp_ranks_cover_the_whole_dataset(fake_data):
    """Three ranks together read every window exactly once (the double-shard regression).

    Disjointness alone cannot catch double sharding -- 1/world**2 slices are disjoint too --
    so this asserts *coverage*: the union over ranks must be the entire dataset.
    """
    world = 3
    counts: list[int] = []
    union: set[tuple[int, ...]] = set()
    for rank in range(world):
        windows = _rank_epoch_windows(fake_data, world, rank)
        counts.append(len(windows))
        union |= set(windows)
    total = N_SHARDS * WINDOWS_PER_SHARD
    assert sum(counts) == len(union), 'a window was streamed by two ranks'
    assert len(union) == total, (
        f'ranks jointly read {len(union)}/{total} windows ({len(union) / total:.1%}), '
        f'per-rank {counts} -- the stream is being sharded twice'
    )


def _assert_partitions_dataset(per_rank: list[list[tuple[int, ...]]], label: str) -> None:
    """Assert the per-rank streams are an exact partition of the dataset.

    Both halves have to be asserted together: disjointness alone passes under double sharding
    (1/world**2 slices are disjoint), coverage alone passes if a rank re-reads windows another
    rank already has. Failures report the offending pair, the form a broken split takes.
    """
    world = len(per_rank)
    total = N_SHARDS * WINDOWS_PER_SHARD
    counts = [len(w) for w in per_rank]
    for rank, windows in enumerate(per_rank):
        assert windows, f'{label}: rank {rank} got no data at all'
        dupes = len(windows) - len(set(windows))
        assert not dupes, f'{label}: rank {rank} streamed {dupes} window(s) twice itself'
    for a in range(world):
        for b in range(a + 1, world):
            shared = set(per_rank[a]) & set(per_rank[b])
            assert not shared, (
                f'{label}: ranks {a} and {b} share {len(shared)}/{counts[a]} window(s) -- the '
                f'stream is being copied to ranks instead of split across them'
            )
    union: set[tuple[int, ...]] = set().union(*(set(w) for w in per_rank))
    assert len(union) == total, (
        f'{label}: ranks jointly read {len(union)}/{total} windows ({len(union) / total:.1%}), '
        f'per-rank {counts} -- data is being dropped or the stream is sharded twice'
    )


@pytest.mark.parametrize('world', [2, 3], ids=['world2', 'world3'])
def test_b4_ranks_partition_the_stream_with_dataloader_workers(fake_data, world):
    """The train#3 fix in the configuration ``train.sbatch`` actually runs: an exact partition.

    ``test_b3`` pins the minimal case: the rank split alone, ``num_workers=0``. Production
    adds ``--dataloader-num-workers`` (default 4), so the shards are split *twice* -- across
    ranks by accelerate, then again across workers inside each rank by ``datasets``. ``b2``
    covers the worker split alone (single rank), ``b3`` the rank split alone (no workers);
    neither covers the composition, and only the composition is what trains.

    ``world=3`` is the sbatch config, where 32 shards do not divide evenly across ranks;
    ``world=2`` divides cleanly, a different path in ``datasets``.
    """
    per_rank = [_rank_epoch_windows(fake_data, world, r, num_workers=4) for r in range(world)]
    _assert_partitions_dataset(per_rank, f'{world} ranks x 4 workers')


def _presplit_rank_epoch_windows(data_dir: Path, world: int, rank: int) -> list[tuple[int, ...]]:
    """The qknorm ``build_streaming_dataset``: slice ``files[rank::world]`` *before* accelerate.

    Reproduced inline (rather than imported) because the fix deleted it -- this keeps the old
    behaviour pinned as a test fixture so the regression stays demonstrable.
    """
    from datasets import load_dataset

    files = sorted(data_dir.glob(f'{SHARD_PREFIX}*{SHARD_SUFFIX}'))
    ds = load_dataset(
        'parquet',
        data_files={'train': [str(f) for f in files[rank::world]]},
        split='train',
        streaming=True,
    )
    dl = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=_identity_collate)
    dl = prepare_data_loader(
        dl,
        num_processes=world,
        process_index=rank,
        put_on_device=False,
        dispatch_batches=False,
        split_batches=False,
    )
    return [tuple(row['input_ids']) for batch in dl for row in batch]


def test_b5_presplit_shards_twice_and_loses_data(fake_data):
    """Characterisation test for the bug: a manual pre-split leaves 1/world**2 per rank.

    This is what qknorm did. Every disjointness invariant still holds -- which is exactly why
    it went unnoticed -- but the ranks jointly reach only 1/world of the dataset, so the
    stream runs dry ``world`` times early and ``train/epoch`` jumps by a whole integer.
    """
    world = 3
    per_rank = [_presplit_rank_epoch_windows(fake_data, world, r) for r in range(world)]
    total = N_SHARDS * WINDOWS_PER_SHARD
    union: set[tuple[int, ...]] = set()
    for windows in per_rank:
        union |= set(windows)
    # Bounds, not exact counts: N_SHARDS need not divide world**2, so the second split lands
    # unevenly. The invariant that matters is "disjoint, yet far short of the whole dataset".
    assert sum(len(w) for w in per_rank) == len(union), 'pre-split is still disjoint per rank'
    assert len(union) <= total * 0.45, (
        f'pre-split reached {len(union)}/{total} windows ({len(union) / total:.1%}); the '
        f'double split should leave roughly 1/{world}'
    )
    for rank, windows in enumerate(per_rank):
        assert len(windows) < total / world, (
            f'rank {rank} reached {len(windows)} windows, more than the {total / world:.0f} a '
            f'single correct split would give it -- the second split did not fire'
        )


def test_b6_stream_epochs_available_flags_a_short_stream():
    """The startup preflight: <1.0 epochs is exactly the condition that wraps train/epoch.

    The two window counts are the real 3xA100 run: 52,918,515 windows on disk just covered
    100k steps, but double sharding left only 17,641,065 reachable -- a third of an epoch,
    hence the jumps to 1.0 and 2.0 in the W&B chart.
    """
    cfg = dict(world=3, per_device_batch=16, grad_accum=11, max_steps=100_000)
    assert stream_epochs_available(52_918_515, **cfg) == pytest.approx(1.002, abs=1e-3)
    assert stream_epochs_available(17_641_065, **cfg) == pytest.approx(1 / 3, abs=1e-3)


class _RecordingTrainer(Trainer):
    """Fingerprints every micro-batch's ``input_ids`` (runs in the rank's main process)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[str] = []

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ARG002
        t = inputs['input_ids'].detach().cpu().contiguous()
        self.seen.append(hashlib.sha256(t.numpy().tobytes()).hexdigest())
        return super().training_step(model, inputs, num_items_in_batch)


def _rank_fn(
    rank: int, world: int, data_dir: str, out_dir: str, port: int, dispatch: bool
) -> None:
    """One CPU DDP rank: stream its shards, train 100 steps, dump seen-batch fingerprints."""
    # Set BEFORE TrainingArguments so accelerate's PartialState() inits gloo from env.
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR='127.0.0.1',
        MASTER_PORT=str(port),
        OMP_NUM_THREADS='1',
    )
    config = build_gpt2_config(
        n_layer=2,
        n_head=2,
        n_embd=64,
        n_inner=256,
        vocab_size=VOCAB_SIZE,
        n_positions=CTX,
    )
    model = build_gpt2_rope(config, attn_implementation='sdpa')
    ds, _ = build_streaming_dataset(Path(data_dir), seed=0, shuffle_buffer=0)
    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=100,
        learning_rate=1e-4,
        logging_steps=100,
        disable_tqdm=True,
        save_strategy='no',
        eval_strategy='no',
        report_to=[],
        bf16=False,
        torch_compile=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        accelerator_config={'dispatch_batches': dispatch},
        remove_unused_columns=False,
        use_cpu=True,
        seed=0,
    )

    trainer = _RecordingTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=make_collator(CTX),
    )
    trainer.train()
    Path(out_dir, f'seen_rank{rank}.json').write_text(json.dumps(trainer.seen))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.slow
@pytest.mark.parametrize('dispatch', [False, True], ids=['no_dispatch', 'dispatch'])
def test_a_ddp_no_cross_rank_batch_overlap(tmp_path, fake_data, dispatch):
    """Over 100 DDP optimizer steps, no batch is seen by both ranks (nor repeated within one).

    Run under both ``dispatch_batches`` settings, since the two use entirely different data
    paths: off, each rank streams its own shard files (``dataset.shard``); on, rank 0 reads
    everything and scatters slices. Both must give every rank distinct batches -- what
    changed between them on qknorm was not correctness but *which* split fired, and with a
    manual pre-split in place only the off path double-shards. ``train()`` pins off.
    """
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(
        _rank_fn,
        args=(2, str(fake_data), str(tmp_path), port, dispatch),
        nprocs=2,
        join=True,
    )
    sets: list[set[str]] = []
    for rank in (0, 1):
        fps = json.loads((tmp_path / f'seen_rank{rank}.json').read_text())
        assert len(fps) == 100, (
            f'rank {rank} exhausted the stream early ({len(fps)}<100); enlarge the dataset'
        )
        assert len(fps) == len(set(fps)), f'rank {rank} has an intra-rank duplicate batch'
        sets.append(set(fps))
    assert sets[0].isdisjoint(sets[1]), 'a training batch was seen by BOTH ranks'
