"""Data-pipeline tests for ``src/train/gpt.py`` (streaming DDP path).

Covers the two multi-GPU data properties of ``build_streaming_dataset``:

* (fast, in-process) each DDP rank takes a disjoint shard-file slice, and DataLoader
  workers within a rank stream disjoint rows -- no duplication across ranks or workers.
* (slow, 2-rank CPU DDP) over 100 optimizer steps no training batch is seen by both ranks.

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
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from src.data.igsm import VOCAB_SIZE, _shard_path, _write_shard_atomic
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope
from src.train.gpt import build_streaming_dataset, make_collator

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


def test_b1_rank_files_disjoint(monkeypatch, fake_data):
    """Each rank's ``files[rank::world]`` slice is disjoint and covers all shards."""
    per_rank: list[set[Path]] = []
    union: set[Path] = set()
    for rank in (0, 1):
        monkeypatch.setenv('RANK', str(rank))
        monkeypatch.setenv('WORLD_SIZE', '2')
        _, all_files, rank_files = build_streaming_dataset(fake_data, seed=0, shuffle_buffer=0)
        rf = set(rank_files)
        assert len(rf) >= 2, 'need >=2 files/rank to stay off the padding shard path'
        per_rank.append(rf)
        union |= rf
    assert per_rank[0].isdisjoint(per_rank[1]), 'a shard file is assigned to BOTH ranks'
    assert union == set(all_files), 'rank slices do not cover every shard (data loss)'


def test_b2_workers_no_duplication(monkeypatch, fake_data):
    """DataLoader workers within a rank stream disjoint rows -- no row appears twice.

    Workers gather the raw parquet rows (plain Python lists) with an identity collate, which is
    all that's needed to exercise HF's per-worker file sharding (``IterableDataset._iter_pytorch``)
    without pinning the test to any particular collation.
    """
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    ds, _, rank_files = build_streaming_dataset(fake_data, seed=0, shuffle_buffer=0)
    dl = DataLoader(ds, batch_size=4, num_workers=4, collate_fn=lambda batch: batch)
    seen = [tuple(row['input_ids']) for batch in dl for row in batch]
    expected = len(rank_files) * WINDOWS_PER_SHARD  # 16 files * 64 windows = 1024 rows
    assert len(seen) == expected, f'expected {expected} rows, got {len(seen)} (loss/gain)'
    assert len(seen) == len(set(seen)), 'a row was streamed by more than one worker'


class _RecordingTrainer(Trainer):
    """Fingerprints every micro-batch's ``input_ids`` (runs in the rank's main process)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seen: list[str] = []

    def training_step(self, model, inputs, num_items_in_batch=None):  # noqa: ARG002
        t = inputs['input_ids'].detach().cpu().contiguous()
        self.seen.append(hashlib.sha256(t.numpy().tobytes()).hexdigest())
        return super().training_step(model, inputs, num_items_in_batch)


def _rank_fn(rank: int, world: int, data_dir: str, out_dir: str, port: int) -> None:
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
    ds, _, _ = build_streaming_dataset(Path(data_dir), seed=0, shuffle_buffer=0)
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
        accelerator_config={'dispatch_batches': False},
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
def test_a_ddp_no_cross_rank_batch_overlap(tmp_path, fake_data):
    """Over 100 DDP optimizer steps, no batch is seen by both ranks (nor repeated within one)."""
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(
        _rank_fn,
        args=(2, str(fake_data), str(tmp_path), port),
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
