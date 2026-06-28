"""Streaming iGSM-med generation for on-the-fly TRL packing.

Companion to :mod:`src.data.igsm`, which generates and packs problems **offline** to
parquet shards. This module instead yields raw iGSM problems lazily as a
``datasets.IterableDataset`` so they can be packed **during training** by TRL (see
:mod:`src.train.gpt_pack`). Each problem is the iGSM token layout
``[222] prob [223] sol [224] ans [50256]`` produced directly by the official generator
(no re-tokenization -- the generator already emits GPT-2 BPE ids).

Sharding (the streaming + multi-worker correctness rule)
---------------------------------------------------------
iGSM uses **module-level** ``random`` / ``numpy`` RNG (``fix_seed`` sets both, not
torch), so a single process can sustain only ONE distinct evolving stream per seed.
We therefore expose exactly one list-valued gen kwarg, ``seed_offsets``:

* its length is the dataset ``num_shards`` (``datasets`` sharding util counts list
  kwargs), so the shards are split **disjointly** across DDP ranks (accelerate's
  ``dataset.shard``) **and** across DataLoader workers (``IterableDataset._iter_pytorch``);
* each (rank, worker) receives a disjoint sub-list of seeds and consumes each seed's
  infinite stream in turn -> distinct problems everywhere, full parallelism, no
  duplicates. If ``num_shards < num_workers`` some workers idle (datasets warns), so
  callers must size ``seed_offsets`` >= ``world_size * num_workers`` (see
  :func:`seed_offsets_for_shards`).

Example::

    ds = build_igsm_stream([0, 1, 2, 3], split='train')   # 4 shards
    for ex in ds:
        ...  # ex == {'input_ids': [...]}
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from datasets import IterableDataset

from src.data.igsm import EOS, IGSM_MED, PROB_BOS, ensure_igsm_submodule, get_bins


def _igsm_generator(
    seed_offsets: list[int],
    split: str = 'train',
    med_cfg: dict[str, Any] | None = None,
) -> Iterator[dict[str, list[int]]]:
    """Yield iGSM-med problems forever, one distinct stream per seed in ``seed_offsets``.

    Args:
        seed_offsets: The disjoint sub-list of seeds assigned to this (rank, worker)
            by datasets sharding. Each seed seeds ``fix_seed`` once -> one independent
            infinite stream.
        split: ``'train'`` (bins 0-15) or ``'test'`` (bins 16-22).
        med_cfg: iGSM-med config (defaults to :data:`src.data.igsm.IGSM_MED`).

    Yields:
        ``{'input_ids': [int, ...]}`` -- one problem's tokens per yield.
    """
    assert seed_offsets, 'seed_offsets is empty -- this shard has no problems to generate'
    ensure_igsm_submodule()
    from data_gen.pretrain.id_gen import IdGen  # type: ignore[import-not-found]
    from tools.tools import fix_seed  # type: ignore[import-not-found]

    cfg = IGSM_MED if med_cfg is None else med_cfg
    bins = get_bins(split)
    for seed in seed_offsets:
        fix_seed(seed)  # seeds module-level random + numpy (NOT torch) -> distinct stream
        gen = IdGen(**cfg)  # gen_prob mutates the instance, so one generator per seed
        while True:
            gen.gen_prob(bins, p_format='pq')  # type: ignore[attr-defined]
            token_id = list(gen.token_id)  # type: ignore[attr-defined]
            assert token_id and token_id[0] == PROB_BOS and token_id[-1] == EOS, (
                'unexpected iGSM token layout'
            )
            yield {'input_ids': token_id}


def build_igsm_stream(
    seed_offsets: list[int], *, split: str = 'train', med_cfg: dict[str, Any] | None = None
) -> IterableDataset:
    """Build a streaming iGSM dataset.

    ``seed_offsets`` MUST be the only list-valued gen kwarg: a second list of a
    different length makes sharding ambiguous and raises (datasets sharding util).
    Its length drives ``num_shards`` -- size it with :func:`seed_offsets_for_shards`.
    """
    assert seed_offsets, 'seed_offsets must be non-empty'
    return IterableDataset.from_generator(
        _igsm_generator,
        gen_kwargs={
            'seed_offsets': list(seed_offsets),
            'split': split,
            'med_cfg': med_cfg,
        },
    )


def seed_offsets_for_shards(world_size: int, num_workers: int, base_seed: int = 0) -> list[int]:
    """Return ``>= world_size * num_workers`` seed offsets, rounded to a multiple of
    ``world_size`` so every DDP rank gets a clean ``.shard()`` (no worker idle warning).

    Args:
        world_size: Number of DDP processes (1 if not distributed).
        num_workers: DataLoader ``num_workers`` (treat 0 as 1 -- the main process is the
            single consumer).
        base_seed: First seed offset; the rest are ``base_seed + 1, ...``.
    """
    assert world_size >= 1
    workers = max(num_workers, 1)
    n = max(world_size * workers, world_size)
    n = ((n + world_size - 1) // world_size) * world_size  # round up to a multiple of world_size
    return list(range(base_seed, base_seed + n))
