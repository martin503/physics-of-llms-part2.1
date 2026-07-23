# iGSM data generation: pipeline, ordering, and difficulty

How the packed training data is produced and laid out on disk, and what that implies for the
training-time data configuration. Source: [`src/data/igsm.py`](../src/data/igsm.py),
[`src/data/merge.py`](../src/data/merge.py), and the iGSM submodule
[`iGSM/data_gen/prototype/id_gen.py`](../iGSM/data_gen/prototype/id_gen.py).

## Takeaways

- Op count (difficulty) is drawn **independently per problem** — no same-difficulty runs.
- Shard *order* on disk is still the order problems were generated, so a final
  [`src/data/merge.py`](../src/data/merge.py) pass row-shuffles across shard groups.
- After that merge the data is i.i.d. enough to train with **`--shuffle-buffer 0`** (off).

## Pipeline

```
src.data.igsm generate  ->  batch_*.parquet   (packed 768-token windows)
src.data.merge          ->  batch_*.parquet   (row-shuffled, fewer/larger shards)
src.train.gpt --streaming --data-dir <merged>
```

### 1. Generate

[`generate_to_dir`](../src/data/igsm.py) produces `num_problems` problems in **batches**;
one batch becomes exactly one parquet shard (`batch_NNNNNN.parquet`), written atomically so
generation is resumable. Each batch is split into many fine **units** (auto-sized by
[`_fine_unit`](../src/data/igsm.py) to ~8 units per worker, capped at 256 problems) and all
units are submitted to one persistent `ProcessPoolExecutor` at once, so the pool dynamically
rebalances around `gen_prob`'s heavy-tailed rejection sampling (~22 retries/problem).

With `--pack` (default) problems are concatenated and chunked into fixed
`context_length`-token windows¹, one row per window; the partial tail of each batch is
dropped. With `--raw`, one variable-length row per problem is written instead, for online
packing at train time.

¹_[`DEFAULT_CONTEXT_LENGTH`](../src/data/igsm.py)` = 768` determines the window size._

**Seeding.** Batch `b`, unit `j` uses `seed + b*stride + j`, where `stride` = units per batch.
Seeds are therefore globally unique, which is what makes generation resumable and
duplicate-free across interrupts.

### 2. Merge and shuffle

[`merge_shards`](../src/data/merge.py) groups `--n` input shards at a time, shuffles every row
across the group, and writes one output shard per group (`ceil(num_shards / n)` shards total).
Only `n` shards are held in memory at once, so this stays cheap at any dataset size. The output
keeps the same `batch_*.parquet` layout, so the merged directory is a drop-in `--data-dir`.

```bash
uv run python -m src.data.merge --in data/igsm_train --out data/igsm_train_merged --n 4 --seed 0
```

## Difficulty

Difficulty in iGSM-med varies almost entirely along one axis: the **operation count** (`op`).
The other med knobs are fixed (`perm_level=5` → full shuffle, `detail_level=0` → one verbosity),
so op count is the only difficulty that varies.

[`_generate_chunk`](../src/data/igsm.py) creates a **fresh `IdGen` per problem**. `op_` is
sampled once in `IdGen.__init__`, so re-instantiating re-draws it and every problem gets an
independent op count. `gen_prob` then rejection-samples until it finds a problem matching that
target ([`id_gen.py:241`](../iGSM/data_gen/prototype/id_gen.py)):

```python
if self.problem.n_op != self.op_ or self.problem.to_hash() not in ava_hash:
    continue        # keep trying until n_op == self.op_
```

Construction is cheap (attribute assignment + O(1) RNG draws); the cost is in the rejection
loop, which runs once per problem regardless.

### Distribution

| `generate` flag | Resulting op distribution |
|---|---|
| *(none)* | Natural iGSM skew: `min` of two draws from `uniform[1, max_op]` (the `'light'` style), skewed toward low op counts |
| `--op K` | Every problem pinned to exactly `K` reasoning steps |
| `--op-range LO-HI` | Uniform over the inclusive range, assigned per unit by round-robin over the global unit index |

`--op` and `--op-range` are mutually exclusive, and `max_op` is auto-raised to cover whichever
is requested. See [`dataset_difficulty.md`](dataset_difficulty.md) for measured distributions.

## On-disk structure

```
shard b (= batch b):
  [ unit 0 ][ unit 1 ] ... [ unit U-1 ]     # op drawn per problem within every unit
concatenated, then chunked into 768-token windows (partial tail dropped per batch)
```

Within a shard, difficulty varies problem to problem. Across shards, the only ordering is
`sorted()` on shard **filenames** in the loaders — never on content. The merge pass in step 2
is what removes the residual generation-order correlation between shards.

## Training-time configuration

Because the merged dataset is already row-shuffled offline, `src/train/gpt.py` defaults to
`--shuffle-buffer 0` (shuffling off) and streams shards in filename order:

```bash
uv run python -m src.train.gpt --streaming --data-dir data/igsm_train_merged
```

Passing `--shuffle-buffer N` re-enables `IterableDataset.shuffle(seed, buffer_size)`, which
shuffles **both** shard order and an N-window sample buffer. That is only needed for a
dataset that has not been through the merge pass.

Shards are split across DDP ranks by file (`files[rank::world]`), then sub-sharded across
DataLoader workers, so generate at least `world_size × dataloader_num_workers` shards — and
ideally a multiple of `world_size`, so no rank carries an extra shard.
