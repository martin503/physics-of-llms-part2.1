#!/usr/bin/env python3
"""Compare batching strategies on the two axes that trade off against each other.

`_length_bucketed_batches` sorts queries by length so a batch pads to near its own mean
instead of the dataset maximum. The sort is stable, so without a pre-shuffle one problem's
near-equal-length queries stay adjacent and fill a batch together -- correlated queries the
optimiser then sees as one step. This measures both effects at once:

* distinct problems per batch -- how close the batch is to an independent draw (ceiling =
  batch size),
* padded tokens per batch -- ``batch_size * max_len``, the memory the batch actually costs.

Queries are synthetic, with the length statistics of real `dep` data: each problem sits at
its own base length, its queries spread ~8 tokens around it. Usage::

    uv run python scripts/debug/batch_composition.py [n_problems] [queries_per_problem]

Defaults are 300 problems at the `--max-queries` cap of 10.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.probe.build_queries import ProbeQuery  # noqa: E402
from src.probe.vprobe_train import _length_bucketed_batches  # noqa: E402

n_problems = int(sys.argv[1]) if len(sys.argv) > 1 else 300
per_problem = int(sys.argv[2]) if len(sys.argv) > 2 else 10
batch_size = 8

rng = np.random.default_rng(0)
queries = [
    ProbeQuery(list(range(base + int(rng.integers(0, 8)))), label=0, group=problem)
    for problem, base in enumerate(rng.integers(46, 100, size=n_problems))
    for _ in range(per_problem)
]
indices = np.arange(len(queries))


def no_bucketing(idx: np.ndarray) -> list[list[int]]:
    """Naive shuffle: independent batches, padded to the worst length each happens to hold."""
    order = np.random.default_rng(1).permutation(idx)
    return [list(order[s : s + batch_size]) for s in range(0, len(order), batch_size)]


def stable_sort(idx: np.ndarray) -> list[list[int]]:
    """Length sort with no pre-shuffle: ties keep dataset order, so problems clump."""
    order = sorted(idx, key=lambda i: len(queries[i].input_ids))
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


def pre_shuffled(idx: np.ndarray) -> list[list[int]]:
    """What training uses: the same sort over shuffled indices, so ties break randomly."""
    batches = _length_bucketed_batches(queries, idx, batch_size, np.random.default_rng(1))
    lookup = {id(q): i for i, q in enumerate(queries)}
    return [[lookup[id(q)] for q in batch] for batch in batches]


print(f'{n_problems} problems x {per_problem} queries = {len(queries)} queries, '
      f'batch size {batch_size}')
print(f'{"strategy":<14} {"distinct/batch":>14} {"tokens/batch":>13} {"padding waste":>14}')
for name, build in (('no bucketing', no_bucketing), ('stable sort', stable_sort),
                    ('pre-shuffled', pre_shuffled)):
    batches = build(indices)
    lengths = [[len(queries[i].input_ids) for i in batch] for batch in batches]
    distinct = np.mean([len({queries[i].group for i in batch}) for batch in batches])
    padded = np.mean([len(b) * max(b) for b in lengths])
    real = np.mean([sum(b) for b in lengths])
    print(f'{name:<14} {distinct:>14.2f} {padded:>13.0f} {1 - real / padded:>13.1%}')
