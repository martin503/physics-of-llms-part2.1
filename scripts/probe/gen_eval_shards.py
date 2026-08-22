#!/usr/bin/env python3
"""Generate the per-difficulty eval shards for the Figure 7(a) probe row (iGSM-med, pq).

The paper evaluates every probe task per op count (its med columns are `op<=15`, `op=15`,
`op=20..23`) over >=4096 problem-parameter pairs per cell. This script builds one dataset
directory per op -- every problem pinned to that op, at most `--max-queries` random queries
per problem (the paper's Appendix E rule) -- so one cell is one directory and
`eval_fig7a.py` is just the loop over them:

    python scripts/probe/gen_eval_shards.py --target nece
    python scripts/probe/gen_eval_shards.py --target dep

Layout: `<out-root>/op_<NN>/` (a single parquet shard + metadata.json, `op` recorded), under
`data/probes/<target>/eval` by default. Existing shards are skipped unless `--overwrite`.

pq only: the repo's probe layouts read the question after the problem description (the pq
pretraining format); a qp-pretrained model would need different read positions and is not
supported.

dep shards sample *random* ordered pairs (`unbalanced`), the natural ~16-20% positive mix the
paper evaluates on (its dep majority-guess row is ~83-85%), not the 1:1 balance used for
probe *training*; pass `--dep-balanced` to override. nece always samples parameters at their
natural rate and handles balance at training time.

Seeds start at 0 in every shard. Disjointness needs no remote seed range: the probe training
data is generated with `--split train` (template bins 0-15) while these shards use the
default `--split test` (bins 16-22) -- the paper's own probe train/test rule -- and two
shards can never hold the same problem because a pinned op is part of a problem's identity
(problems in different shards have different op counts).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.igsm import IGSM_MED  # noqa: E402
from src.probe.data import METADATA_NAME, generate_queries_to_dir  # noqa: E402

# The paper's iGSM-med columns: op<=15 pools ops 1..15; 16-19 are skipped by the paper but
# cheap to keep available.
DEFAULT_OPS = '1-15,20-23'


def parse_ops(spec: str) -> list[int]:
    """`'1-15,20-23'` -> `[1, ..., 15, 20, ..., 23]` (inclusive ranges, comma-separated)."""
    ops: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition('-')
        ops.extend(range(int(lo), int(hi or lo) + 1))
    if not ops or len(set(ops)) != len(ops) or min(ops) < 1:
        raise argparse.ArgumentTypeError(f'bad --ops spec: {spec!r}')
    return sorted(ops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, choices=['nece', 'dep'])
    ap.add_argument('--ops', type=parse_ops, default=parse_ops(DEFAULT_OPS),
                    help=f'op counts to pin shards at, as ranges (default: {DEFAULT_OPS})')
    ap.add_argument('--problems-per-op', type=int, default=400,
                    help='problems per shard; 400 x <=10 queries ~= 4k pairs, the paper uses >=4096')
    ap.add_argument('--max-queries', type=int, default=10,
                    help='queries per problem, sampled without replacement (paper Appendix E)')
    ap.add_argument('--seed-start', type=int, default=0,
                    help='first problem seed, same for every shard (see docstring on disjointness)')
    ap.add_argument('--split', default='test', help="'test' (bins 16-22; default) or 'train'")
    ap.add_argument('--max-op', type=int, default=23,
                    help='iGSM-med difficulty cap; must cover every pinned op (default 23)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-root', type=Path, default=None,
                    help='shard root (default: data/probes/<target>/eval)')
    ap.add_argument('--dep-balanced', action='store_true',
                    help='dep: sample pairs 1:1 like training instead of the natural mix')
    ap.add_argument('--overwrite', action='store_true', help='regenerate shards that already exist')
    args = ap.parse_args()

    if args.max_op < max(args.ops):
        ap.error(f'--max-op {args.max_op} does not cover the largest requested op {max(args.ops)}')

    out_root = args.out_root or Path('data/probes') / args.target / 'eval'
    med_cfg = dict(IGSM_MED)
    med_cfg['max_op'] = args.max_op

    for op in args.ops:
        out = out_root / f'op_{op:02d}'
        if (out / METADATA_NAME).exists() and not args.overwrite:
            print(f'op {op:2d}: {out} exists, skipping (--overwrite to redo)')
            continue
        meta = generate_queries_to_dir(
            out, args.problems_per_op, target=args.target, split=args.split,
            seed_start=args.seed_start, workers=args.workers, med_cfg=med_cfg,
            max_queries=args.max_queries, unbalanced=(args.target == 'dep' and not args.dep_balanced),
            op=op, overwrite=args.overwrite,
        )
        assert meta['op'] == op

    print(f'\n{args.target} eval shards under {out_root} '
          f'(split={args.split}, <= {args.max_queries} queries/problem):')
    print(f'  {"op":>4} {"problems":>8} {"queries":>8} {"pos%":>6}  n_op unique')
    for shard in sorted(out_root.glob('op_*')):
        meta = json.loads((shard / METADATA_NAME).read_text(encoding='utf-8'))
        print(f'  {meta["op"]:>4} {meta["n_problems"]:>8} {meta["n_queries"]:>8} '
              f'{100 * meta["positive_frac"]:>6.1f}  op={meta["op"]} pinned')


if __name__ == '__main__':
    main()
