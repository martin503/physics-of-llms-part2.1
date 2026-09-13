#!/usr/bin/env python3
"""Evaluate a trained V-probe per op count and print the Figure 7(a)-style row (iGSM-med, pq).

Runs `test` (`src.probe.evaluate.evaluate_run`) on each per-op shard for `--run-dir`, then
prints accuracy, the majority-guess baseline and MCC per op, plus the pooled `op<=15` column,
next to the paper's reported values (Physics of LMs Part 2.1, arXiv 2407.20311 v1, Figure 7a,
iGSM-med pretrained-pq row; values extracted from the vector figure source) so the
reproduction can be read at a glance. Also writes `<run-dir>/fig7a_eval.json`.

    python scripts/probe/eval_fig7a.py --run-dir trained_probes/<pretrained-run>

Shards come from `gen_eval_shards.py`. The target is read from the run's config.json, the
shard root defaults to `data/probes/<target>/eval`, and the op list is whatever shards are
found there. The paper's cells average >=4096 problem-parameter pairs; a shard under that is
marked `*`. Ops 16-19 exist in our shards but not in the paper's columns (med OOD starts at
op=20) -- they print with blank paper values.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.probe.data import METADATA_NAME, git_commit  # noqa: E402
from src.probe.evaluate import classification_metrics, evaluate_run, load_predictions  # noqa: E402

PAPER_MIN_PAIRS = 4096

# Paper Figure 7(a), iGSM-med block, pretrained-pq row and majority-guess baseline row.
PAPER_MED_PQ: dict[str, dict[str, dict]] = {
    'nece': {
        'acc': {'<=15': 99.8, 15: 99.8, 20: 98.7, 21: 97.9, 22: 96.9, 23: 94.7},
        'majority': {'<=15': 74.7, 15: 54.8, 20: 50.1, 21: 50.4, 22: 51.4, 23: 52.1},
    },
    'dep': {
        'acc': {'<=15': 99.7, 15: 99.3, 20: 100.0, 21: 100.0, 22: 100.0, 23: 100.0},
        'majority': {'<=15': 84.6, 15: 82.2, 20: 83.0, 21: 83.5, 22: 83.5, 23: 83.7},
    },
}


def discover_shards(root: Path) -> list[tuple[int, Path]]:
    """`(op, dir)` for every pinned-op shard under `root`, by its metadata (dir name as fallback)."""
    shards: list[tuple[int, Path]] = []
    for d in sorted(root.glob('op_*')):
        if not (d / METADATA_NAME).exists():
            continue
        meta = json.loads((d / METADATA_NAME).read_text(encoding='utf-8'))
        op = meta['op'] if meta.get('op') is not None else int(d.name.removeprefix('op_'))
        shards.append((int(op), d))
    if not shards:
        raise FileNotFoundError(
            f'no pinned-op shards under {root} -- run scripts/probe/gen_eval_shards.py first'
        )
    return sorted(shards)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', type=Path, required=True, help='trained probe run dir (config.json + probe.pt)')
    ap.add_argument('--shards-root', type=Path, default=None,
                    help='pinned-op shard root (default: data/probes/<target>/eval)')
    ap.add_argument('--pool-max', type=int, default=15,
                    help="pool ops up to this value into the paper's op<=15 column")
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    params = json.loads((args.run_dir / 'config.json').read_text(encoding='utf-8'))['params']
    target = params['target']
    shards_root = args.shards_root or Path('data/probes') / target / 'eval'
    shards = discover_shards(shards_root)

    rows: list[dict] = []
    pooled: list[dict[str, np.ndarray]] = []
    for op, shard in shards:
        metrics = evaluate_run(args.run_dir, shard, batch_size=args.batch_size, device=args.device)
        rows.append({
            'op': op, 'shard': str(shard),
            **{k: metrics[k] for k in ('n', 'acc', 'acc_majority', 'mcc', 'positive_frac')},
        })
        if op <= args.pool_max:
            pooled.append(load_predictions(args.run_dir, shard))

    pool_row = None
    if pooled:
        labels = np.concatenate([p['label'] for p in pooled])
        preds = np.concatenate([p['pred'] for p in pooled])
        pool_row = {'op': '<=15', **classification_metrics(labels, preds)}

    paper_acc, paper_maj = PAPER_MED_PQ[target]['acc'], PAPER_MED_PQ[target]['majority']

    def fmt(name: object, m: dict) -> str:
        pa, pm = paper_acc.get(name), paper_maj.get(name)  # '<=15' str key or int op key
        star = '*' if m['n'] < PAPER_MIN_PAIRS else ''
        return (f"{str(name):>5} {m['n']:>8,d} {100 * m['acc']:>7.1f} {100 * m['acc_majority']:>7.1f}"
                f" {m['mcc']:>7.3f}     {f'{pa:>7.1f}' if pa is not None else '       -'}"
                f" {f'{pm:>7.1f}' if pm is not None else '       -'}{star}")

    print(f'\nfig7a eval   target={target}   run={args.run_dir}   shards={len(shards)}'
          f'   ops={[op for op, _ in shards]}')
    print(f'{"op":>5} {"n":>8} {"acc%":>7} {"maj%":>7} {"mcc":>7}  | {"paper%":>7} {"paper-maj%":>9}')
    if pool_row:
        print(fmt('<=15', pool_row))
    for row in rows:
        print(fmt(row['op'], row))
    print(f'(* fewer than the paper\'s {PAPER_MIN_PAIRS} pairs/cell; ops 16-19 have no paper column)')

    out = {
        'kind': 'fig7a_eval',
        'run_dir': str(args.run_dir),
        'target': target,
        'shards_root': str(shards_root),
        'pool_max': args.pool_max,
        'pooled': pool_row,
        'per_op': rows,
        'paper_med_pq': PAPER_MED_PQ[target],
        'repo_commit': git_commit(Path.cwd()),
        'created': datetime.now(UTC).isoformat(timespec='seconds'),
    }
    path = args.run_dir / 'fig7a_eval.json'
    path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
