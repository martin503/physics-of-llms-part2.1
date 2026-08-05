#!/usr/bin/env python3
"""Print one regenerated iGSM problem next to the parameters the probe may be asked about.

The dep(A, B) report identifies a problem by its *seed*, which alone is not enough to get
it back: the problem only reproduces under the same `med_cfg` the dataset was generated
with. Regenerating seed 1515006390 under the default `max_op=15` yields an unrelated
2-operation problem. Hence `--data`, which reads split and med_cfg from a dataset's
`metadata.json`.

Two problems worth keeping around, both with a node that is isolated in the structure
graph and therefore mentioned by no sentence at all:

    python scripts/debug/show_probe_problem.py 1515006390 --data data/probe/vprobe_dep_10x23

        The text is about Moray Eels and establishes 3 variables, but Crab is a second
        creature with no organs, so `all_param` also offers "each Crab's Elbow Joint",
        "each Crab's Biceps" and "each Crab's Organs" -- 6 candidates for a 3-variable
        problem. This is the problem shown in the published report.

    python scripts/debug/show_probe_problem.py 57 --split test

        Green Field Elementary is isolated too, but here the *question* asks about it
        ("How many Classroom does Green Field Elementary have?", answer 0), so the two
        cases need different treatment: 12 candidates, 4 of them for that school.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.probe.labels import regenerate_problem  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('seed', type=int, help='problem seed (the report calls it the group id)')
    ap.add_argument('--split', default='test', help="'test' (default) or 'train'")
    ap.add_argument('--max-op', type=int, default=None, help='iGSM-med difficulty cap override')
    ap.add_argument('--max-edge', type=int, default=None, help='iGSM-med graph-width cap override')
    ap.add_argument(
        '--data', type=Path, default=None,
        help='probe dataset dir: read split/med_cfg from its metadata.json instead',
    )
    args = ap.parse_args()

    med_cfg = None
    split = args.split
    if args.data is not None:
        meta = json.loads((args.data / 'metadata.json').read_text(encoding='utf-8'))
        med_cfg, split = meta.get('med_cfg'), meta.get('split', split)
    elif args.max_op is not None or args.max_edge is not None:
        from src.data.igsm import IGSM_MED

        med_cfg = dict(IGSM_MED)
        if args.max_op is not None:
            med_cfg['max_op'] = args.max_op
        if args.max_edge is not None:
            med_cfg['max_edge'] = args.max_edge

    p = regenerate_problem(args.seed, split=split, med_cfg=med_cfg).problem

    print(f'seed {args.seed}   split {split}   med_cfg {med_cfg or "IGSM_MED (default)"}')
    print(f'n_op {p.n_op}   depth {p.d}   layer widths {list(p.l)}   answer {p.ans}')
    for i, layer in enumerate(p.N):
        print(f'  layer {i} ({p.ln[i]}): {list(layer)}')

    print('\nSTRUCTURE GRAPH')
    for i, g in enumerate(p.G):
        for j, row in enumerate(g):
            kids = [p.N[i + 1][k] for k, has in enumerate(row) if has]
            print(f'  {p.N[i][j]!r:45s} -> {kids if kids else "(no children -- isolated)"}')

    print('\nPROBLEM DESCRIPTION')
    for s in p.problem[:-1]:
        print(f'  {s}.')
    print('\nQUESTION')
    print(f'  {p.problem[-1]}')
    print('\nSOLUTION')
    for s in p.solution:
        print(f'  {s}.')

    print(f'\nCANDIDATE PARAMETERS ({len(p.all_param)} offered by iGSM)')
    for param in p.all_param:
        print(f'  {str(param):<14}  {p.get_param(param)}')


if __name__ == '__main__':
    main()
