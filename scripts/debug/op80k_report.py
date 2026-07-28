#!/usr/bin/env python3
"""Report for the 80k-continue run: step | eq15 | le15 | eq20 table + verdict.

Key question: does continuing from the PLASTIC 80k checkpoint (vs SFT on the
converged 100k) move eq20 off 0? le15/eq15 should stay high on op<=15 data.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RES = REPO / (sys.argv[1] if len(sys.argv) > 1 else 'sweeps/op80k_results.json')
data = json.loads(RES.read_text())


def fmt(v):
    return f'{v:.3f}' if isinstance(v, (int, float)) else '  -  '


rows = []
for r in data:
    if 'error' in r or r.get('eq15') is None:
        print(f'skip: {r["tag"]} {r.get("error", "")[:60]}')
        continue
    _, s_part = r['tag'].split('_s')
    rows.append((int(s_part), r['eq15'], r['le15'], r.get('eq20')))

rows.sort()
print(f"\n{'step':>6} {'eq15':>7} {'le15':>7} {'eq20':>7}")
for s, e, l, q in rows:
    print(f'{s:>6} {fmt(e):>7} {fmt(l):>7} {fmt(q):>7}')

print('\n baselines: base_100k le15=0.99/eq15=0.32/eq20=0;'
      ' op1-15 SFT (from 100k) le15~0.98/eq15~1.0/eq20=0')
best_le15 = max((l for _, _, l, _ in rows), default=None)
best_eq15 = max((e for _, e, _, _ in rows), default=None)
eq20_max = max((q for *_, q in rows if q is not None), default=None)
print(f' best le15={fmt(best_le15)}  best eq15={fmt(best_eq15)}  max eq20={fmt(eq20_max)}')
if best_le15 is not None and best_eq15 is not None and best_le15 >= 0.95 and best_eq15 >= 0.95:
    same = next((s for s, e, l, _ in rows if e >= 0.95 and l >= 0.95), None)
    print(f' -> BOTH le15>=0.95 and eq15>=0.95 at step {same}.')
else:
    print(' -> not both >=0.95 at any checkpoint.')
print(f' eq20 max={fmt(eq20_max)} -> '
      f'{"MOVED off 0 (plasticity hypothesis supported!)" if eq20_max and eq20_max > 0.02 else "still ~0 (op>15 data required; checkpoint plasticity alone insufficient)"}')
