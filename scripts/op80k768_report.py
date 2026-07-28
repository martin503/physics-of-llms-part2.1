#!/usr/bin/env python3
"""Report for the 80k-continue ctx768/gbs512/fa2 run: step | eq15 | le15 | eq20 table + verdict.

Re-run of the plasticity experiment at ctx768/gbs512/fa2 (full 10k, save 500). Key question: does
the eq20 transient (0.119@step1000 in the gbs256/ctx2048 run) reappear / shift / stabilize here,
and is it captured by the early checkpoints (500/1000/1500) where it should peak at 393k tok/step?
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / (sys.argv[1] if len(sys.argv) > 1 else 'sweeps/op80k768_results.json')
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

print('\n baselines: base_80k le15=0.977/eq15=0.000/eq20=0;'
      ' ctx2048 gbs256 run: step1000 eq20=0.119 (peak), step2000 eq20=0 (over-fit)')
best_le15 = max((l for _, _, l, _ in rows), default=None)
best_eq15 = max((e for _, e, _, _ in rows), default=None)
eq20_max = max((q for *_, q in rows if q is not None), default=None)
eq20_peak_step = next((s for s, *_, q in rows if q == eq20_max and q is not None), None)
print(f' best le15={fmt(best_le15)}  best eq15={fmt(best_eq15)}  max eq20={fmt(eq20_max)} @ step {eq20_peak_step}')
if best_le15 is not None and best_eq15 is not None and best_le15 >= 0.95 and best_eq15 >= 0.95:
    same = next((s for s, e, l, _ in rows if e >= 0.95 and l >= 0.95), None)
    print(f' -> BOTH le15>=0.95 and eq15>=0.95 at step {same}.')
else:
    print(' -> not both >=0.95 at any checkpoint.')
ctx2048_peak = 0.119
print(f' eq20 max={fmt(eq20_max)} (ctx2048 gbs256 peak was {ctx2048_peak:.3f}) -> '
      f'{"ABOVE ctx2048 peak" if eq20_max and eq20_max > ctx2048_peak else "at/below ctx2048 peak"}; '
      f'{"MOVED off 0" if eq20_max and eq20_max > 0.02 else "still ~0"}')
