#!/usr/bin/env python3
"""Cross-LR table for the gbs64 constant-LR staircase (eq15 / le15 / eq20 vs step)."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "sweeps" / "lr_stair_results.json"
data = json.loads(RES.read_text())


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "  -  "


rows = []
for r in data:
    if "error" in r or r.get("eq15") is None:
        print(f"skip: {r['tag']} {r.get('error', '')[:60]}")
        continue
    _, s_part = r["tag"].split("_s")
    rows.append((r.get("lr", 0.0), int(s_part), r["eq15"], r["le15"], r.get("eq20")))

rows.sort(key=lambda x: (-x[0], x[1]))
print(f"\n{'lr':>8} {'step':>5} {'eq15':>7} {'le15':>7} {'eq20':>7}")
for lr, s, e, l, q in rows:
    print(f"{lr:>8.1e} {s:>5} {fmt(e):>7} {fmt(l):>7} {fmt(q):>7}")

# eq15 saturation per LR (first step with eq15>=0.9)
print("\n eq15 saturation (first step >=0.9) per LR:")
for lr in sorted({r[0] for r in rows}, reverse=True):
    ge = [s for (l, s, e, _, _) in rows if abs(l - lr) < 1e-20 and e >= 0.9]
    print(f"  lr={lr:.1e}: {'step '+str(min(ge)) if ge else 'NEVER within range'}")
