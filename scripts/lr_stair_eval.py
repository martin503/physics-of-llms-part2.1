#!/usr/bin/env python3
"""Eval the gbs64 constant-LR staircase checkpoints on eq15 + le15 + eq20.

Builds one combined target list across all run dirs (tags lr{X}_s{N}) so the 2-GPU
pool is shared across the whole staircase (no inter-run idle). Reuses sweep_eval.run_eval.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_eval import run_eval

REPO = Path(__file__).resolve().parents[1]
TOK = 49152  # gbs64 per-step tokens (64 * 768)
RUNS = [
    ("8e6", 8e-6, "sweeps/disc_gbs64_lr8e6"),
    ("6e6", 6e-6, "sweeps/disc_gbs64_lr6e6"),
    ("4e6", 4e-6, "sweeps/disc_gbs64_lr4e6"),
    ("2e6", 2e-6, "sweeps/disc_gbs64_lr2e6"),
    ("1e6", 1e-6, "sweeps/disc_gbs64_lr1e6"),
    ("5e7", 5e-7, "sweeps/disc_gbs64_lr5e7"),
]

targets, tokens_map, lr_map = [], {}, {}
for tag, lr, run_rel in RUNS:
    run = REPO / run_rel
    rows: dict[int, str] = {}
    for d in sorted(run.glob("checkpoint-*")):
        try:
            step = int(d.name.split("-", 1)[1])
        except ValueError:
            continue
        rows[step] = f"{run_rel}/{d.name}"
    if (run / "final").exists():
        ep = max(rows) if rows else 0
        rows[ep] = f"{run_rel}/final"  # manual final at the run endpoint
    for step, p in sorted(rows.items()):
        t = f"lr{tag}_s{step}"
        targets.append((t, p))
        tokens_map[t] = step * TOK
        lr_map[t] = lr

print(f"staircase targets: {len(targets)} -> {[t for t,_ in targets]}")
run_eval(targets, ["med_pq_op_eq15", "med_pq_op_le15", "med_pq_op_eq20"],
         REPO / "sweeps/lr_stair_results.json",
         tokens_map=tokens_map, lr_map=lr_map, limit=128, batch_size=64)
