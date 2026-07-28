#!/usr/bin/env python3
"""Eval the 80k-continue ctx768/gbs512/fa2 run checkpoints (500..10000 + final) on le15 + eq15 + eq20.

Re-run of the plasticity experiment: continue-train from the PLASTIC 80k checkpoint at ctx768,
gbs512, fa2 (pure bf16), lr2e-4, cosine floored at 7e-5, full 10k, save every 500. Key signal is
eq20 across the EARLY checkpoints (500/1000/1500) where the transient peak is expected to land at
393k tok/step. Reuses sweep_eval.run_eval (2-GPU pool, resumable).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_eval import run_eval

REPO = Path(__file__).resolve().parents[1]
TOK = 393216  # gbs512 per-step tokens at ctx768 (512 * 768)
RUN_REL = sys.argv[1] if len(sys.argv) > 1 else 'sweeps/disc_80k_op1_15_lr2e4_gbs512_ctx768_fa2'
RESULTS_REL = sys.argv[2] if len(sys.argv) > 2 else 'sweeps/op80k768_results.json'
run = REPO / RUN_REL

rows: dict[int, str] = {}
for d in sorted(run.glob('checkpoint-*')):
    try:
        step = int(d.name.split('-', 1)[1])
    except ValueError:
        continue
    rows[step] = f'{RUN_REL}/{d.name}'
if (run / 'final').exists():
    ep = max(rows) if rows else 0
    rows[ep] = f'{RUN_REL}/final'  # Trainer also auto-saves checkpoint-{max_steps}; dedup via dict

targets, tokens_map = [], {}
for step, p in sorted(rows.items()):
    t = f'op80k768_s{step}'
    targets.append((t, p))
    tokens_map[t] = step * TOK

lr_map = {t: 2e-4 for t, _ in targets}
print(f'op80k768 targets: {len(targets)} -> {[t for t, _ in targets]}')
run_eval(targets, ['med_pq_op_eq15', 'med_pq_op_le15', 'med_pq_op_eq20'],
         REPO / RESULTS_REL,
         tokens_map=tokens_map, lr_map=lr_map, limit=128, batch_size=64)
