"""Run the iGSM-med Figure-3 eval over precomputed parquet slices.

Pipeline: load slices -> torch-generate all prompts (batched) -> score each generation with
iGSM's ``true_correct`` via inspect_ai -> aggregate accuracy per slice (the Figure-3 bars) ->
optionally log to wandb.

The 7 med slices per format, in Figure-3 x-axis order:
``op_le15, op_eq15, op_eq20, op_eq21, op_eq22, op_eq23, reask``.

Usage::

    # smoke (one slice, few problems)
    uv run python -m src.eval.run --model models/gpt2-rope-igsm-pack/final \\
        --slices med_pq_op_eq20 --limit 32
    # full Figure 3 for the med/pq model
    uv run python -m src.eval.run --model models/gpt2-rope-igsm-pack/final --p-format pq --wandb
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from inspect_ai import eval as ia_eval

from src.data.igsm import ensure_igsm_submodule
from src.eval.generate import IgsmGenerator
from src.eval.task import build_eval_task, load_slices

# Figure-3 x-axis order (med), per format.
MED_SLICES: tuple[str, ...] = (
    'op_le15',
    'op_eq15',
    'op_eq20',
    'op_eq21',
    'op_eq22',
    'op_eq23',
    'reask',
)

app = typer.Typer(add_completion=False, help='iGSM-med Figure-3 eval (inspect_ai + torch).')


@app.command()
def run(
    model: Annotated[
        Path, typer.Option('--model', help='HF checkpoint dir with GPT2LMHeadModelWithRoPE.')
    ],
    data_root: Annotated[
        Path, typer.Option('--data-root', help='Dir with med_<fmt>_<spec>.parquet slices.')
    ] = Path('data/igsm_eval'),
    p_format: Annotated[
        str, typer.Option('--p-format', help="'pq' or 'qp' (which trained model family).")
    ] = 'pq',
    slices: Annotated[
        str | None,
        typer.Option('--slices', help='Comma list; default = all 7 med slices for --p-format.'),
    ] = None,
    limit: Annotated[
        int | None, typer.Option('--limit', help='Per-slice cap for smoke runs (None = all).')
    ] = None,
    device: Annotated[str, typer.Option('--device')] = 'cuda',
    batch_size: Annotated[int, typer.Option('--batch-size')] = 64,
    max_new_tokens: Annotated[
        int, typer.Option('--max-new-tokens', help='Generation cap (paper eval context = 2048).')
    ] = 2048,
    log_dir: Annotated[Path, typer.Option('--log-dir')] = Path('logs/eval'),
    log_samples: Annotated[
        bool,
        typer.Option('--log-samples/--no-log-samples', help='Per-sample results in the eval log.'),
    ] = True,
    wandb: Annotated[
        bool, typer.Option('--wandb/--no-wandb', help='Log the Figure-3 table to wandb.')
    ] = False,
    wandb_name: Annotated[
        str | None, typer.Option('--wandb-name', help='wandb run name (default <ckpt-stem>).')
    ] = None,
) -> None:
    """Generate, score (true_correct), and aggregate accuracy per slice for Figure 3."""
    ensure_igsm_submodule()  # before any pickle.loads inside the scorer
    names = (
        [s.strip() for s in slices.split(',')]
        if slices
        else [f'med_{p_format}_{spec}' for spec in MED_SLICES]
    )
    rows = load_slices(data_root, names, limit)
    assert rows, f'no eval rows loaded from {data_root} for {names}'
    typer.echo(f'loaded {len(rows)} problems across {len(names)} slice(s)')

    generator = IgsmGenerator(str(model), device=device)
    gens = generator.batch_generate([r['prompt_ids'] for r in rows], batch_size, max_new_tokens)
    assert len(gens) == len(rows), 'generation count mismatch'
    for r, g in zip(rows, gens, strict=True):
        r['generated_ids'] = g

    task = build_eval_task(generator, rows)
    log = ia_eval(
        task,
        log_dir=str(log_dir),
        log_format='json',
        display='none',
        log_samples=log_samples,
    )[0]
    typer.echo(f'inspect_ai status={log.status}; log={log.location}')

    acc = _per_slice_accuracy(log, names)
    _print_table(acc, names)

    if wandb:
        from src.eval.wandb_log import log_figure3

        log_figure3(acc, run_name=wandb_name or model.stem)


def _per_slice_accuracy(log: Any, order: list[str]) -> dict[str, tuple[int, float]]:
    """Group logged samples by slice -> ``{slice: (n, accuracy)}`` (Figure-3 x-axis order)."""
    counts: dict[str, list[int]] = defaultdict(list)
    for s in log.samples:
        counts[s.metadata['slice']].append(int(bool(s.score.value)))
    acc: dict[str, tuple[int, float]] = {}
    for name in order:
        vals = counts.get(name, [])
        if vals:
            acc[name] = (len(vals), sum(vals) / len(vals))
    return acc


def _print_table(acc: dict[str, tuple[int, float]], order: list[str]) -> None:
    typer.echo('\nFigure 3 (med) -- slice accuracy:')
    typer.echo(f'  {"slice":<24}{"n":>6}{"accuracy":>12}')
    for name in order:
        if name in acc:
            n, a = acc[name]
            typer.echo(f'  {name:<24}{n:>6}{a:>12.4f}')


if __name__ == '__main__':
    app()
