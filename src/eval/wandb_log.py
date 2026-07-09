"""Optional wandb logging of the Figure-3 per-slice accuracy table.

inspect_ai has no built-in wandb writer, so we log manually after ``eval()``. The
project/entity are read from env to reuse the training wandb project.
"""

from __future__ import annotations

import os

import typer


def log_figure3(acc: dict[str, tuple[int, float]], run_name: str) -> None:
    """Log a per-slice accuracy table + one summary scalar per slice to wandb.

    ``acc`` maps slice name -> ``(n, accuracy)``. Reads ``WANDB_ENTITY`` / ``WANDB_PROJECT``
    from env (the training project); no-op-safe if wandb is unavailable or ``WANDB_MODE`` is
    offline/disabled.
    """
    import wandb

    entity = os.environ.get('WANDB_ENTITY')
    project = os.environ.get('WANDB_PROJECT', 'physics_of_llms')
    run = wandb.init(project=project, entity=entity, name=run_name, reinit=True)
    try:
        table = wandb.Table(
            columns=['slice', 'n', 'accuracy'],
            data=[[k, n, a] for k, (n, a) in acc.items()],
        )
        run.log({'figure3': table})
        for k, (n, a) in acc.items():
            run.log({f'acc/{k}': a, f'n/{k}': n})
    finally:
        run.finish()
    typer.echo(f'wandb: logged {len(acc)} slices as "{run_name}"')
