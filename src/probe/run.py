"""CLI for the probe workflows (mirrors `src/eval/run.py` conventions).

Linear probe (two-stage, cached activations -- the degenerate baseline):

    uv run python -m src.probe.run extract --model-path final_models/gpt2-rope-igsm/final \\
        --layer 6 --n-problems 500 --out data/probe/nece_l6.npz
    uv run python -m src.probe.run train --data data/probe/nece_l6.npz

V-probe (paper section 4.1; trains through the frozen model, no cache):

    uv run python -m src.probe.run vprobe --model-path final_models/gpt2-rope-igsm/final \\
        --n-problems 500
    uv run python -m src.probe.run vprobe --random-model --n-problems 500   # paper's control
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

app = typer.Typer(add_completion=False, help='Probes for the iGSM GPT2-RoPE model.')


@app.command()
def extract(
    model_path: Annotated[str, typer.Option('--model-path', help='HF checkpoint dir.')],
    layer: Annotated[int, typer.Option('--layer', help='Residual-stream layer to read (1..n_layer).')] = 6,
    n_problems: Annotated[int, typer.Option('--n-problems')] = 500,
    seed_start: Annotated[int, typer.Option('--seed-start')] = 0,
    split: Annotated[str, typer.Option('--split', help="'test' = held-out (bins 16-22).")] = 'test',
    device: Annotated[str, typer.Option('--device')] = 'cuda',
    out: Annotated[Path, typer.Option('--out')] = Path('data/probe/nece_l6.npz'),
) -> None:
    """Stage A: cache (hidden states, nece labels, problem groups) for a single layer."""
    from src.probe.extract import build_dataset, save_dataset

    X, y, groups = build_dataset(
        model_path, layer=layer, n_problems=n_problems, seed_start=seed_start,
        split=split, device=device,
    )
    save_dataset(out, X, y, groups, layer=layer)
    typer.echo(
        f'{X.shape[0]} rows (d={X.shape[1]}) from {len(np.unique(groups))} problems, '
        f'{y.mean():.3f} positive -> {out}'
    )


@app.command()
def train(
    data: Annotated[Path, typer.Option('--data', help='.npz from `extract`.')],
    n_classes: Annotated[int, typer.Option('--n-classes')] = 2,
    epochs: Annotated[int, typer.Option('--epochs')] = 50,
    lr: Annotated[float, typer.Option('--lr')] = 1e-3,
    weight_decay: Annotated[float, typer.Option('--weight-decay')] = 1e-3,
    balance_classes: Annotated[
        bool,
        typer.Option(
            '--balance-classes/--no-balance-classes',
            help='Weight the loss by inverse train-set class frequency (counters imbalance).',
        ),
    ] = False,
    device: Annotated[str, typer.Option('--device')] = 'cpu',
    seed: Annotated[int, typer.Option('--seed', help='Split/init seed; fix it across a sweep.')] = 0,
) -> None:
    """Stage B: train the linear probe (group split) and print held-out metrics."""
    from src.probe.probe import train_probe

    d = np.load(data)
    groups = d['groups'] if 'groups' in d.files else None
    if groups is None:
        typer.echo('WARNING: no groups in npz -- row split leaks between problems')
    _probe, metrics = train_probe(
        d['X'], d['y'], groups=groups, n_classes=n_classes, epochs=epochs, lr=lr,
        weight_decay=weight_decay, balance_classes=balance_classes, device=device, seed=seed,
    )
    typer.echo(f'layer {int(d["layer"])}: {metrics}')


@app.command()
def vprobe(
    model_path: Annotated[
        str | None,
        typer.Option('--model-path', help='HF checkpoint dir (omit with --random-model).'),
    ] = None,
    random_model: Annotated[
        bool,
        typer.Option(
            '--random-model',
            help="Paper's control: identical probe on a random-init LM. The pretrained-vs-"
            'random gap is the evidence of knowledge in the pretrained weights.',
        ),
    ] = False,
    target: Annotated[str, typer.Option('--target', help='Probe task (nece only, so far).')] = 'nece',
    n_problems: Annotated[int, typer.Option('--n-problems')] = 500,
    seed_start: Annotated[int, typer.Option('--seed-start')] = 0,
    split: Annotated[str, typer.Option('--split')] = 'test',
    rank: Annotated[int, typer.Option('--rank', help='Rank of the embedding update.')] = 8,
    epochs: Annotated[int, typer.Option('--epochs')] = 3,
    batch_size: Annotated[
        int,
        typer.Option('--batch-size', help='Rows per step. Main VRAM knob; lower if you OOM.'),
    ] = 8,
    lr: Annotated[float, typer.Option('--lr')] = 1e-3,
    weight_decay: Annotated[float, typer.Option('--weight-decay')] = 1e-3,
    balance_classes: Annotated[
        bool,
        typer.Option(
            '--balance-classes/--no-balance-classes',
            help='Weight the loss by inverse train-set class frequency. Without it the ~80/20 '
            'nece imbalance lets the probe collapse to the majority class (MCC ~0).',
        ),
    ] = False,
    grad_checkpointing: Annotated[
        bool,
        typer.Option(
            '--grad-checkpointing/--no-grad-checkpointing',
            help='Recompute block activations in backward (~30%% slower, far less VRAM).',
        ),
    ] = True,
    vram_fraction: Annotated[
        float,
        typer.Option(
            '--vram-fraction',
            help='Hard cap on VRAM. Prevents the Windows driver spilling into system RAM '
            '(which hangs the desktop); a too-large run OOMs cleanly instead.',
        ),
    ] = 0.85,
    device: Annotated[str, typer.Option('--device')] = 'cuda',
    seed: Annotated[int, typer.Option('--seed', help='Split/init seed.')] = 0,
) -> None:
    """V-probe (paper 4.1): frozen LM + rank-8 embedding delta + linear head at [END]."""
    from src.probe.vprobe import build_vprobe_rows, load_lm, train_vprobe

    if random_model == (model_path is not None):
        raise typer.BadParameter('pass exactly one of --model-path or --random-model')

    rows = build_vprobe_rows(
        n_problems, target=target, split=split, seed_start=seed_start
    )
    lm = load_lm(None if random_model else model_path, device=device)
    _probe, metrics = train_vprobe(
        rows, lm, rank=rank, epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, balance_classes=balance_classes,
        grad_checkpointing=grad_checkpointing, vram_fraction=vram_fraction,
        device=device, seed=seed,
    )
    label = 'RANDOM-INIT control' if random_model else model_path
    typer.echo(f'[{label}] {target}: ' + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))


if __name__ == '__main__':
    app()
