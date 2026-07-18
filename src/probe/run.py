"""CLI for the probe workflows (mirrors `src/eval/run.py` conventions).

Linear probe (two-stage, cached activations -- the degenerate baseline):

    uv run python -m src.probe.run extract --model-path final_models/gpt2-igsm-med \\
        --layer 6 --n-problems 500 --out data/probe/nece_l6.npz
    uv run python -m src.probe.run train --data data/probe/nece_l6.npz

V-probe (paper section 4.1; trains through the frozen model, no cache). Generate rows
offline first (multiprocess; a few hundred online problems overfit badly), then train.
`--target` selects the task: `nece` (necessity, read at end of question) or `dep`
(pairwise dependency, read at end of problem description; see `src.probe.vprobe`):

    uv run python -m src.probe.run gen-data --target nece --n-problems 20000 --workers 8 \\
        --model-path final_models/gpt2-igsm-med --out data/probe/vprobe_nece_test_20k
    uv run python -m src.probe.run vprobe --target nece --model-path final_models/gpt2-igsm-med \\
        --data data/probe/vprobe_nece_test_20k
    uv run python -m src.probe.run vprobe --target nece --random-model \\
        --data data/probe/vprobe_nece_test_20k   # paper's control

Each `vprobe` run writes a timestamped directory under `trained_probes/` containing
`config.json` (all parameters + git commit), `train.log`, `metrics.json` (final + per-epoch
history), and `probe.pt` (the trainable delta/head only; reload with `vprobe.load_vprobe`).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

app = typer.Typer(add_completion=False, help='Probes for the iGSM GPT2-RoPE model.')

DEFAULT_RUNS_DIR = Path('trained_probes')


def _setup_logging(log_file: Path | None = None) -> None:
    """Console logging for every command; tee into `log_file` for tracked training runs."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=handlers,
        force=True,
    )


def _start_run(runs_dir: Path, name: str, params: dict[str, Any]) -> Path:
    """Create `runs_dir/<timestamp>_<name>/`, wire logging into it, write config.json.

    The config is written *before* training so a crashed run still records what was
    attempted (params, command line, repo commit, start time).
    """
    from src.probe.data import git_commit

    run_dir = runs_dir / f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}_{name}'
    run_dir.mkdir(parents=True, exist_ok=False)
    _setup_logging(run_dir / 'train.log')
    config = {
        'params': params,
        'command': ' '.join(sys.argv),
        'repo_commit': git_commit(Path.cwd()),
        'started': datetime.now().astimezone().isoformat(timespec='seconds'),
    }
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n')
    logging.getLogger(__name__).info('run dir: %s', run_dir)
    return run_dir


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


@app.command(name='gen-data')
def gen_data(
    n_problems: Annotated[int, typer.Option('--n-problems')] = 20_000,
    target: Annotated[str, typer.Option('--target', help="Probe task: 'nece' or 'dep'.")] = 'nece',
    split: Annotated[str, typer.Option('--split', help="'test' = held-out (bins 16-22).")] = 'test',
    seed_start: Annotated[int, typer.Option('--seed-start')] = 0,
    workers: Annotated[int, typer.Option('--workers', help='Parallel generation processes.')] = 8,
    problems_per_shard: Annotated[
        int,
        typer.Option('--problems-per-shard', help='Problems per parquet shard (resume granularity).'),
    ] = 1_000,
    model_path: Annotated[
        str | None,
        typer.Option(
            '--model-path',
            help='Recorded in metadata.json as provenance (which model the data is for). '
            'The rows themselves are model-independent -- only the tokenizer is involved.',
        ),
    ] = None,
    out: Annotated[Path, typer.Option('--out')] = Path('data/probe/vprobe_nece_test_20k'),
    overwrite: Annotated[
        bool, typer.Option('--overwrite', help='Delete the output dir first and start fresh.')
    ] = False,
) -> None:
    """Generate V-probe rows offline (multiprocess, resumable parquet shards + metadata)."""
    from src.probe.data import generate_rows_to_dir

    _setup_logging()
    generate_rows_to_dir(
        out, n_problems, target=target, split=split, seed_start=seed_start, workers=workers,
        problems_per_shard=problems_per_shard, model_path=model_path, overwrite=overwrite,
    )


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
    target: Annotated[str, typer.Option('--target', help="Probe task: 'nece' or 'dep'.")] = 'nece',
    data: Annotated[
        Path | None,
        typer.Option(
            '--data',
            help='Offline row dataset dir from `gen-data`. When given, rows are loaded from '
            'disk and --n-problems/--seed-start/--split/--target are taken from its metadata.',
        ),
    ] = None,
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
    runs_dir: Annotated[
        Path, typer.Option('--runs-dir', help='Where run directories are created.')
    ] = DEFAULT_RUNS_DIR,
    run_name: Annotated[
        str | None,
        typer.Option('--run-name', help='Run-dir suffix (default: vprobe-<target>[-random]).'),
    ] = None,
) -> None:
    """V-probe (paper 4.1): frozen LM + rank-8 embedding delta + linear head at [END]."""
    from src.probe.data import load_vprobe_rows
    from src.probe.vprobe import build_vprobe_rows, load_lm, save_vprobe, train_vprobe

    if random_model == (model_path is not None):
        raise typer.BadParameter('pass exactly one of --model-path or --random-model')

    name = run_name or f'vprobe-{target}' + ('-random' if random_model else '')
    params = {
        'model_path': model_path, 'random_model': random_model, 'target': target,
        'data': str(data) if data else None, 'n_problems': n_problems,
        'seed_start': seed_start, 'split': split, 'rank': rank, 'epochs': epochs,
        'batch_size': batch_size, 'lr': lr, 'weight_decay': weight_decay,
        'balance_classes': balance_classes, 'grad_checkpointing': grad_checkpointing,
        'vram_fraction': vram_fraction, 'device': device, 'seed': seed,
    }
    run_dir = _start_run(runs_dir, name, params)
    log = logging.getLogger(__name__)

    if data is not None:
        rows, data_meta = load_vprobe_rows(data)
        if data_meta:
            if data_meta.get('target', target) != target:
                raise typer.BadParameter(
                    f"--target {target} but {data} holds '{data_meta['target']}' rows"
                )
            log.info(
                'loaded %d rows from %s (split=%s, %d problems, model=%s)',
                len(rows), data, data_meta.get('split'), data_meta.get('n_problems'),
                data_meta.get('model_path'),
            )
        else:
            log.warning('no metadata.json in %s -- provenance unknown', data)
    else:
        rows = build_vprobe_rows(n_problems, target=target, split=split, seed_start=seed_start)

    lm = load_lm(None if random_model else model_path, device=device)
    probe, metrics, history = train_vprobe(
        rows, lm, rank=rank, epochs=epochs, batch_size=batch_size, lr=lr,
        weight_decay=weight_decay, balance_classes=balance_classes,
        grad_checkpointing=grad_checkpointing, vram_fraction=vram_fraction,
        device=device, seed=seed,
    )

    save_vprobe(probe, run_dir / 'probe.pt')
    (run_dir / 'metrics.json').write_text(
        json.dumps({'final': metrics, 'history': history}, indent=2) + '\n'
    )
    label = 'RANDOM-INIT control' if random_model else model_path
    log.info('[%s] %s: %s', label, target, ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
    log.info('saved probe + metrics to %s', run_dir)


if __name__ == '__main__':
    app()
