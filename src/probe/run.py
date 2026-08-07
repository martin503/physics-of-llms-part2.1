"""CLI for the probe workflows (mirrors `src/eval/run.py` conventions).

Linear probe (two-stage, cached activations -- the degenerate baseline):

    uv run python -m src.probe.run extract --model-path final_models/gpt2-igsm-med \\
        --layer 6 --n-problems 500 --out data/probe/nece_l6.npz
    uv run python -m src.probe.run train --data data/probe/nece_l6.npz

V-probe (paper section 4.1; trains through the frozen model, no cache). Generate queries
offline first (multiprocess; a few hundred online problems overfit badly), then train.
`--target` selects the task: `nece` (necessity, read at end of question) or `dep`
(pairwise dependency, read at end of problem description; see `src.probe.build_queries`):

    uv run python -m src.probe.run gen-data --target nece --n-problems 20000 --workers 8 \\
        --model-path final_models/gpt2-igsm-med --out data/probe/vprobe_nece_test_20k
    uv run python -m src.probe.run vprobe --target nece --model-path final_models/gpt2-igsm-med \\
        --data data/probe/vprobe_nece_test_20k
    uv run python -m src.probe.run vprobe --target nece --random-model \\
        --data data/probe/vprobe_nece_test_20k   # paper's control

Each `vprobe` run writes a timestamped directory under `trained_probes/` containing
`config.json` (all parameters + git commit), `train.log`, `metrics.json` (final + per-epoch
history), and `probe.pt` (the trainable delta/head only; reload with `vprobe.load_vprobe`).

Testing a trained probe on fresh problems (disjoint seed range; for `dep` use
`--dep-all-pairs` so every ordered (A, B) pair is present -- the natural distribution the
graph report needs). Then evaluate each run dir and render the interactive report:

    uv run python -m src.probe.run gen-data --target dep --dep-all-pairs --n-problems 200 \\
        --seed-start 1000000 --workers 8 --out data/probe/vprobe_dep_eval_200
    uv run python -m src.probe.run test --run-dir trained_probes/<pretrained-run> \\
        --data data/probe/vprobe_dep_eval_200
    uv run python -m src.probe.run test --run-dir trained_probes/<random-run> \\
        --data data/probe/vprobe_dep_eval_200
    uv run python -m src.probe.run report-dep --pretrained-run trained_probes/<pretrained-run> \\
        --random-run trained_probes/<random-run> --data data/probe/vprobe_dep_eval_200
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


def _med_cfg_override(max_op: int | None, max_edge: int | None) -> dict[str, Any] | None:
    """Build an iGSM-med config with `max_op`/`max_edge` overridden, or None to use defaults.

    iGSM-med caps difficulty at `max_op=15` (the training range). Raising it lets the
    generator emit harder out-of-distribution problems (the paper evaluates at op 20-23);
    n_op is still sampled across `1..max_op`, so op>15 is rare -- scan/generate more.
    """
    from src.data.igsm import IGSM_MED

    if max_op is None and max_edge is None:
        return None
    cfg = dict(IGSM_MED)
    if max_op is not None:
        cfg['max_op'] = max_op
    if max_edge is not None:
        cfg['max_edge'] = max_edge
    return cfg


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
        typer.Option('--problems-per-shard', help='Problems per parquet shard (memory bound).'),
    ] = 1_000,
    model_path: Annotated[
        str | None,
        typer.Option(
            '--model-path',
            help='Recorded in metadata.json as provenance (which model the data is for). '
            'The queries themselves are model-independent -- only the tokenizer is involved.',
        ),
    ] = None,
    out: Annotated[Path, typer.Option('--out')] = Path('data/probe/vprobe_nece_test_20k'),
    overwrite: Annotated[
        bool, typer.Option('--overwrite', help='Delete the output dir first and start fresh.')
    ] = False,
    dep_all_pairs: Annotated[
        bool,
        typer.Option(
            '--dep-all-pairs',
            help='dep only: keep every ordered (A, B) pair instead of the balanced subsample '
            '-- the natural test distribution (~85-90%% negative), needed for the graph report.',
        ),
    ] = False,
    seeds_file: Annotated[
        Path | None,
        typer.Option(
            '--seeds-file',
            help='JSON from `find-seeds`: generate queries for exactly those problem seeds '
            '(overrides --n-problems/--seed-start/--split/--max-op/--max-edge).',
        ),
    ] = None,
    max_op: Annotated[
        int | None,
        typer.Option('--max-op', help='Override iGSM-med difficulty cap (default 15). Must match '
                     'the value the seeds were found under; ignored when --seeds-file is given.'),
    ] = None,
    max_edge: Annotated[
        int | None, typer.Option('--max-edge', help='Override iGSM-med graph-width cap (default 20).')
    ] = None,
) -> None:
    """Generate V-probe queries offline (multiprocess, parquet shards + metadata)."""
    import json as _json

    from src.probe.data import generate_queries_to_dir

    if dep_all_pairs and target != 'dep':
        raise typer.BadParameter('--dep-all-pairs only applies to --target dep')
    seed_list = None
    med_cfg = _med_cfg_override(max_op, max_edge)
    if seeds_file is not None:
        showcase = _json.loads(seeds_file.read_text(encoding='utf-8'))
        seed_list = showcase['seeds']
        split = showcase['split']  # the seeds are only meaningful under their own split
        med_cfg = showcase.get('med_cfg', med_cfg)  # regenerate under the config they were found with
    _setup_logging()
    generate_queries_to_dir(
        out, n_problems, target=target, split=split, seed_start=seed_start, workers=workers,
        problems_per_shard=problems_per_shard, model_path=model_path, overwrite=overwrite,
        dep_all_pairs=dep_all_pairs, seed_list=seed_list, med_cfg=med_cfg,
    )


@app.command(name='find-seeds')
def find_seeds(
    out: Annotated[Path, typer.Option('--out')] = Path('data/probe/showcase_seeds.json'),
    per_op: Annotated[
        int, typer.Option('--per-op', help='Problems to keep per difficulty (op count).')
    ] = 3,
    split: Annotated[str, typer.Option('--split')] = 'test',
    scan_seed: Annotated[
        int, typer.Option('--scan-seed', help='RNG seed for the (scattered) candidate stream.')
    ] = 0,
    max_scan: Annotated[int, typer.Option('--max-scan')] = 2_000,
    workers: Annotated[int, typer.Option('--workers')] = 8,
    max_op: Annotated[
        int | None,
        typer.Option('--max-op', help='Override iGSM-med difficulty cap (default 15). Raise for '
                     'harder out-of-distribution problems, e.g. 23 for the paper op-20-23 eval.'),
    ] = None,
    max_edge: Annotated[
        int | None, typer.Option('--max-edge', help='Override iGSM-med graph-width cap (default 20).')
    ] = None,
) -> None:
    """Pick scattered showcase seeds: `--per-op` problems per difficulty (iGSM `n_op`).

    Difficulty runs `1..max_op` (15 by default; raise with `--max-op`). Candidates are drawn
    randomly from a huge seed range, so the accepted seeds are non-sequential and disjoint
    from training's sequential ranges. High op counts are rare (~1-2%), so filling their
    buckets needs a large `--max-scan`. Feed the JSON to `gen-data --seeds-file` (which reads
    back the same med_cfg)."""
    from src.probe.data import find_showcase_seeds

    _setup_logging()
    find_showcase_seeds(
        out, per_op=per_op, split=split, scan_seed=scan_seed, max_scan=max_scan, workers=workers,
        med_cfg=_med_cfg_override(max_op, max_edge),
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
            help='Offline query dataset dir from `gen-data`. When given, queries are loaded from '
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
        typer.Option('--batch-size', help='Queries per step. Main VRAM knob; lower if you OOM.'),
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
    from src.probe.data import load_vprobe_queries
    from src.probe.build_queries import build_vprobe_queries
    from src.probe.vprobe import load_lm, save_vprobe
    from src.probe.vprobe_train import train_vprobe

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
        queries, data_meta = load_vprobe_queries(data)
        if data_meta:
            if data_meta.get('target', target) != target:
                raise typer.BadParameter(
                    f"--target {target} but {data} holds '{data_meta['target']}' queries"
                )
            log.info(
                'loaded %d queries from %s (split=%s, %d problems, model=%s)',
                len(queries), data, data_meta.get('split'), data_meta.get('n_problems'),
                data_meta.get('model_path'),
            )
        else:
            log.warning('no metadata.json in %s -- provenance unknown', data)
    else:
        queries = build_vprobe_queries(n_problems, target=target, split=split, seed_start=seed_start)

    # `seed` also fixes the random-init control's weights: `test` rebuilds the same LM from
    # the recorded seed, so a saved probe.pt can be re-paired with its transformer later.
    lm = load_lm(None if random_model else model_path, device=device, seed=seed)
    probe, metrics, history = train_vprobe(
        queries, lm, rank=rank, epochs=epochs, batch_size=batch_size, lr=lr,
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


@app.command(name='test')
def test_probe(
    run_dir: Annotated[
        Path,
        typer.Option('--run-dir', help='Trained probe run dir (config.json + probe.pt).'),
    ],
    data: Annotated[
        Path,
        typer.Option(
            '--data',
            help='Offline eval dataset from `gen-data` -- use a seed range disjoint from '
            'training and, for dep, --dep-all-pairs (natural distribution).',
        ),
    ],
    batch_size: Annotated[int, typer.Option('--batch-size')] = 32,
    device: Annotated[str, typer.Option('--device')] = 'cuda',
) -> None:
    """Evaluate a trained V-probe on held-out problems; write predictions + metrics into the
    run dir (`<run-dir>/test_<dataset>/`). Run once per probe (pretrained AND random control)
    before `report-dep`."""
    from src.probe.evaluate import evaluate_run

    _setup_logging()
    metrics = evaluate_run(run_dir, data, batch_size=batch_size, device=device)
    typer.echo(
        f'acc={metrics["acc"]:.4f} mcc={metrics["mcc"]:.4f} '
        f'(majority {metrics["acc_majority"]:.4f}) | '
        f'tp={metrics["tp"]} tn={metrics["tn"]} fp={metrics["fp"]} fn={metrics["fn"]}'
    )


@app.command(name='report-dep')
def report_dep(
    pretrained_run: Annotated[
        Path, typer.Option('--pretrained-run', help='Run dir of the pretrained-model probe.')
    ],
    random_run: Annotated[
        Path, typer.Option('--random-run', help='Run dir of the random-init control probe.')
    ],
    data: Annotated[
        Path, typer.Option('--data', help='The eval dataset both probes were `test`ed on.')
    ],
    n_problems: Annotated[
        int,
        typer.Option('--n-problems', help='How many problems to include in the report.'),
    ] = 12,
    out: Annotated[Path, typer.Option('--out')] = Path('visualizations/dep_probe_report.html'),
) -> None:
    """Render the interactive dep(A, B) report: problem text + dependency-graph view of both
    probes' predictions (needs `test` output for both run dirs on the same dataset)."""
    from src.probe.report_dep import build_report

    _setup_logging()
    build_report(
        pretrained_run=pretrained_run, random_run=random_run, data_dir=data,
        n_problems=n_problems, out=out,
    )
    typer.echo(f'report written to {out}')


if __name__ == '__main__':
    app()
