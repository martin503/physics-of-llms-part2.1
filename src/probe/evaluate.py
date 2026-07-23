"""Test-time evaluation of trained V-probes on held-out offline datasets.

Training (``vprobe.py``) reports val metrics on a slice of its *own* dataset -- same seed
range, same balanced pair sampling for ``dep``. This module answers the follow-up question:
what does a saved probe do on **fresh problems** (a disjoint seed range) and, for ``dep``,
on the **natural pair distribution** (every ordered (A, B) pair, ~85-90% negative, generated
with ``gen-data --dep-all-pairs``) instead of the balanced training universe?

Flow (see ``run.py test``): a run directory from ``vprobe`` holds everything needed to
resurrect the probe -- ``config.json`` says which LM it trained through (path, or random-init
+ seed) and ``probe.pt`` holds the trainable delta/head. Predictions land *inside* the run
directory, keyed by dataset name::

    trained_probes/<run>/test_<dataset>/predictions.parquet   per-row: group, param_a,
                                                              param_b, label, pred, p1
    trained_probes/<run>/test_<dataset>/metrics.json          acc/mcc/confusion + provenance

Keeping per-row predictions (not just metrics) is what makes the graph report
(``report_dep.py``) possible: each dep row carries its (A, B) pair identity, so predictions
can be drawn as edges on the problem's dependency graph.

Random-model caveat: the random-init control's transformer lives nowhere but the training
process. Runs trained since the seeding fix rebuild it exactly from the recorded ``seed``;
runs from before that fix cannot be re-paired with their transformer, and evaluating them
here silently uses a *different* random model (still a valid "untrained control" reference
point, but not the exact model the probe was trained through -- expect chance-level output).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

from src.probe.vprobe import VProbe, VProbeRow, _pad_batch

logger = logging.getLogger(__name__)

PREDICTIONS_NAME = 'predictions.parquet'
METRICS_NAME = 'metrics.json'


@torch.no_grad()
def predict_vprobe(
    probe: VProbe,
    rows: list[VProbeRow],
    *,
    batch_size: int = 32,
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray]:
    """Run the probe over `rows`; return `(preds, p1)` aligned to the *input* row order.

    `p1` is the softmax probability of class 1 (useful for threshold sweeps later; argmax
    `preds` corresponds to the 0.5 threshold). Batches are length-sorted internally (same
    memory argument as `_evaluate` in vprobe.py) but results are scattered back, so
    `preds[i]` always belongs to `rows[i]`.
    """
    probe.eval()
    preds = np.empty(len(rows), dtype=np.int64)
    p1 = np.empty(len(rows), dtype=np.float64)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
    for start in tqdm(range(0, len(order), batch_size), desc='predict', unit='batch'):
        idx = order[start : start + batch_size]
        batch = [rows[i] for i in idx]
        ids, mask, end, _y = _pad_batch(batch, device)
        with torch.autocast(
            device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')
        ):
            logits = probe(ids, mask, end)
        probs = torch.softmax(logits.float(), dim=-1)
        preds[idx] = probs.argmax(dim=-1).cpu().numpy()
        p1[idx] = probs[:, 1].cpu().numpy()
    return preds, p1


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    """Accuracy, MCC, majority baseline, and raw confusion counts (binary labels).

    The confusion counts are computed directly (not via sklearn's confusion_matrix) so the
    tn/fp/fn/tp naming is explicit and an all-one-class edge case cannot reshuffle cells.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    majority = np.bincount(labels, minlength=2).argmax()
    return {
        'n': int(len(labels)),
        'acc': float((preds == labels).mean()),
        'mcc': float(matthews_corrcoef(labels, preds)) if len(set(labels.tolist())) > 1 else 0.0,
        'acc_majority': float((labels == majority).mean()),
        'positive_frac': float(labels.mean()),
        'tp': int(((preds == 1) & (labels == 1)).sum()),
        'tn': int(((preds == 0) & (labels == 0)).sum()),
        'fp': int(((preds == 1) & (labels == 0)).sum()),
        'fn': int(((preds == 0) & (labels == 1)).sum()),
    }


def test_output_dir(run_dir: Path, data_dir: Path) -> Path:
    """Where `evaluate_run` writes for this (run, dataset) pair: `<run_dir>/test_<dataset>`."""
    return Path(run_dir) / f'test_{Path(data_dir).name}'


def evaluate_run(
    run_dir: Path | str,
    data_dir: Path | str,
    *,
    batch_size: int = 32,
    device: str = 'cuda',
) -> dict[str, Any]:
    """Load the probe from `run_dir`, predict on the offline dataset at `data_dir`, save both
    per-row predictions and aggregate metrics into `test_output_dir(...)`; return the metrics.
    """
    from src.probe.data import git_commit, load_vprobe_rows
    from src.probe.vprobe import apply_memory_guardrails, load_lm, load_vprobe

    run_dir, data_dir = Path(run_dir), Path(data_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    params = config['params']

    rows, data_meta = load_vprobe_rows(data_dir)
    if data_meta and data_meta.get('target') != params.get('target'):
        raise ValueError(
            f"run {run_dir.name} was trained for target '{params.get('target')}' but "
            f"{data_dir} holds '{data_meta.get('target')}' rows"
        )

    if params.get('random_model'):
        logger.warning(
            'random-init control: rebuilding the LM from seed=%s. Faithful only if the run '
            'was trained with seeded init (runs from before that fix cannot be re-paired '
            'with their transformer; expect chance-level output).',
            params.get('seed'),
        )
    apply_memory_guardrails(device, params.get('vram_fraction', 0.85))
    lm = load_lm(params.get('model_path'), device=device, seed=params.get('seed'))
    probe = load_vprobe(run_dir / 'probe.pt', lm, device=device)

    preds, p1 = predict_vprobe(probe, rows, batch_size=batch_size, device=device)
    labels = np.array([r.label for r in rows])
    metrics: dict[str, Any] = classification_metrics(labels, preds)
    metrics['run_dir'] = str(run_dir)
    metrics['data_dir'] = str(data_dir)
    metrics['data_metadata'] = data_meta
    metrics['repo_commit'] = git_commit(Path.cwd())

    out_dir = test_output_dir(run_dir, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            'group': [r.group for r in rows],
            'param_a': [r.param_a for r in rows],
            'param_b': [r.param_b for r in rows],
            'label': labels,
            'pred': preds,
            'p1': p1,
        }
    )
    pq.write_table(table, str(out_dir / PREDICTIONS_NAME))
    (out_dir / METRICS_NAME).write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    logger.info(
        'test [%s on %s]: acc=%.4f mcc=%.4f (majority %.4f, n=%d) -> %s',
        run_dir.name, data_dir.name, metrics['acc'], metrics['mcc'],
        metrics['acc_majority'], metrics['n'], out_dir,
    )
    return metrics


def load_predictions(run_dir: Path | str, data_dir: Path | str) -> dict[str, np.ndarray]:
    """Load saved predictions for (run, dataset) as a dict of column arrays.

    Raises FileNotFoundError with the exact command to run if `evaluate_run` hasn't
    produced them yet -- the report builder depends on this.
    """
    path = test_output_dir(Path(run_dir), Path(data_dir)) / PREDICTIONS_NAME
    if not path.exists():
        raise FileNotFoundError(
            f'no predictions at {path} -- run: uv run python -m src.probe.run test '
            f'--run-dir {run_dir} --data {data_dir}'
        )
    table = pq.read_table(str(path))
    return {name: table.column(name).to_numpy() for name in table.column_names}
