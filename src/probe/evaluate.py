"""Test-time evaluation of trained V-probes on held-out offline datasets.

Training (``vprobe_train.py``) reports val metrics on a slice of its *own* dataset -- same seed
range, same balanced pair sampling for ``dep``. This module answers the follow-up question:
what does a saved probe do on **fresh problems** (a disjoint seed range) and, for ``dep``,
on the **natural pair distribution** (every ordered (A, B) pair, ~85-90% negative, generated
with ``gen-data --dep-all-pairs``) instead of the balanced training universe?

Flow (see ``run.py test``): a run directory from ``vprobe`` holds everything needed to
resurrect the probe -- ``config.json`` says which LM it trained through (a checkpoint path, or
the random-init control stored alongside as ``vprobe.RANDOM_LM_NAME``) and ``probe.pt`` holds
the trainable delta/head. Predictions land *inside* the run directory, keyed by dataset name::

    trained_probes/<run>/test_<dataset>/predictions.parquet   per-query: group, param_a,
                                                              param_b, label, pred, p1
    trained_probes/<run>/test_<dataset>/metrics.json          acc/mcc/confusion + provenance

Keeping per-query predictions (not just metrics) is what makes the graph report
(``report_dep.py``) possible: each dep query carries its (A, B) pair identity, so predictions
can be drawn as edges on the problem's dependency graph.
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

from src.probe.build_queries import ProbeQuery
from src.probe.vprobe import VProbe, _pad_batch

logger = logging.getLogger(__name__)

PREDICTIONS_NAME = 'predictions.parquet'
METRICS_NAME = 'metrics.json'


@torch.no_grad()
def predict_vprobe(
    probe: VProbe,
    queries: list[ProbeQuery],
    *,
    batch_size: int = 32,
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray]:
    """Run the probe over `queries`; return `(preds, p1)` aligned to the *input* query order.

    `p1` is the softmax probability of class 1 (useful for threshold sweeps later; argmax
    `preds` corresponds to the 0.5 threshold). Batches are length-sorted internally (same
    memory argument as `_evaluate` in vprobe_train.py) but results are scattered back, so
    `preds[i]` always belongs to `queries[i]`.
    """
    probe.eval()
    preds = np.empty(len(queries), dtype=np.int64)
    p1 = np.empty(len(queries), dtype=np.float64)
    order = sorted(range(len(queries)), key=lambda i: len(queries[i].input_ids))
    for start in tqdm(range(0, len(order), batch_size), desc='predict', unit='batch'):
        idx = order[start : start + batch_size]
        batch = [queries[i] for i in idx]
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
    per-query predictions and aggregate metrics into `test_output_dir(...)`; return the metrics.
    """
    from src.probe.data import git_commit, load_vprobe_queries
    from src.probe.vprobe import (
        RANDOM_LM_NAME,
        apply_memory_guardrails,
        load_lm,
        load_random_lm,
        load_vprobe,
    )

    run_dir, data_dir = Path(run_dir), Path(data_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    params = config['params']

    queries, data_meta = load_vprobe_queries(data_dir)
    if data_meta and data_meta.get('target') != params.get('target'):
        raise ValueError(
            f"run {run_dir.name} was trained for target '{params.get('target')}' but "
            f"{data_dir} holds '{data_meta.get('target')}' queries"
        )

    apply_memory_guardrails(device, params.get('vram_fraction', 0.85))
    if params.get('random_model'):
        lm = load_random_lm(run_dir / RANDOM_LM_NAME, device=device)
    else:
        lm = load_lm(params.get('model_path'), device=device)
    probe = load_vprobe(run_dir / 'probe.pt', lm, device=device)

    preds, p1 = predict_vprobe(probe, queries, batch_size=batch_size, device=device)
    labels = np.array([q.label for q in queries])
    metrics: dict[str, Any] = classification_metrics(labels, preds)
    metrics['run_dir'] = str(run_dir)
    metrics['data_dir'] = str(data_dir)
    metrics['data_metadata'] = data_meta
    metrics['repo_commit'] = git_commit(Path.cwd())

    out_dir = test_output_dir(run_dir, data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            'group': [q.group for q in queries],
            'param_a': [q.param_a for q in queries],
            'param_b': [q.param_b for q in queries],
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
