"""Training loop for the V-probe: fit head + embedding delta on labelled rows.

The pieces come from either side of this module: rows from `queries.py`, the model and its
batching from `vprobe.py`. What lives here is everything about *how the fitting is run* --
the group-wise train/val split, batch construction, the epoch loop, and the metrics reported
back to `run.py`.

Unlike the cached-activation pipeline in `extract.py`, V-probe training must forward through
the frozen LM every step: gradients have to flow *through* the transformer to reach the
embedding delta at the bottom. That single fact drives most of the choices below (grad
checkpointing, bf16 autocast, length bucketing, the VRAM cap).

Paper control: run the identical procedure on a randomly-initialised model (`--random-model`).
The pretrained-vs-random gap is the evidence; the absolute number alone is not.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn
from tqdm import tqdm

from src.probe.build_queries import VProbeRow
from src.probe.vprobe import DEFAULT_VRAM_FRACTION, VProbe, _pad_batch, apply_memory_guardrails

logger = logging.getLogger(__name__)


def _length_bucketed_batches(
    rows: list[VProbeRow], indices: np.ndarray, batch_size: int
) -> list[list[VProbeRow]]:
    """Group rows of similar length into batches, then shuffle the batch order.

    Two wins over naive shuffling:
      * memory: a batch's cost is `batch_size * max_len_in_batch`, so mixing a 600-token row
        with 31 short ones pads them all to 600. Bucketing keeps peak allocation near the
        average rather than the worst case.
      * speed: constant-ish shapes stop the caching allocator from fragmenting across many
        distinct padded lengths -- the fragmentation that made batch times creep up (1.4s ->
        15s) until the run spilled into system RAM.
    Batch *order* is still shuffled, so the optimiser does not see length-sorted data.
    """
    by_len = sorted(indices, key=lambda i: len(rows[i].input_ids))
    batches = [
        [rows[i] for i in by_len[s : s + batch_size]] for s in range(0, len(by_len), batch_size)
    ]
    return batches


def _split_by_group(
    rows: list[VProbeRow], val_frac: float, seed: int
) -> tuple[list[VProbeRow], list[VProbeRow]]:
    """Assign whole problems to train or val (see the leakage discussion in probe.py)."""
    rng = np.random.default_rng(seed)
    unique_groups = rng.permutation(np.unique([r.group for r in rows]))
    val_groups = set(unique_groups[: max(1, int(val_frac * len(unique_groups)))].tolist())
    train = [r for r in rows if r.group not in val_groups]
    val = [r for r in rows if r.group in val_groups]
    return train, val


@torch.no_grad()
def _evaluate(
    probe: VProbe,
    rows: list[VProbeRow],
    batch_size: int,
    device: str,
    loss_fn: nn.Module | None = None,
) -> dict[str, float]:
    """Accuracy + MCC over `rows` (accuracy for comparability with the paper's Figure 7).

    `loss_fn`, if given, is also evaluated batch-wise and averaged; used to report a
    val loss comparable to the training loss without running eval twice.
    """
    probe.eval()
    preds: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    # length-sorted so eval batches don't pad to the global max either
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
    for start in range(0, len(order), batch_size):
        batch = [rows[i] for i in order[start : start + batch_size]]
        ids, mask, end, y = _pad_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            logits = probe(ids, mask, end)
            if loss_fn is not None:
                total_loss += loss_fn(logits, y).item() * len(batch)
        preds.extend(logits.float().argmax(dim=-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    preds_arr, labels_arr = np.asarray(preds), np.asarray(labels)
    majority = np.bincount(labels_arr).argmax()
    metrics = {
        'acc': float((preds_arr == labels_arr).mean()),
        'mcc': float(matthews_corrcoef(labels_arr, preds_arr)),
        'acc_majority': float((labels_arr == majority).mean()),  # the paper's baseline row
    }
    if loss_fn is not None:
        metrics['loss'] = total_loss / len(rows)
    return metrics


def train_vprobe(
    rows: list[VProbeRow],
    lm,
    *,
    n_classes: int = 2,
    rank: int = 8,
    val_frac: float = 0.2,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    balance_classes: bool = False,
    grad_checkpointing: bool = True,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
    device: str = 'cuda',
    seed: int = 0,
) -> tuple[VProbe, dict[str, float], list[dict[str, float]]]:
    """Train head + embedding delta on `rows` through the frozen `lm`; report val metrics.

    Returns `(probe, metrics, history)`: `metrics` has `acc_val` / `mcc_val`, the train-set
    counterparts (memorisation check), `acc_majority` (the paper's baseline), timing, and
    `peak_vram_gib`; `history` has one dict per epoch (losses, MCCs, seconds) for logging.
    Interpretation needs the random-model control: rerun with a random-init lm and compare.

    `balance_classes` weights the loss by inverse train-set class frequency (as in
    `probe.train_probe`). With the ~80/20 `nece` imbalance the unweighted loss lets the
    optimiser collapse to always predicting the majority class -- loss parks at the prior's
    entropy and MCC stays ~0. Weighting removes that trivial minimum; watch MCC, not accuracy.

    Memory: gradients must reach the embedding delta at the bottom of the LM, so autograd would
    hold every block's activations. `grad_checkpointing` + bf16 autocast + length bucketing +
    a hard VRAM cap keep this bounded (see `apply_memory_guardrails`).
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    apply_memory_guardrails(device, vram_fraction)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    train_rows, val_rows = _split_by_group(rows, val_frac, seed)

    probe = VProbe(lm, n_classes=n_classes, rank=rank, grad_checkpointing=grad_checkpointing)
    probe = probe.to(device)
    trainable = [p for p in probe.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # optional per-class loss weight: INVERSE train-set frequency (bincount gives counts;
    # minlength guarantees n_classes entries; clip avoids 1/0=inf for an absent class). With
    # reduction='mean' the global weight scale cancels, so the reported loss stays comparable.
    if balance_classes:
        counts = np.bincount([r.label for r in train_rows], minlength=n_classes).clip(min=1)
        class_weights = torch.tensor(1 / counts, dtype=torch.float32, device=device)
    else:
        class_weights = None
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    indices = np.arange(len(train_rows))
    history: list[dict[str, float]] = []
    t_start = time.perf_counter()
    # last known val numbers, carried into the postfix until the next epoch's eval pass
    val_loss, val_mcc = float('nan'), float('nan')
    for epoch in range(epochs):
        t_epoch = time.perf_counter()
        probe.train()
        batches = _length_bucketed_batches(train_rows, indices, batch_size)
        rng.shuffle(batches)  # shuffle batch order, keep within-batch lengths homogeneous
        total_loss = 0.0
        n_seen = 0
        epoch_preds: list[int] = []
        epoch_labels: list[int] = []
        pbar = tqdm(batches, desc=f'epoch {epoch + 1}/{epochs}', unit='batch')
        for batch in pbar:
            ids, mask, end, y = _pad_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')
            ):
                logits = probe(ids, mask, end)
                loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            n_seen += len(batch)
            epoch_preds.extend(logits.detach().float().argmax(dim=-1).cpu().tolist())
            epoch_labels.extend(y.cpu().tolist())
            # MCC needs both classes seen at least once, else sklearn warns and returns 0
            train_mcc = (
                matthews_corrcoef(epoch_labels, epoch_preds) if len(set(epoch_labels)) > 1 else float('nan')
            )
            pbar.set_postfix(
                {
                    'loss': total_loss / n_seen,
                    'mcc': train_mcc,
                    'val_loss': val_loss,
                    'val_mcc': val_mcc,
                },
                refresh=False,
            )
        train_epoch_loss = total_loss / n_seen
        val_metrics = _evaluate(probe, val_rows, batch_size, device, loss_fn=loss_fn)
        val_loss, val_mcc = val_metrics['loss'], val_metrics['mcc']
        epoch_seconds = time.perf_counter() - t_epoch
        history.append(
            {
                'epoch': epoch + 1,
                'train_loss': train_epoch_loss,
                'train_mcc': float(train_mcc),
                'val_loss': val_loss,
                'val_mcc': val_mcc,
                'val_acc': val_metrics['acc'],
                'seconds': epoch_seconds,
            }
        )
        logger.info(
            'epoch %d/%d: train loss %.4f mcc %.3f | val loss %.4f mcc %.3f | %.1fs',
            epoch + 1, epochs, train_epoch_loss, train_mcc, val_loss, val_mcc, epoch_seconds,
        )

    val_metrics = _evaluate(probe, val_rows, batch_size, device)
    train_metrics = _evaluate(probe, train_rows, batch_size, device)
    metrics = {
        'acc_val': val_metrics['acc'],
        'mcc_val': val_metrics['mcc'],
        'acc_majority': val_metrics['acc_majority'],
        'acc_train': train_metrics['acc'],
        'mcc_train': train_metrics['mcc'],
        'n_train': float(len(train_rows)),
        'n_val': float(len(val_rows)),
        'train_seconds': time.perf_counter() - t_start,
    }
    if device.startswith('cuda'):
        metrics['peak_vram_gib'] = torch.cuda.max_memory_allocated() / 2**30
    return probe, metrics, history
