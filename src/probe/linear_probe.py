"""Linear probe. Trained on cached activations; the transformer is absent here.

A linear probe is deliberately tiny: one affine map `h -> logits`. That's the whole point
-- if a single linear layer can recover a property, the property is *linearly present* in the
residual stream.

The query-conditioned V-probe lives in `src.probe.vprobe`.
"""

from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float
from torch import nn
from sklearn.metrics import matthews_corrcoef


class LinearProbe(nn.Module):
    """`h -> logits` for a per-token property. That's it.

    Args:
        d_model: Residual-stream width (768 for GPT2-12-12).
        n_classes: 2 for `nece`, `dep`, `known`, `can_next`, `nece_next`; 24 for `value` (0-22 plus 'unknown'); etc.
    """

    def __init__(self, d_model: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, n_classes)

    def forward(
        self, h: Float[torch.Tensor, 'batch d_model']
    ) -> Float[torch.Tensor, 'batch n_classes']:
        return self.linear(h)


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_classes: int = 2,
    val_frac: float = 0.2,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    balance_classes: bool = False,
    device: str = 'cpu',
    seed: int = 0,
) -> tuple[LinearProbe, dict[str, float]]:
    """
    Train the probe on `(X, y)` and report held-out MCC.

    Args:
        X (np.ndarray): `(N, d_model)` hidden states.
        y (np.ndarray): `(N,)` int64 labels.
        groups (np.ndarray | None): `(N,)` problem id per row; when given, whole problems
            (not rows) go to train or val, so correlated rows never leak across the split.
        n_classes (int): number of label classes.
        val_frac (float): fraction of data to hold out for validation.
        epochs (int): number of training epochs.
        lr (float): learning rate for AdamW.
        weight_decay (float): L2 regularization for AdamW.
        balance_classes (bool): weight the loss by inverse class frequency. When False,
            imbalance is left to the reported MCC, which is imbalance-robust.
        device (str): 'cpu' or 'cuda'.
        seed (int): random seed for reproducibility.

    Returns:
        LinearProbe: The trained probe (torch.nn.Module).
        dict[str, float]: metrics, `mcc_val` and `mcc_train`.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)  # LinearProbe init draws from torch's global RNG
    if groups is None:
        # row-level split: only valid when rows are independent samples
        indices = rng.permutation(len(X))
        split_idx = int((1 - val_frac) * len(X))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
    else:
        # group-level split: rows from one problem are correlated (or identical), so whole
        # problems go to either train or val -- never both (see leakage discussion)
        unique_groups = rng.permutation(np.unique(groups))
        n_val_groups = max(1, int(val_frac * len(unique_groups)))
        val_mask = np.isin(groups, unique_groups[:n_val_groups])
        train_indices = np.nonzero(~val_mask)[0]
        val_indices = np.nonzero(val_mask)[0]
    # optional per-class loss weight: INVERSE frequency (bincount returns counts directly;
    # minlength guarantees n_classes entries; clip avoids 1/0=inf for absent classes)
    if balance_classes:
        counts = np.bincount(y[train_indices], minlength=n_classes).clip(min=1)
        class_weights = torch.tensor(1 / counts, dtype=torch.float32, device=device)
    else:
        class_weights = None
    # move X, y tensors to `device` (float32 features, long labels).
    x_train = torch.tensor(X[train_indices], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[train_indices], dtype=torch.int64, device=device)
    x_val = torch.tensor(X[val_indices], dtype=torch.float32, device=device)
    y_val = torch.tensor(y[val_indices], dtype=torch.int64, device=device)

    probe = LinearProbe(d_model=X.shape[1], n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    for _epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        y_pred = probe(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            probe.eval()
            # Matthews correlation coefficient: 0 = chance, 1 = perfect, independent of class
            # imbalance and n_classes -- one number comparable across layers AND probe targets.
            # mcc_train >> mcc_val signals the probe memorising rather than reading structure.
            mcc_val = matthews_corrcoef(
                y_val.cpu().numpy(), probe(x_val).argmax(dim=1).cpu().numpy()
            )
            mcc_train = matthews_corrcoef(
                y_train.cpu().numpy(), probe(x_train).argmax(dim=1).cpu().numpy()
            )
    metrics: dict[str, float] = {
        'mcc_val': float(mcc_val),
        'mcc_train': float(mcc_train),
    }
    return probe, metrics
