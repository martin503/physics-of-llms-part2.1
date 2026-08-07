"""Tests for probe test-time evaluation (src.probe.evaluate) -- the pure parts.

``evaluate_run`` needs the frozen LM + a run dir and is integration-shaped (covered by the
CPU smoke path, not here). Two pieces are pure and load-bearing:

* ``classification_metrics`` -- the confusion counts feed the report's matrices directly;
  a swapped cell would silently mislabel every edge colour in the visualisation.
* ``predict_vprobe`` -- sorts queries by length internally (memory), then must scatter results
  back to *input* order. A mis-scatter would assign predictions to the wrong (A, B) pairs,
  which the aggregate metrics could never detect (same multiset of predictions).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.probe.evaluate import classification_metrics, predict_vprobe
from src.probe.build_queries import ProbeQuery


def test_classification_metrics_confusion_counts():
    """Each of the four cells counted into its own bucket, never reshuffled."""
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    preds  = np.array([1, 1, 0, 1, 0, 0, 0, 0])
    m = classification_metrics(labels, preds)
    assert (m['tp'], m['fn'], m['fp'], m['tn']) == (2, 1, 1, 4)
    assert m['n'] == 8
    assert m['acc'] == 6 / 8
    assert m['acc_majority'] == 5 / 8  # majority class is 0
    assert m['positive_frac'] == 3 / 8


def test_classification_metrics_perfect_and_inverted():
    labels = np.array([0, 1, 0, 1])
    assert classification_metrics(labels, labels)['mcc'] == 1.0
    assert classification_metrics(labels, 1 - labels)['mcc'] == -1.0


def test_classification_metrics_single_class_labels_no_crash():
    """All-one-class labels (a tiny all-pairs problem can be all-negative) -> mcc 0, no crash."""
    labels = np.zeros(5, dtype=int)
    m = classification_metrics(labels, np.array([0, 0, 1, 0, 0]))
    assert m['mcc'] == 0.0
    assert m['tn'] == 4 and m['fp'] == 1


class _ParityProbe(nn.Module):
    """Stand-in probe: predicts class 1 iff the token at `end_index` is even.

    Deterministic per-query output lets the test detect any misalignment between the
    internally length-sorted batches and the returned (input-order) arrays.
    """

    def forward(self, input_ids, attention_mask, end_index):
        end_tok = input_ids[torch.arange(len(end_index)), end_index]
        logit1 = torch.where(end_tok % 2 == 0, 10.0, -10.0)
        return torch.stack([-logit1, logit1], dim=-1)


def test_predict_vprobe_restores_input_order():
    """`preds[i]` must belong to `queries[i]` even though batching is length-sorted."""
    rng = np.random.default_rng(0)
    queries = []
    for i in range(57):  # varied lengths -> heavy internal reordering; odd count -> ragged batch
        length = int(rng.integers(1, 30))
        queries.append(ProbeQuery(input_ids=[7] * (length - 1) + [i], label=i % 2, group=0))
    preds, p1 = predict_vprobe(_ParityProbe(), queries, batch_size=8, device='cpu')
    expected = np.array([1 - (i % 2) for i in range(57)])  # token i even -> class 1
    assert np.array_equal(preds, expected)
    # probabilities follow the same alignment: saturated logits -> ~0 or ~1
    assert np.allclose(p1[expected == 1], 1.0, atol=1e-4)
    assert np.allclose(p1[expected == 0], 0.0, atol=1e-4)
