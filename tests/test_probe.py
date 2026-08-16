"""Tests for the linear probe (src.probe.probe).

The probe trainer cannot be tested against the real model -- we have no independent
ground truth for "is nece(A) linearly decodable from layer 6". So we test it against
**synthetic activations whose answer we control**:

* separable data  -> the probe MUST score MCC ~1.0. Catches a trainer that doesn't learn
  (e.g. a missing ``optimizer.step()``, a bad lr, an inverted class weight).
* pure-noise data -> the probe MUST score MCC ~0 (chance). Catches a trainer that
  "succeeds" through leakage (e.g. val rows leaking into train).

A probe that passes both is measuring the *data*, not itself -- which is exactly the
property you need before you can trust any number it reports about the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.probe.linear_probe import LinearProbe, train_probe

D_MODEL = 32
N_SAMPLES = 600


def _separable(n: int = N_SAMPLES, d: int = D_MODEL, seed: int = 0) -> tuple:
    """
    Two Gaussian blobs pushed apart along a random direction -> linearly separable.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    X = rng.normal(size=(n, d)) + 5.0 * y[:, None] * direction[None, :]
    return X.astype(np.float32), y.astype(np.int64)


def _noise(n: int = N_SAMPLES, d: int = D_MODEL, seed: int = 0) -> tuple:
    """Features carry zero information about the labels."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32), rng.integers(0, 2, size=n).astype(np.int64)


def test_probe_recovers_linearly_separable_signal():
    """A probe that trains correctly must near-perfectly recover a planted linear boundary."""
    X, y = _separable()
    _probe, metrics = train_probe(X, y, n_classes=2, epochs=300, lr=1e-2)
    assert metrics['mcc_val'] > 0.95, f'probe failed to learn a separable signal: {metrics}'


def test_probe_finds_no_signal_in_noise():
    """On noise, the probe's MCC must stay near chance (0)."""
    X, y = _noise()
    _probe, metrics = train_probe(X, y, n_classes=2, epochs=300, lr=1e-2)
    assert abs(metrics['mcc_val']) < 0.2, (
        f'probe "found" signal in pure noise -- suspect leakage between train/val: {metrics}'
    )


def test_weights_actually_change():
    """Regression test for the missing-``optimizer.step()`` class of bug.

    Without ``step()``, ``backward()`` populates ``.grad`` and the loop still runs to
    completion -- reporting a plausible F1 from an *untrained* probe. Assert the parameters
    moved.
    """
    X, y = _separable()
    before = LinearProbe(d_model=D_MODEL, n_classes=2).linear.weight.detach().clone()
    probe, _metrics = train_probe(X, y, n_classes=2, epochs=50, lr=1e-2, seed=0)
    after = probe.linear.weight.detach()
    assert not np.allclose(before.numpy(), after.numpy()), 'probe weights never moved'


def test_multiclass_mcc_runs():
    """`value(A)` is 24-way: MCC must handle the multiclass case, not just binary."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 24, size=N_SAMPLES).astype(np.int64)
    centers = rng.normal(size=(24, D_MODEL)) * 4.0
    X = (centers[y] + rng.normal(size=(N_SAMPLES, D_MODEL))).astype(np.float32)
    _probe, metrics = train_probe(X, y, n_classes=24, epochs=300, lr=1e-2)
    assert metrics['mcc_val'] > 0.9, f'failed on separable 24-class data: {metrics}'


def test_class_balancing_upweights_rare_class():
    """`balance_classes` must use INVERSE frequency (rare class gets recalled, not ignored)."""
    rng = np.random.default_rng(0)
    y = (rng.random(N_SAMPLES) < 0.05).astype(np.int64)  # 5% positive: heavy imbalance
    direction = rng.normal(size=D_MODEL)
    direction /= np.linalg.norm(direction)
    X = (rng.normal(size=(N_SAMPLES, D_MODEL)) + 1.5 * y[:, None] * direction).astype(np.float32)

    _p, balanced = train_probe(X, y, n_classes=2, epochs=200, lr=1e-2, balance_classes=True)
    # Without upweighting, the probe predicts all-zeros on 5%-positive data -> MCC == 0.
    assert balanced['mcc_val'] > 0.0, f'balanced probe never predicts the rare class: {balanced}'


def test_class_balancing_beats_no_balancing():
    """The *contrast* the previous test asserts in prose: on heavy imbalance, balancing is
    what recovers the rare class. Same data through both settings -- unbalanced collapses to
    the majority (MCC ~ 0), balanced scores meaningfully higher. Without this comparison, the
    single-setting test can't tell "balancing helped" from "the signal was easy anyway".
    """
    rng = np.random.default_rng(0)
    y = (rng.random(N_SAMPLES) < 0.05).astype(np.int64)  # 5% positive
    direction = rng.normal(size=D_MODEL)
    direction /= np.linalg.norm(direction)
    X = (rng.normal(size=(N_SAMPLES, D_MODEL)) + 1.5 * y[:, None] * direction).astype(np.float32)

    _p, unbalanced = train_probe(X, y, n_classes=2, epochs=200, lr=1e-2, balance_classes=False, seed=0)
    _p, balanced = train_probe(X, y, n_classes=2, epochs=200, lr=1e-2, balance_classes=True, seed=0)
    assert unbalanced['mcc_val'] < 0.05, f'unbalanced probe should ignore the rare class: {unbalanced}'
    assert balanced['mcc_val'] > unbalanced['mcc_val'] + 0.15, (
        f'balancing did not recover the rare class: {unbalanced} vs {balanced}'
    )


def test_group_split_prevents_leakage():
    """The whole point of the ``groups=`` argument.

    Construct data where every row's features are just a per-*problem* fingerprint (+ noise)
    and each problem's label is random. There is no cross-problem structure to learn, so the
    only way to score on held-out data is to have *seen that problem's fingerprint in train*:

      * row-level split (``groups=None``) puts some rows of every problem in train -> the probe
        memorises fingerprints and val MCC hits ~1.0. That number is pure leakage.
      * group-level split (``groups=g``) sends whole problems to val -> their fingerprints are
        unseen, labels are random, and MCC collapses to chance.

    A regression that made the group path silently fall back to row splitting would flip the
    second number back up to ~1.0 and this test would catch it.
    """
    rng = np.random.default_rng(0)
    n_groups, per_group, d = 30, 20, 64
    fingerprint = rng.normal(size=(n_groups, d)) * 3.0
    group_label = rng.integers(0, 2, size=n_groups)  # random -> nothing generalisable
    X, y, g = [], [], []
    for gi in range(n_groups):
        for _ in range(per_group):
            X.append(fingerprint[gi] + rng.normal(size=d) * 0.3)
            y.append(group_label[gi])
            g.append(gi)
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int64)
    g = np.asarray(g, np.int64)

    _p, leaky = train_probe(X, y, n_classes=2, epochs=300, lr=1e-2, seed=0)
    _p, grouped = train_probe(X, y, groups=g, n_classes=2, epochs=300, lr=1e-2, seed=0)
    assert leaky['mcc_val'] > 0.9, f'row split should memorise fingerprints: {leaky}'
    assert grouped['mcc_val'] < 0.3, f'group split still leaked -- rows crossed the split: {grouped}'
    assert leaky['mcc_val'] - grouped['mcc_val'] > 0.5, (
        f'grouping removed no leaked signal: row={leaky} group={grouped}'
    )


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_deterministic_given_seed(seed: int):
    """Same seed -> same metrics (probe init + split + shuffle are all seeded)."""
    X, y = _separable()
    _p1, m1 = train_probe(X, y, n_classes=2, epochs=50, seed=seed)
    _p2, m2 = train_probe(X, y, n_classes=2, epochs=50, seed=seed)
    assert m1 == m2
