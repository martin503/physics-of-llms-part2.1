# Tests

Run from the repo root (needs the project root on `sys.path`, which `-m` provides):

```bash
uv run python -m pytest tests/ -q
```

## What these tests can and can't do

The probe pipeline has **no independent ground truth** — nobody can tell you, from first
principles, whether `nece(A)` is linearly decodable from layer 6. So the tests never assert a
number about the *model*. Instead they validate the **instrument**: feed the trainer synthetic
data whose answer we control, and check it reports the answer we planted. A probe that scores
~1.0 on separable data and ~0.0 on noise is measuring the data, not itself — which is the
property you need before you trust any number it later reports about the real model.

That means the **model-dependent code is deliberately not unit-tested** (see gaps below); it's
integration-shaped and needs the frozen LM + iGSM submodule in the loop.

## Coverage

### `test_probe.py` — the linear probe trainer (`src/probe/probe.py`)

| Test | Guarantees | Catches |
|---|---|---|
| `test_probe_recovers_linearly_separable_signal` | Separable data → MCC > 0.95 | Dead trainer: missing `optimizer.step()`, bad lr, inverted class weight |
| `test_probe_finds_no_signal_in_noise` | Pure-noise data → \|MCC\| < 0.2 | A trainer that manufactures signal from nothing |
| `test_weights_actually_change` | Params move after training | The specific missing-`step()` bug that still "completes" with an untrained probe |
| `test_multiclass_mcc_runs` | 24-class separable data → MCC > 0.9 | Binary-only assumptions breaking the `value(A)` (24-way) path |
| `test_class_balancing_upweights_rare_class` | `balance_classes=True` predicts the 5%-rare class at all | Non-inverse / no class weighting |
| `test_class_balancing_beats_no_balancing` | Same data: balanced MCC > unbalanced (which ≈ 0) | Balancing that looks on but does nothing — the single-setting test above can't tell "balancing helped" from "easy signal" |
| `test_group_split_prevents_leakage` | `groups=` sends whole problems to val; correlated-row leakage collapses MCC to chance while a row split memorises to ~1.0 | A regression where the group path silently falls back to row-level splitting |
| `test_deterministic_given_seed` | Same seed → identical metrics | Unseeded init / split / shuffle |

### `test_vprobe.py` — the V-probe's pure helpers (`src/probe/vprobe.py`)

`VProbe`/`train_vprobe` need the LM in the loop and aren't unit-tested. These three helpers are
plain array/list logic and two of them encode correctness guarantees the code leans on:

| Test(s) | Guarantees | Catches |
|---|---|---|
| `test_pad_batch_*` | `end_index` = last real token; short rows right-padded with EOS and masked; labels preserved | Off-by-one that reads the hidden state at a pad token → probe trains on noise |
| `test_split_by_group_*` | No problem in both splits; every group placed; ≥1 val group always | The V-probe's own anti-leakage invariant breaking |
| `test_length_bucketed_batches_*` | Bucketing covers every row exactly once, respects `batch_size`, groups similar lengths | A memory optimisation that drops/duplicates rows or stops bucketing |

### `test_evaluate.py` — probe test-time evaluation (`src/probe/evaluate.py`)

| Test(s) | Guarantees | Catches |
|---|---|---|
| `test_classification_metrics_*` | Confusion cells counted into the right buckets; MCC ±1 on perfect/inverted; single-class labels don't crash | A swapped cell silently mislabelling every edge colour in the dep report |
| `test_predict_vprobe_restores_input_order` | `preds[i]` belongs to `rows[i]` despite internal length-sorted batching (checked with a stub probe whose output is a deterministic function of each row) | A mis-scatter assigning predictions to the wrong (A, B) pairs — invisible in aggregate metrics, fatal for the graph report |

## Known gaps (intentional, integration-shaped)

These have **no unit tests** because they need the frozen model and/or the iGSM submodule:

- **`src/probe/extract.py`** — the frozen forward pass and hidden-state caching. Correct
  layer indexing (`hidden_states[layer]`) and read-position selection are only exercised end
  to end.
- **`src/probe/vprobe.py` training loop** — `VProbe` (embedding delta + frozen LM + head),
  `train_vprobe`, grad checkpointing, the VRAM guardrails. Only the pure helpers above are unit-tested.
- **`src/probe/labels.py`** — label extraction and token/step alignment against real iGSM
  `Problem` objects. The `assert` in `_solution_step_positions` is the in-code guard here; there's
  no external check that the labels themselves are right.
- **`mcc_train`** is returned but not asserted anywhere; the memorisation check
  (`mcc_train >> mcc_val`) is documented in code comments, not enforced by a test.
