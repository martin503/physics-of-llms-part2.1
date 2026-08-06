"""V-probing (paper section 4.1, Figures 6 & 13): nearly-linear probing with the query in the input.

The linear probe in `probe.py` is degenerate for per-parameter questions: one hidden state,
many labels. V-probing fixes this by injecting the *queried parameter(s) into the input*:

    nece(A):    input = [EOS] problem question [START] desc(A) [END]
    dep(A, B):  input = [EOS] problem          [START] desc(A) [MID] desc(B) [END]
    logits = linear_head( last_layer_hidden_state[at END] )

The two tasks differ in read position, matching the paper (Figure 13):
  * nece(A) is probed at the end of the *question* (our `sol_bos_index`);
  * dep(A, B) is probed at the end of the *problem description*, before the question is asked
    (the question tokens are dropped), and injects two parameter descriptions separated by [MID].

Two small things train jointly (the LM stays frozen):
  * the linear classification head, and
  * a rank-8 update `delta_a @ delta_b` on the input embedding -- needed because [START]/[MID]/[END]
    (ids 225/227/226) never occur in iGSM text; through weight tying their pretrained embedding
    rows collapsed to near-identical "never the next token" vectors, so without this update
    the model literally cannot distinguish the probe markers from each other.

Unlike the cached-activation pipeline in `extract.py`, V-probe training must forward through
the frozen LM every step: gradients have to flow *through* the transformer to reach the
embedding delta. So this module owns its own dataset (token sequences, not hidden states).

Paper control: run the identical procedure on a randomly-initialised model (`--random-model`).
The pretrained-vs-random gap is the evidence; the absolute number alone is not.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn.metrics import matthews_corrcoef
from torch import nn
from tqdm import tqdm

from src.data.igsm import EOS, ensure_igsm_submodule
from src.probe.labels import ProbeProblem, problem_desc_end_index, regenerate_problem

logger = logging.getLogger(__name__)

# Byte-fallback ids in the same never-in-ASCII dead zone as 222/223/224 (verified: 0 occurrences
# in 300 iGSM-med test problems). [MID] separates the two parameter descriptions in dep(A, B).
START, MID, END = 225, 227, 226


# Resource guardrails. On Windows/WDDM, exhausting VRAM does NOT raise OOM: the driver silently
# spills to "shared GPU memory" (= system RAM) and the machine thrashes over PCIe until it is
# unresponsive. `set_per_process_memory_fraction` makes PyTorch's allocator refuse to grow past
# a fraction of VRAM and raise a clean OOM instead, so a too-large run fails fast rather than
# taking the desktop down with it.
DEFAULT_VRAM_FRACTION = 0.85
MAX_SEQ_LEN = 1024  # rows longer than this are dropped (iGSM-med problems are far shorter)


def apply_memory_guardrails(device: str, vram_fraction: float = DEFAULT_VRAM_FRACTION) -> None:
    """Cap the allocator so VRAM exhaustion raises OOM instead of spilling into system RAM."""
    if not device.startswith('cuda') or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(vram_fraction)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    logger.info('VRAM guardrail: capped at %.0f%% of %.1f GiB', vram_fraction * 100, total)


# --------------------------------------------------------------------------- #
# Dataset: (token sequence, label, group) rows
# --------------------------------------------------------------------------- #


@dataclass
class VProbeRow:
    """One probe query. `input_ids` ends with the injected parameter block: [START] desc(A) [END]
    for nece, or [START] desc(A) [MID] desc(B) [END] for dep. The head reads the final token.

    `param_a`/`param_b` index the queried A and B into `ProbeProblem.all_param` (B is -1 for
    nece rows), so predictions map back onto the dependency graph by regenerating the
    problem. `n_op` is the source problem's reasoning-step count, carried so a dataset's
    difficulty mix can be read off it directly. Training ignores all three.
    """

    input_ids: list[int]
    label: int
    group: int  # problem seed; split train/val by this, never by row
    param_a: int = -1
    param_b: int = -1
    n_op: int = -1


def rows_for_problem(
    seed: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    med_cfg: dict[str, Any] | None = None,
    max_seq_len: int = MAX_SEQ_LEN,
    dep_all_pairs: bool = False,
) -> tuple[list[VProbeRow], int]:
    """Build the V-probe rows for one regenerated problem; return `(rows, n_skipped)`.

    Targets share the candidate parameters (`labels.named_params`) and differ only in their
    builder (:data:`ROW_BUILDERS`), which owns the input layout and read position.
    `dep_all_pairs` keeps every ordered off-diagonal pair instead of the balanced subsample
    -- the natural distribution used for testing; one-parameter targets ignore it.

    `(seed, split, med_cfg, target)` fixes the result, including `dep`'s seeded negative
    sampling, so the multiprocess generation in `src.probe.data` stays byte-identical
    however the seeds are split across workers.
    """
    if target not in ROW_BUILDERS:
        raise ValueError(f'target must be one of {TARGETS}, got {target!r}')
    pp = regenerate_problem(seed, split=split, med_cfg=med_cfg)
    return ROW_BUILDERS[target](pp, seed, max_seq_len, dep_all_pairs)


def _nece_rows_for_problem(
    pp: ProbeProblem, seed: int, max_seq_len: int, all_pairs: bool = False
) -> tuple[list[VProbeRow], int]:
    """`nece(A)` rows: one per candidate parameter, read at the end of the question.

    Reached through `rows_for_problem(target='nece')`, which supplies `pp`.

    The prefix is the full problem+question (everything before the [223] solution marker),
    preceded by EOS -- matching pretraining, where every problem is preceded by the previous
    one's EOS. The parameter description comes from iGSM's own `Problem.get_param` (the same
    "each X's Y" phrasing used in problem sentences) and is wrapped in [START]/[END].
    `all_pairs` is a pairwise-target option and does nothing for a one-parameter query.
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    prefix = [EOS, *pp.token_id[: pp.sol_bos_index]]  # [EOS] [222] problem+question
    rows: list[VProbeRow] = []
    skipped = 0
    for p_idx, param in enumerate(pp.all_param):
        desc = tokenizer.encode(' ' + pp.problem.get_param(param))
        input_ids = [*prefix, START, *desc, END]
        if len(input_ids) > max_seq_len:
            skipped += 1  # guardrail: one pathological row must not set the batch's memory
            continue
        rows.append(
            VProbeRow(
                input_ids=input_ids, label=int(pp.nece[p_idx]), group=seed, param_a=p_idx,
                n_op=pp.problem.n_op,
            )
        )
    return rows, skipped


def _dep_rows_for_problem(
    pp: ProbeProblem, seed: int, max_seq_len: int, all_pairs: bool = False
) -> tuple[list[VProbeRow], int]:
    """`dep(A, B)` rows: balanced (A, B) pairs, read at the end of the problem description.

    Reached through `rows_for_problem(target='dep')`, which supplies `pp`.

    Input layout (paper Figure 13b / Appendix B): the question is dropped and the two parameter
    descriptions are injected in one [START]..[END] block separated by [MID]::

        [EOS] [222] <problem description> [START] desc(A) [MID] desc(B) [END]
                                                                        ^ read the head here

    Label = `dep(A, B)` = "does A (recursively) depend on B" = `pp.dep()[a, b]`.

    Pair selection (the `--target dep` universe): the full dep matrix is `n_param x n_param`, so we
    keep **all** positive pairs and sample an **equal** number of negatives (both off-diagonal),
    giving a per-problem-balanced dataset. Sampling is seeded from the problem `seed`, so offline
    generation stays deterministic. Unlike `nece`, no `--balance-classes` is needed:
    the classes are ~50/50 by construction (the paper's natural dep distribution is ~83% negative).

    `all_pairs=True` skips the subsampling and keeps every ordered off-diagonal pair -- the
    natural (heavily negative) distribution. Use it for *test* datasets, where predictions are
    mapped back onto the full dependency graph; training on it would need class re-weighting.
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    prefix = [EOS, *pp.token_id[: problem_desc_end_index(pp) + 1]]  # [EOS] [222] problem description
    dep = pp.dep()  # (n_param, n_param); dep[a, b] == 1 iff param a depends on param b
    n = len(pp.all_param)
    off_diag = ~np.eye(n, dtype=bool)  # a param never "depends on itself" (dep diagonal is 0)
    pos = np.argwhere((dep == 1) & off_diag)
    neg = np.argwhere((dep == 0) & off_diag)

    if not all_pairs:
        rng = np.random.default_rng(seed)  # per-problem: keeps offline generation reproducible
        n_neg = min(len(pos), len(neg))  # balance 1:1; keep all positives, subsample negatives
        if len(neg) > n_neg:
            neg = neg[rng.choice(len(neg), size=n_neg, replace=False)]

    # get_param is not free, so encode each parameter's description once (n_param is small)
    desc_ids = [tokenizer.encode(' ' + pp.problem.get_param(param)) for param in pp.all_param]

    rows: list[VProbeRow] = []
    skipped = 0
    for pairs, label in ((pos, 1), (neg, 0)):
        for a, b in pairs:
            input_ids = [*prefix, START, *desc_ids[a], MID, *desc_ids[b], END]
            if len(input_ids) > max_seq_len:
                skipped += 1  # guardrail: one pathological row must not set the batch's memory
                continue
            rows.append(
                VProbeRow(
                    input_ids=input_ids, label=label, group=seed,
                    param_a=int(a), param_b=int(b), n_op=pp.problem.n_op,
                )
            )
    return rows, skipped


# Probe tasks, each mapped to the builder turning one problem into its rows. Builders share
# the signature `(pp, seed, max_seq_len, all_pairs)`, so dispatch needs no per-target
# branching and each owns its own read position. A new probe is one entry plus its builder.
ROW_BUILDERS: dict[str, Callable[[ProbeProblem, int, int, bool], tuple[list[VProbeRow], int]]] = {
    'nece': _nece_rows_for_problem,
    'dep': _dep_rows_for_problem,
}
TARGETS = tuple(ROW_BUILDERS)


def build_vprobe_rows(
    n_problems: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    seed_start: int = 0,
    dep_all_pairs: bool = False,
) -> list[VProbeRow]:
    """Build V-probe rows over `n_problems` problems, in-process (see `rows_for_problem`).

    Convenient for small runs; for the row counts that actually avoid overfitting,
    generate offline with `python -m src.probe.run gen-data` and pass `--data` instead.
    """
    rows: list[VProbeRow] = []
    skipped = 0
    for seed in tqdm(range(seed_start, seed_start + n_problems), desc='build rows', unit='prob'):
        problem_rows, n_skipped = rows_for_problem(
            seed, target=target, split=split, dep_all_pairs=dep_all_pairs
        )
        rows.extend(problem_rows)
        skipped += n_skipped
    if skipped:
        logger.info('dropped %d rows longer than MAX_SEQ_LEN=%d', skipped, MAX_SEQ_LEN)
    lens = [len(r.input_ids) for r in rows]
    logger.info(
        '%d rows, token length: median %d, max %d', len(rows), int(np.median(lens)), max(lens)
    )
    return rows


# --------------------------------------------------------------------------- #
# Model: frozen LM + rank-r embedding delta + linear head
# --------------------------------------------------------------------------- #


class VProbe(nn.Module):
    """Frozen GPT2-RoPE + trainable rank-`rank` embedding update + linear head at [END].

    The embedding update is LoRA-style: `wte(ids) + delta_a[ids] @ delta_b`, with `delta_a`
    zero-initialised so training starts exactly at the pretrained model's behaviour.
    """

    def __init__(
        self, lm, n_classes: int = 2, rank: int = 8, grad_checkpointing: bool = True
    ) -> None:
        super().__init__()
        self.transformer = lm.transformer  # GPT2ModelWithRoPE (returns last_hidden_state)
        for p in self.transformer.parameters():
            p.requires_grad_(False)
        if grad_checkpointing:
            # The trainable embedding delta sits at the BOTTOM of the network, so backprop must
            # traverse all 12 blocks and autograd would otherwise store every block's
            # activations for the whole batch (the dominant memory cost -- see docs). Checkpointing
            # keeps only block boundaries and recomputes the rest in backward: ~sqrt-ish memory
            # for ~30% more compute. use_reentrant=False is required for frozen-param modules.
            self.transformer.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
        vocab, d_model = self.transformer.wte.weight.shape
        self.delta_a = nn.Parameter(torch.zeros(vocab, rank))
        self.delta_b = nn.Parameter(torch.randn(rank, d_model) * 0.02)
        self.head = nn.Linear(d_model, n_classes)

    def train(self, mode: bool = True) -> VProbe:
        """Keep the frozen transformer in eval mode (no dropout) even while the probe trains."""
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(
        self,
        input_ids: Int[torch.Tensor, 'batch seq'],
        attention_mask: Int[torch.Tensor, 'batch seq'],
        end_index: Int[torch.Tensor, 'batch'],
    ) -> Float[torch.Tensor, 'batch n_classes']:
        emb = self.transformer.wte(input_ids) + self.delta_a[input_ids] @ self.delta_b
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
        out = self.transformer(
            inputs_embeds=emb, attention_mask=attention_mask, position_ids=position_ids
        )
        h_end = out.last_hidden_state[torch.arange(len(end_index)), end_index]
        return self.head(h_end)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _pad_batch(
    rows: list[VProbeRow], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad a batch with EOS; return (input_ids, attention_mask, end_index, labels).

    Right padding is safe here (unlike generation) because we read the hidden state at each
    row's own [END] position, and RoPE positions of masked pad tokens never enter attention.
    """
    max_len = max(len(r.input_ids) for r in rows)
    ids = torch.full((len(rows), max_len), EOS, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    end = torch.empty(len(rows), dtype=torch.long)
    labels = torch.empty(len(rows), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r.input_ids)
        ids[i, :n] = torch.tensor(r.input_ids, dtype=torch.long)
        mask[i, :n] = 1
        end[i] = n - 1  # the [END] token is always last
        labels[i] = r.label
    return ids.to(device), mask.to(device), end.to(device), labels.to(device)


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


def save_vprobe(probe: VProbe, path: Path | str) -> None:
    """Save only the *trainable* parts (delta_a, delta_b, head) -- a few MB, not the LM.

    The frozen transformer is reproducible from its own checkpoint dir, so duplicating
    ~500MB of weights per probe run would be waste; `load_vprobe` recombines them.
    """
    state = {
        k: v.cpu() for k, v in probe.state_dict().items() if not k.startswith('transformer.')
    }
    torch.save(
        {
            'state_dict': state,
            'rank': probe.delta_a.shape[1],
            'n_classes': probe.head.out_features,
        },
        str(path),
    )


def load_vprobe(
    path: Path | str, lm, device: str = 'cuda', grad_checkpointing: bool = False
) -> VProbe:
    """Rebuild a `VProbe` from `save_vprobe` output around a freshly loaded `lm`."""
    ckpt = torch.load(str(path), map_location='cpu')
    probe = VProbe(
        lm, n_classes=ckpt['n_classes'], rank=ckpt['rank'], grad_checkpointing=grad_checkpointing
    )
    missing, unexpected = probe.load_state_dict(ckpt['state_dict'], strict=False)
    assert not unexpected, f'unexpected keys in probe checkpoint: {unexpected}'
    non_transformer_missing = [k for k in missing if not k.startswith('transformer.')]
    assert not non_transformer_missing, f'missing probe params: {non_transformer_missing}'
    return probe.to(device).eval()


def load_lm(model_path: str | None, device: str = 'cuda', seed: int | None = None):
    """Load the pretrained GPT2-RoPE, or a fresh random-init one if `model_path` is None.

    The random-init model is the paper's control: whatever the V-probe scores on it is the
    capability added by the probe's own finetuning, not knowledge read out of pretraining.

    `seed` (random-init only) makes the control's weights reproducible: a saved `probe.pt`
    is meaningless without the exact transformer it was trained through, and the random LM
    exists nowhere but this process. Training and later test evaluation must both call this
    with the run's recorded seed to get the *same* control model back.
    """
    from src.model.gpt2_rope import GPT2LMHeadModelWithRoPE, build_gpt2_rope, verify_rope_buffers

    if model_path is None:
        if seed is not None:
            torch.manual_seed(seed)
        model = build_gpt2_rope()
    else:
        model = GPT2LMHeadModelWithRoPE.from_pretrained(model_path)
        verify_rope_buffers(model)  # reject checkpoints that load with garbage inv_freq
    model.config.use_cache = False
    model.eval().to(device)
    return model
