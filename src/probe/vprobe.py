"""V-probing (paper section 4.1, Figure 6): nearly-linear probing with the query in the input.

The linear probe in `probe.py` is degenerate for per-parameter questions: one hidden state,
many labels. V-probing fixes this by injecting the *queried parameter A into the input*:

    input = [BOS] problem question [START] <description of A> [END]
    logits = linear_head( last_layer_hidden_state[at END] )

Two small things train jointly (the LM stays frozen):
  * the linear classification head, and
  * a rank-8 update `delta_a @ delta_b` on the input embedding -- needed because [START]/[END]
    (ids 225/226) never occur in iGSM text; through weight tying their pretrained embedding
    rows collapsed to near-identical "never the next token" vectors, so without this update
    the model literally cannot distinguish the probe markers from each other.

Unlike the cached-activation pipeline in `extract.py`, V-probe training must forward through
the frozen LM every step: gradients have to flow *through* the transformer to reach the
embedding delta. So this module owns its own dataset (token sequences, not hidden states).

Paper control: run the identical procedure on a randomly-initialised model (`--random-model`).
The pretrained-vs-random gap is the evidence; the absolute number alone is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from sklearn.metrics import matthews_corrcoef
from torch import nn
from tqdm import tqdm

from src.data.igsm import EOS, ensure_igsm_submodule
from src.probe.labels import regenerate_problem

START, END = 225, 226  # byte-fallback ids, same never-in-ASCII dead zone as 222/223/224

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
    print(f'VRAM guardrail: capped at {vram_fraction:.0%} of {total:.1f} GiB')


# --------------------------------------------------------------------------- #
# Dataset: (token sequence, label, group) rows
# --------------------------------------------------------------------------- #


@dataclass
class VProbeRow:
    """One (problem, parameter) query: `input_ids` ends with [START] desc(A) [END]."""

    input_ids: list[int]
    label: int
    group: int  # problem seed; split train/val by this, never by row


def build_vprobe_rows(
    n_problems: int,
    *,
    target: str = 'nece',
    split: str = 'test',
    seed_start: int = 0,
) -> list[VProbeRow]:
    """Build V-probe rows for `target` over `n_problems` regenerated problems.

    For `nece` the paper probes at the end of the question, so the prefix is the full
    problem+question (everything before the [223] solution marker), preceded by EOS --
    matching pretraining, where every problem is preceded by the previous one's EOS.
    The parameter description comes from iGSM's own `Problem.get_param` (the same
    "each X's Y" phrasing used in problem sentences) and is wrapped in [START]/[END].
    """
    ensure_igsm_submodule()
    from tools.tools import tokenizer  # iGSM's GPT-2 tokenizer

    assert target == 'nece', f'only nece is implemented so far, got {target!r}'
    rows: list[VProbeRow] = []
    skipped = 0
    for seed in tqdm(range(seed_start, seed_start + n_problems), desc='build rows', unit='prob'):
        pp = regenerate_problem(seed, split=split)
        prefix = [EOS, *pp.token_id[: pp.sol_bos_index]]  # [EOS] [222] problem+question
        for p_idx, param in enumerate(pp.all_param):
            desc = tokenizer.encode(' ' + pp.problem.get_param(param))
            input_ids = [*prefix, START, *desc, END]
            if len(input_ids) > MAX_SEQ_LEN:
                skipped += 1  # guardrail: one pathological row must not set the batch's memory
                continue
            rows.append(
                VProbeRow(input_ids=input_ids, label=int(pp.nece[p_idx]), group=seed)
            )
    if skipped:
        print(f'dropped {skipped} rows longer than MAX_SEQ_LEN={MAX_SEQ_LEN}')
    lens = [len(r.input_ids) for r in rows]
    print(f'{len(rows)} rows, token length: median {int(np.median(lens))}, max {max(lens)}')
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
    probe: VProbe, rows: list[VProbeRow], batch_size: int, device: str
) -> dict[str, float]:
    """Accuracy + MCC over `rows` (accuracy for comparability with the paper's Figure 7)."""
    probe.eval()
    preds: list[int] = []
    labels: list[int] = []
    # length-sorted so eval batches don't pad to the global max either
    order = sorted(range(len(rows)), key=lambda i: len(rows[i].input_ids))
    for start in range(0, len(order), batch_size):
        batch = [rows[i] for i in order[start : start + batch_size]]
        ids, mask, end, y = _pad_batch(batch, device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')):
            logits = probe(ids, mask, end)
        preds.extend(logits.float().argmax(dim=-1).cpu().tolist())
        labels.extend(y.cpu().tolist())
    preds_arr, labels_arr = np.asarray(preds), np.asarray(labels)
    majority = np.bincount(labels_arr).argmax()
    return {
        'acc': float((preds_arr == labels_arr).mean()),
        'mcc': float(matthews_corrcoef(labels_arr, preds_arr)),
        'acc_majority': float((labels_arr == majority).mean()),  # the paper's baseline row
    }


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
    grad_checkpointing: bool = True,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
    device: str = 'cuda',
    seed: int = 0,
) -> tuple[VProbe, dict[str, float]]:
    """Train head + embedding delta on `rows` through the frozen `lm`; report val metrics.

    Returns `(probe, metrics)` with `acc_val` / `mcc_val`, the train-set counterparts
    (memorisation check), `acc_majority` (the paper's baseline), and `peak_vram_gib`.
    Interpretation needs the random-model control: rerun with a random-init lm and compare.

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
    loss_fn = nn.CrossEntropyLoss()

    indices = np.arange(len(train_rows))
    for epoch in range(epochs):
        probe.train()
        batches = _length_bucketed_batches(train_rows, indices, batch_size)
        rng.shuffle(batches)  # shuffle batch order, keep within-batch lengths homogeneous
        total_loss = 0.0
        for batch in tqdm(batches, desc=f'epoch {epoch + 1}/{epochs}', unit='batch'):
            ids, mask, end, y = _pad_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=device.startswith('cuda')
            ):
                loss = loss_fn(probe(ids, mask, end), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f'epoch {epoch + 1}: train loss {total_loss / len(train_rows):.4f}')

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
    }
    if device.startswith('cuda'):
        metrics['peak_vram_gib'] = torch.cuda.max_memory_allocated() / 2**30
    return probe, metrics


def load_lm(model_path: str | None, device: str = 'cuda'):
    """Load the pretrained GPT2-RoPE, or a fresh random-init one if `model_path` is None.

    The random-init model is the paper's control: whatever the V-probe scores on it is the
    capability added by the probe's own finetuning, not knowledge read out of pretraining.
    """
    from src.model.gpt2_rope import GPT2LMHeadModelWithRoPE, build_gpt2_rope

    if model_path is None:
        model = build_gpt2_rope()
    else:
        model = GPT2LMHeadModelWithRoPE.from_pretrained(model_path)
    model.config.use_cache = False
    model.eval().to(device)
    return model
