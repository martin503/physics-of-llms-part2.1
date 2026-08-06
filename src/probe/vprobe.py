"""V-probing (paper section 4.1, Figures 6 & 13): the model that reads a query-conditioned input.

The linear probe in `probe.py` is degenerate for per-parameter questions: one hidden state,
many labels. V-probing fixes this by injecting the *queried parameter(s) into the input* --
`queries.py` builds those inputs; this module is what consumes them::

    logits = linear_head( last_layer_hidden_state[at END] )

Two small things train jointly (the LM stays frozen):
  * the linear classification head, and
  * a rank-8 update `delta_a @ delta_b` on the input embedding -- needed because
    [START]/[MID]/[END] (ids 225/227/226) never occur in iGSM text; through weight tying their
    pretrained embedding rows collapsed to near-identical "never the next token" vectors, so
    without this update the model literally cannot distinguish the probe markers from each other.

The module also owns what pairs a trained probe back with its transformer: `save_vprobe`
stores only the trainable parts, and `load_lm` rebuilds the LM -- including the paper's
random-init control, which exists nowhere but its own seed. Fitting lives in `vprobe_train.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from jaxtyping import Float, Int
from torch import nn

from src.data.igsm import EOS
from src.probe.queries import VProbeRow

logger = logging.getLogger(__name__)


# Resource guardrails. On Windows/WDDM, exhausting VRAM does NOT raise OOM: the driver silently
# spills to "shared GPU memory" (= system RAM) and the machine thrashes over PCIe until it is
# unresponsive. `set_per_process_memory_fraction` makes PyTorch's allocator refuse to grow past
# a fraction of VRAM and raise a clean OOM instead, so a too-large run fails fast rather than
# taking the desktop down with it.
DEFAULT_VRAM_FRACTION = 0.85


def apply_memory_guardrails(device: str, vram_fraction: float = DEFAULT_VRAM_FRACTION) -> None:
    """Cap the allocator so VRAM exhaustion raises OOM instead of spilling into system RAM."""
    if not device.startswith('cuda') or not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(vram_fraction)
    torch.cuda.empty_cache()
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    logger.info('VRAM guardrail: capped at %.0f%% of %.1f GiB', vram_fraction * 100, total)


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


# --------------------------------------------------------------------------- #
# Persistence: the trainable parts, and the LM they were trained through
# --------------------------------------------------------------------------- #


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
