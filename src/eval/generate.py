"""Torch generation engine for the iGSM-med Figure-3 eval.

Generation is owned by torch (not inspect_ai) because the model prompt is a raw
token-id sequence ``[50256, 222, <problem>, 223]`` whose sentinel ids (222/223/224)
do not survive a decode->encode text round-trip under GPT-2 BPE (they merge at the
``222|problem`` boundary). So we feed raw ids straight to the model and let
inspect_ai own only scoring + logging.

Decoding is greedy (``do_sample=False``, ``num_beams=1``) with a KV-cache, stopping at
``EOS = 50256`` -- the paper's deterministic eval. Batched generation uses LEFT padding
(``pad_token_id = eos_token_id = 50256``) plus an attention mask. RoPE attention scores
depend only on *relative* positions, so an absolute shift of the real tokens by the pad
amount leaves every pairwise relative position unchanged; left-padded batched generation
is therefore exact (cross-checked against the unpadded single-sequence path in the eval
smoke test).
"""

from collections.abc import Sequence

import torch
from jaxtyping import Int
from tqdm import tqdm

EOS: int = 50256


class IgsmGenerator:
    """Load the GPT2-RoPE model once and greedily generate solutions from token ids.

    Args:
        model_path: HF checkpoint dir (e.g. ``models/gpt2-rope-igsm-pack/final``) whose
            ``config.json`` lists ``architectures: [GPT2LMHeadModelWithRoPE]``.
        device: Torch device (``cuda`` / ``cpu``).
        dtype: Inference dtype (``bfloat16`` matches training).
    """

    def __init__(
        self, model_path: str, device: str = 'cuda', dtype: torch.dtype = torch.float32
    ) -> None:
        from transformers import AutoTokenizer

        from src.model.gpt2_rope import GPT2LMHeadModelWithRoPE

        model = GPT2LMHeadModelWithRoPE.from_pretrained(model_path)
        model.config.use_cache = True  # saved config has use_cache=False -> force on for gen
        model.eval()
        model.to(device)
        model.to(dtype)
        # model = torch.compile(model)
        self.model = model
        self.device = device
        self.dtype = dtype
        self.n_positions: int = int(model.config.n_positions)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # decode-only, for logs
        except Exception:  # noqa: BLE001 -- decoding is best-effort; generation needs no tokenizer
            self.tokenizer = None

    @torch.inference_mode()
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int) -> list[int]:
        """Greedy-decode from ``prompt_ids`` (unpadded) until EOS; return new ids only."""
        cap = min(max_new_tokens, self.n_positions - len(prompt_ids))
        assert cap > 0, f'prompt ({len(prompt_ids)}) >= n_positions ({self.n_positions})'
        ids = torch.tensor([list(prompt_ids)], device=self.device, dtype=torch.long)
        out = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),  # explicit; pad==eos would otherwise warn
            do_sample=False,
            num_beams=1,
            max_new_tokens=cap,
            eos_token_id=EOS,
            pad_token_id=EOS,
        )
        return _strip_at_eos(out[0, len(prompt_ids) :].tolist())

    @torch.inference_mode()
    def batch_generate(
        self,
        prompts: Sequence[Sequence[int]],
        batch_size: int = 64,
        max_new_tokens: int = 2048,
    ) -> list[list[int]]:
        """Left-padded greedy batched generation; returns new ids per prompt (EOS-stripped)."""
        out: list[list[int]] = []
        for start in tqdm(range(0, len(prompts), batch_size), desc='generate', unit='batch'):
            batch = [list(p) for p in prompts[start : start + batch_size]]
            out.extend(self._generate_batch(batch, max_new_tokens))
        return out

    def _generate_batch(self, batch: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        max_len = max(len(p) for p in batch)
        cap = min(max_new_tokens, self.n_positions - max_len)
        assert cap > 0, f'batch max prompt ({max_len}) >= n_positions ({self.n_positions})'
        input_ids, attn = _left_pad(batch, max_len, pad=EOS, device=self.device)
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            do_sample=False,
            num_beams=1,
            max_new_tokens=cap,
            eos_token_id=EOS,
            pad_token_id=EOS,
        )
        return [_strip_at_eos(g) for g in out[:, max_len:].tolist()]

    def decode(self, ids: Sequence[int]) -> str:
        """Best-effort decode of token ids to text (for the eval log / debugging)."""
        if self.tokenizer is None:
            return ''
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)


def _strip_at_eos(ids: list[int]) -> list[int]:
    """Return ``ids`` truncated to just before the first EOS (50256), exclusive."""
    if EOS in ids:
        return ids[: ids.index(EOS)]
    return ids


def _left_pad(
    batch: list[list[int]], max_len: int, *, pad: int, device: str
) -> tuple[
    Int[torch.Tensor, 'batch seq'],
    Int[torch.Tensor, 'batch seq'],
]:
    """Left-pad ``batch`` to ``max_len`` with ``pad``; return ``(input_ids, attention_mask)``."""
    padded_ids = []
    attn_masks = []

    for p in batch:
        n = len(p)
        assert n <= max_len, 'prompt longer than the batch max_len'

        pad_length = max_len - n

        # Pure Python list concatenation is highly optimized
        padded_ids.append(([pad] * pad_length) + p)
        attn_masks.append(([0] * pad_length) + ([1] * n))

    # Create tensors and transfer to device in a single operation
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    attn = torch.tensor(attn_masks, dtype=torch.long, device=device)

    return input_ids, attn
