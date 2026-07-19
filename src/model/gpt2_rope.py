"""GPT-2 with rotary positional embeddings (RoPE).

Reproduces the model of "Physics of Language Models: Part 2.1" (arXiv:2407.20311),
which is a stock HuggingFace GPT-2 whose absolute positional embeddings are replaced
by RoPE.

Implementation note: ``position_ids`` already flows ``GPT2Model`` -> ``GPT2Block`` (via
``**kwargs``) -> ``GPT2Attention`` (via ``**kwargs``) in transformers 5.12, so adding RoPE
only requires subclassing the attention and rotating ``query``/``key`` *after* the head
reshape and *before* the KV-cache update (so cached keys stay rotary-encoded). The unused
absolute positional embedding ``wpe`` is zero-initialised and frozen, which makes
``inputs_embeds + wpe(position_ids) == inputs_embeds`` without copying the ~110-line
``GPT2Model.forward``.

Adapted from ``transformers/models/gpt2/modeling_gpt2.py`` (transformers 5.12.1); the
``GPT2AttentionWithRoPE.forward`` body is copied verbatim with a single RoPE insertion.
"""

from collections.abc import Callable

import einops
import torch
from jaxtyping import Float, Int
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Attention,
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    eager_attention_forward,
)


class RotaryEmbedding(nn.Module):
    """Rotary embedding (full rotation over all ``dim`` head dims).

    Args:
        dim: Head dimension (e.g. 64 for GPT2-12-12).
        base: RoPE base frequency (``rope_theta``).
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=True)

    def forward(
        self, position_ids: Int[torch.Tensor, 'batch seq']
    ) -> tuple[
        Float[torch.Tensor, 'batch 1 seq dim'],
        Float[torch.Tensor, 'batch 1 seq dim'],
    ]:
        """Return ``(cos, sin)`` of shape ``(batch, 1, seq, dim)`` for the given positions."""
        freqs = torch.einsum('bi,j->bij', position_ids.float(), self.inv_freq)
        emb = einops.repeat(freqs, 'b s d -> b s (r d)', r=2)
        cos = emb.cos()[:, None, :, :]
        sin = emb.sin()[:, None, :, :]
        return cos, sin


def rotate_half(x: Float[torch.Tensor, '... dim']) -> Float[torch.Tensor, '... dim']:
    """Rotate the second half of the last dim: ``cat([-x[d/2:], x[:d/2]])``."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    q: Float[torch.Tensor, 'batch heads seq dim'],
    k: Float[torch.Tensor, 'batch heads seq dim'],
    cos: Float[torch.Tensor, 'batch 1 seq dim'],
    sin: Float[torch.Tensor, 'batch 1 seq dim'],
) -> tuple[
    Float[torch.Tensor, 'batch heads seq dim'],
    Float[torch.Tensor, 'batch heads seq dim'],
]:
    """Apply rotary embeddings to ``q`` and ``k`` (cos/sin broadcast over heads)."""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class GPT2AttentionWithRoPE(GPT2Attention):
    """GPT-2 self-attention with rotary embeddings applied to queries and keys."""

    def __init__(
        self, config, is_cross_attention: bool = False, layer_idx: int | None = None
    ) -> None:
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim, base=getattr(config, 'rope_theta', 10000.0)
        )

    def forward(  # noqa: C901, PLR0912 -- mirrors the copied GPT2Attention.forward
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        # Position ids reach here via **kwargs from GPT2Model -> GPT2Block -> attn.
        # Read (do NOT pop): they must stay in ``kwargs`` so the attention interface receives
        # them too -- FlashAttention derives ``cu_seq_lens`` from position-id resets to block
        # cross-attention between packed sequences (TRL bfd packing + padding_free). Popping
        # here would silently make packed samples attend across problem boundaries.
        position_ids = kwargs.get('position_ids', None)
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, re-use cached key/value layers
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, 'q_attn'):
                raise ValueError(
                    'Cross attention requires `q_attn`; instantiate with '
                    '`GPT2Attention(..., is_cross_attention=True)`.'
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(
                    self.split_size, dim=2
                )
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # ----- RoPE insertion (self-attention only, before KV-cache update) -----
        # Applied to query/key after the head reshape and before the cache update so that
        # cached keys are rotary-encoded at their original positions (correct for generation).
        if not is_cross_attention and position_ids is not None:
            seq_len_q = query_states.shape[-2]
            pos = position_ids[:, -seq_len_q:]  # (batch, seq_q) absolute positions
            cos, sin = self.rotary_emb(pos)
            cos = cos.to(query_states.dtype)
            sin = sin.to(query_states.dtype)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # -----------------------------------------------------------------------

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx
            )
            # mark cross-attn layer as updated so its cache can be re-used
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == 'eager'
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights


class GPT2BlockWithRoPE(GPT2Block):
    """GPT-2 block whose self-attention uses RoPE."""

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.attn = GPT2AttentionWithRoPE(config=config, layer_idx=layer_idx)


class GPT2ModelWithRoPE(GPT2Model):
    """GPT-2 base model with RoPE attention blocks (absolute ``wpe`` disabled downstream)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.h = nn.ModuleList(
            [GPT2BlockWithRoPE(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        # Re-run weight init over the freshly swapped RoPE blocks and re-tie weights.
        self.post_init()


class GPT2LMHeadModelWithRoPE(GPT2LMHeadModel):
    """GPT-2 causal LM with RoPE. Inherits ``forward`` (calls ``self.transformer``)."""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.transformer = GPT2ModelWithRoPE(config)
        self.post_init()


def build_gpt2_config(
    *,
    n_layer: int = 12,
    n_head: int = 12,
    n_embd: int = 768,
    n_inner: int = 3072,
    vocab_size: int = 50257,
    n_positions: int = 2048,
    activation_function: str = 'gelu_new',
    rope_theta: float = 10000.0,
    attn_pdrop: float = 0.1,
    resid_pdrop: float = 0.1,
    embd_pdrop: float = 0.1,
    initializer_range: float = 0.02,
) -> GPT2Config:
    """Build a ``GPT2Config`` for the paper's GPT2-12-12 with ``rope_theta`` attached."""
    config = GPT2Config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_inner=n_inner,
        vocab_size=vocab_size,
        n_positions=n_positions,
        activation_function=activation_function,
        attn_pdrop=attn_pdrop,
        resid_pdrop=resid_pdrop,
        embd_pdrop=embd_pdrop,
        initializer_range=initializer_range,
        tie_word_embeddings=True,
    )
    # Extra attribute: absorbed by PreTrainedConfig, survives save/from_pretrained.
    config.rope_theta = rope_theta
    return config


def build_gpt2_rope(
    config: GPT2Config | None = None,
    *,
    attn_implementation: str = 'sdpa',
) -> GPT2LMHeadModelWithRoPE:
    """Build a from-scratch GPT2-12-12 + RoPE model with the dead ``wpe`` neutralised.

    ``wpe`` (absolute positional embedding) is zero-initialised and frozen so it contributes
    nothing -- positions are encoded solely by RoPE -- while keeping ``GPT2Model.forward``
    unchanged and avoiding checkpoint key mismatches.
    """
    config = config if config is not None else build_gpt2_config()
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    model = GPT2LMHeadModelWithRoPE(config)
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    # Does NOT survive save_pretrained/from_pretrained, but we dont care
    # cause we are NOT resuming trainings.
    model.transformer.wpe.requires_grad_(False)
    return model


def recompute_rope_inv_freq(model: GPT2LMHeadModelWithRoPE) -> None:
    """Recompute every RoPE ``inv_freq`` buffer in-place from ``config.rope_theta``.

    Belt-and-suspenders for continued training from an older checkpoint saved before
    ``inv_freq`` was made ``persistent``: such checkpoints lack the buffer, so
    ``from_pretrained`` leaves it as uninitialised heap memory (~1e38), ``pos * inv_freq``
    overflows to ``inf``/``NaN`` cosines -> garbage rotations and ~0% eval accuracy. With
    ``persistent=True`` checkpoints the loaded values are identical, so this is a no-op and
    always safe.
    """
    base = getattr(model.config, 'rope_theta', 10000.0)
    for block in model.transformer.h:
        rotary = block.attn.rotary_emb
        dim = rotary.inv_freq.shape[0] * 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        rotary.inv_freq.copy_(inv_freq)
