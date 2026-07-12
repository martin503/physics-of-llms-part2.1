"""GPT-2 + RoPE + QK-normalization (numerically stabilized variant of gpt2_rope.py).

Identical to ``src/model/gpt2_rope.py`` except each self-attention layer applies
**QK-norm**: an RMSNorm over the head dimension on Q and K (after RoPE, before the
attention dot product). This bounds |Q|, |K| (and therefore attention scores) regardless
of how high-gain the ``c_attn`` projection becomes during training.

Motivation: the plain RoPE models develop a high-gain forward pass when memorizing iGSM
(block0 residual ~500, attention scores at the softmax-overflow cliff) -> nondeterministic
NaN at inference -> greedy argmax = token 0 -> 0% eval. QK-norm keeps scores ~sqrt(head_dim),
moving the forward off the cliff while still letting the model fit.

Drop-in builders mirror gpt2_rope.py: ``build_gpt2_rope_qknorm`` / ``GPT2LMHeadModelWithRoPEQK``.
The attention ``forward`` body is copied from ``GPT2AttentionWithRoPE.forward`` with the
QK-norm insertion marked.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from transformers.cache_utils import Cache, EncoderDecoderCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2LMHeadModel,
    GPT2Model,
    eager_attention_forward,
)

from src.model.gpt2_rope import (
    GPT2AttentionWithRoPE,
    GPT2BlockWithRoPE,
    apply_rotary_pos_emb,
    build_gpt2_config,
)


class HeadRMSNorm(nn.Module):
    """RMSNorm over the last dim (the head dimension of Q / K)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = (x.float() * torch.rsqrt(var + self.eps))
        return (self.weight.to(x.dtype) * x).to(dtype)


class GPT2AttentionWithRoPEQK(GPT2AttentionWithRoPE):
    """RoPE self-attention + QK-norm (RMSNorm on Q and K after RoPE)."""

    def __init__(self, config, is_cross_attention: bool = False, layer_idx: int | None = None) -> None:
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.q_norm = HeadRMSNorm(self.head_dim)
        self.k_norm = HeadRMSNorm(self.head_dim)

    def forward(  # noqa: C901, PLR0912 -- mirrors GPT2AttentionWithRoPE.forward + QK-norm
        self,
        hidden_states,
        past_key_values: Cache | None = None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        position_ids = kwargs.get('position_ids', None)
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
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
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        # ----- RoPE (self-attention only) -----
        if not is_cross_attention and position_ids is not None:
            seq_len_q = query_states.shape[-2]
            pos = position_ids[:, -seq_len_q:]
            cos, sin = self.rotary_emb(pos)
            cos = cos.to(query_states.dtype)
            sin = sin.to(query_states.dtype)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # ----- QK-norm (NEW): bound Q,K so attention scores stay off the softmax-overflow cliff. -----
        if not is_cross_attention:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)
        # -----------------------------------------------------------------------------------------------

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            key_states, value_states = curr_past_key_values.update(key_states, value_states, self.layer_idx)
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


class GPT2BlockWithRoPEQK(GPT2BlockWithRoPE):
    """RoPE block whose self-attention uses QK-norm."""

    def __init__(self, config, layer_idx: int | None = None) -> None:
        super().__init__(config, layer_idx=layer_idx)
        self.attn = GPT2AttentionWithRoPEQK(config=config, layer_idx=layer_idx)


class GPT2ModelWithRoPEQK(GPT2Model):
    """GPT-2 base model with RoPE + QK-norm attention blocks."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.h = nn.ModuleList(
            [GPT2BlockWithRoPEQK(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.post_init()


class GPT2LMHeadModelWithRoPEQK(GPT2LMHeadModel):
    """GPT-2 causal LM with RoPE + QK-norm. Inherits ``forward``."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.transformer = GPT2ModelWithRoPEQK(config)
        self.post_init()


def build_gpt2_rope_qknorm(config=None, *, attn_implementation: str = 'sdpa') -> GPT2LMHeadModelWithRoPEQK:
    """Build a from-scratch GPT2-12-12 + RoPE + QK-norm model (wpe zeroed & frozen)."""
    config = config if config is not None else build_gpt2_config()
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
    model = GPT2LMHeadModelWithRoPEQK(config)
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    model.transformer.wpe.requires_grad_(False)
    return model
