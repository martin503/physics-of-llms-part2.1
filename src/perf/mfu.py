"""FLOP / parameter / peak-FLOPS / MFU math for the GPT-2 + RoPE iGSM model.

Two FLOP conventions are exposed (see :func:`forward_flops` / :func:`total_flops`):

* **Detailed (default everywhere).** Per forward pass::

      linear = 2 * n_params * tokens                         # mul+add per linear weight per token
      attn   = 4 * n_layer * n_embd * seg_sq_sum             # QK^T + softmax*V, varlen sum(seg^2)

  Forward+backward multiplies this by 3 (no grad checkpointing) or **4** (grad
  checkpointing: fwd + recompute-fwd + bwd). ``gpt_pack`` runs with grad
  checkpointing, so MFU for it uses the 4x factor -- getting this wrong biases MFU
  by 33%.

* **Simple (nanoGPT-style).** ``(8 if grad_ckpt else 6) * n_params * tokens`` -- ignores the
  attention content term. Fine for the real *packed* workload (~264-token iGSM segments, where
  attention is ~4% of linear) but **wrong for long monolithic context** (at ctx=12288 attention
  content is ~1.8x the linear term), so the detailed formula is the default.

**Varlen / packing subtlety (load-bearing).** Under TRL ``bfd`` padding-free packing,
FlashAttention computes attention per packed segment via ``cu_seq_lens`` -- the attention FLOPs are
``4 * n_layer * n_embd * sum(seg_len**2)``, NOT ``4 * n_layer * n_embd * total_T**2``. Pass the
real ``seg_sq_sum`` (from the packed dataset's ``seq_lengths`` field) so the FLOP budget matches
what the kernel actually does.

Peak FLOPS use **dense** tensor throughput (structured 2:4 sparsity off, which is what unpruned
training uses). For RTX 3090 that is **71.2 TF** bf16 -- the widely-quoted 142 TF is the 2:4
*sparse* figure and would inflate the denominator 2x. Cross-check empirically with
:func:`calibrate_dense_tflops`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

# Dense bf16/fp16 tensor-core peak, by substring of torch.cuda.get_device_name() (lower-cased).
# Order matters: more specific keys first (e.g. 'h100 pcie' before 'h100'). These are NOT the
# sparse marketing numbers -- see module docstring.
PEAK_BF16_TFLOPS: list[tuple[str, float]] = [
    ('rtx 3090', 71.2),
    ('rtx 4090', 165.2),
    ('a100', 312.0),
    ('h100 pcie', 756.0),
    ('h100', 989.0),  # SXM
]


def count_trainable_params(model: nn.Module) -> int:
    """Trainable parameter count (excludes frozen/zeroed ``wpe``; tied ``wte``/``lm_head`` once).

    For the GPT2-12-12 config this is 123,653,376.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def segment_lengths_from_position_ids(position_ids: torch.Tensor) -> list[int]:
    """Reconstruct per-segment lengths from reset ``position_ids`` (shape ``(1, T)`` or ``(T,)``).

    A new segment starts wherever ``pos[i] == 0`` for ``i > 0`` -- the same reset FlashAttention
    uses to build ``cu_seq_lens``. ``assert sum(lengths) == T``.

    Warning: padding-free collators pad ``position_ids`` with 0 at the tail; strip real padding
    first or prefer :func:`flops_from_seq_lengths` (reads the dataset's ``seq_lengths`` directly).
    """
    pos = position_ids.view(-1)
    n = pos.numel()
    if n == 0:
        return []
    starts = pos[1:] == 0
    boundaries = [0, *(starts.nonzero(as_tuple=True)[0] + 1).tolist(), n]
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    assert sum(lengths) == n, f'segment lengths {sum(lengths)} != T {n}'
    return lengths


def _seg_sq_sum_from_lengths(seg_lengths: Sequence[int]) -> int:
    return sum(s * s for s in seg_lengths)


def forward_flops(
    *,
    n_layer: int,
    n_embd: int,
    n_params: int,
    tokens: int,
    seg_sq_sum: int,
    count_attn: bool = True,
) -> int:
    """FLOPs for one forward pass (detailed formula; see module docstring).

    Args:
        n_layer, n_embd: model geometry.
        n_params: trainable parameter count (drives the linear term; biases/LN negligible).
        tokens: total tokens in the batch (``T``).
        seg_sq_sum: ``sum(seg_len**2)`` across all packed segments in the batch. For a single
            monolithic sequence of length ``L`` this is ``L*L``.
        count_attn: set False to match the simple ``2*n_params*tokens`` linear-only convention.
    """
    linear = 2 * n_params * tokens
    attn = 4 * n_layer * n_embd * seg_sq_sum if count_attn else 0
    return linear + attn


def total_flops(
    *,
    n_layer: int,
    n_embd: int,
    n_params: int,
    tokens: int,
    seg_sq_sum: int,
    grad_ckpt: bool,
    count_attn: bool = True,
) -> int:
    """Forward+backward FLOPs (detailed). Multiplier is 4 with grad checkpointing, else 3."""
    mult = 4 if grad_ckpt else 3
    return mult * forward_flops(
        n_layer=n_layer,
        n_embd=n_embd,
        n_params=n_params,
        tokens=tokens,
        seg_sq_sum=seg_sq_sum,
        count_attn=count_attn,
    )


def flops_from_seq_lengths(
    seq_lengths: Sequence[Sequence[int]],
    *,
    n_layer: int,
    n_embd: int,
    n_params: int,
    grad_ckpt: bool,
    count_attn: bool = True,
) -> int:
    """Total FLOPs from a batch's per-window ``seq_lengths`` (the packed dataset column).

    Each element is one packed window's list of segment lengths. Sums tokens and ``seg_sq`` over
    the whole batch, then applies :func:`total_flops`. Preferred over position-ids reconstruction
    (no padding ambiguity).
    """
    tokens = sum(sum(segs) for segs in seq_lengths)
    seg_sq_sum = sum(_seg_sq_sum_from_lengths(segs) for segs in seq_lengths)
    return total_flops(
        n_layer=n_layer,
        n_embd=n_embd,
        n_params=n_params,
        tokens=tokens,
        seg_sq_sum=seg_sq_sum,
        grad_ckpt=grad_ckpt,
        count_attn=count_attn,
    )


def model_flops(
    model: nn.Module,
    seq_lengths: Sequence[Sequence[int]],
    *,
    grad_ckpt: bool,
    count_attn: bool = True,
) -> int:
    """Read geometry + trainable params from ``model``, then :func:`flops_from_seq_lengths`."""
    cfg = model.config
    return flops_from_seq_lengths(
        seq_lengths,
        n_layer=cfg.num_hidden_layers,
        n_embd=cfg.hidden_size,
        n_params=count_trainable_params(model),
        grad_ckpt=grad_ckpt,
        count_attn=count_attn,
    )


def peak_bf16_tflops(device_name: str | None = None) -> float:
    """Dense bf16 tensor peak (TFLOPS) for the given GPU name (or the current CUDA device).

    Raises ``ValueError`` (with a ``--peak-tflops`` hint) for unknown GPUs.
    """
    if device_name is None:
        if not torch.cuda.is_available():
            err = 'no CUDA device; pass --peak-tflops explicitly'
            raise ValueError(err)
        device_name = torch.cuda.get_device_name()
    name = device_name.lower()
    for key, tf in PEAK_BF16_TFLOPS:
        if key in name:
            return tf
    err = f'unknown GPU {device_name!r}; pass --peak-tflops to override'
    raise ValueError(err)


def mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs utilization fraction (achieved / peak)."""
    return achieved_tflops / peak_tflops


def calibrate_dense_tflops(
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,
    size: int = 8192,
    warmup: int = 5,
    iters: int = 20,
) -> float:
    """Empirically measure dense matmul TFLOPS on ``device`` -- the ground-truth MFU denominator.

    Runs square bf16 ``size x size`` GEMMs (cuBLAS, no sparsity) and reports
    ``2*size**3 / median_time``. Use this to sanity-check :func:`peak_bf16_tflops` for your card.
    """
    assert device.type == 'cuda', 'calibration needs CUDA'
    assert dtype in (torch.float16, torch.bfloat16, torch.float32), f'unsupported dtype {dtype}'
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    for _ in range(warmup):
        a = a @ b
    torch.cuda.synchronize()
    ts: list[float] = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        a = a @ b
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1000.0)
    ts.sort()
    median = ts[len(ts) // 2]
    flops = 2 * size**3
    return flops / median / 1e12
