"""Synthetic GPU compute ceiling: fwd+bwd(+opt) MFU, sweep ctx x windows, 1-GPU vs DDP.

Isolates **GPU compute** from everything else (no data, no IO, no Trainer). Feed random
``input_ids`` of shape ``(1, T)`` with ``T = windows * ctx`` and reset ``position_ids`` (mirroring
TRL's padding-free collator output) through the real GPT-2 + RoPE model, time fwd/bwd/opt with CUDA
events, and report achieved TFLOPS + MFU%.

This is the **upper bound** for the model on this GPU. If real ``gpt_pack`` training is much slower
than this, the bottleneck is data or sync, not compute. The ``--packing-mode`` axis is mandatory:
monolithic (full ``T`` cross-attention, attention FLOPs ~1.8x the linear term at ctx=12288) vs
packed (~264-token varlen segments, attention ~4% of linear) have very different FLOP budgets;
a packed/monolithic mixup would invert the MFU conclusion.

Run single-GPU and 2-GPU at equal per-rank tokens; if 2-GPU per-rank step time is much larger, the
delta is DDP all-reduce/sync cost.

Examples::

    # CPU smoke (tiny model, sdpa, fp32):
    uv run python -m src.perf.gpu_ceiling --smoke

    # Single-GPU ceiling, packed varlen, grad checkpointing (matches gpt_pack):
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.perf.gpu_ceiling --ctx 12288 \
        --packing-mode packed --seg-len 264 --grad-checkpointing --calibrate

    # 2-GPU DDP ceiling (compare per-rank step time to the single-GPU run):
    CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 \
        -m src.perf.gpu_ceiling --ctx 12288 --packing-mode packed --seg-len 264
"""

from __future__ import annotations

import statistics
import time
from typing import Annotated

import torch
import typer
from accelerate import Accelerator

from src.data.igsm import VOCAB_SIZE
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope
from src.perf.mfu import (
    calibrate_dense_tflops,
    count_trainable_params,
    forward_flops,
    peak_bf16_tflops,
)

app = typer.Typer(add_completion=False, help='Synthetic GPU compute ceiling + MFU.')


def _synthetic_batch(
    ctx: int, windows: int, packing_mode: str, seg_len: int, vocab: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build ``(input_ids, position_ids, labels)`` each ``(1, T)`` with ``T = windows*ctx``.

    Returns the batch plus its ``seg_sq_sum`` (for FLOP accounting). ``monolithic`` -> one segment
    over the whole flat ``T``; ``packed`` -> segments of length ``seg_len`` with resets.
    """
    t = windows * ctx
    input_ids = torch.randint(0, vocab, (1, t), device=device, dtype=torch.long)
    labels = input_ids.clone()
    if packing_mode == 'monolithic':
        position_ids = torch.arange(t, device=device).unsqueeze(0)
        seg_sq_sum = t * t
    elif packing_mode == 'packed':
        seg = torch.arange(seg_len, device=device)
        position_ids = seg.repeat(t // seg_len + 1)[:t].unsqueeze(0)
        full, rem = divmod(t, seg_len)
        seg_sq_sum = full * seg_len * seg_len + rem * rem
    else:  # pragma: no cover - typer guards this
        raise ValueError(f'invalid packing_mode {packing_mode!r}')
    return input_ids, position_ids, labels, seg_sq_sum


def _time_region(fn, device: torch.device) -> float:
    """Time a thunk in seconds (CUDA events on GPU, perf_counter on CPU)."""
    if device.type == 'cuda':
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / 1000.0
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@app.command()
def main(
    ctx: Annotated[int, typer.Option('--ctx', help='Tokens per packed window.')] = 12288,
    windows: Annotated[
        int, typer.Option('--windows', help='Packed windows per step (= per-device batch).')
    ] = 1,
    packing_mode: Annotated[
        str, typer.Option('--packing-mode', help='monolithic (full T attn) | packed (varlen seg).')
    ] = 'packed',
    seg_len: Annotated[
        int, typer.Option('--seg-len', help='Segment length for packed mode (iGSM ~264).')
    ] = 264,
    n_layer: Annotated[int, typer.Option('--n-layer')] = 12,
    n_head: Annotated[int, typer.Option('--n-head')] = 12,
    n_embd: Annotated[int, typer.Option('--n-embd')] = 768,
    n_inner: Annotated[int, typer.Option('--n-inner')] = 3072,
    vocab_size: Annotated[int, typer.Option('--vocab-size')] = VOCAB_SIZE,
    attn_implementation: Annotated[
        str, typer.Option('--attn-implementation', help='flash_attention_2 | sdpa | eager.')
    ] = 'flash_attention_2',
    bf16: Annotated[bool, typer.Option('--bf16/--no-bf16', help='bf16 on Ampere/3090.')] = True,
    grad_checkpointing: Annotated[
        bool,
        typer.Option(
            '--grad-checkpointing/--no-grad-checkpointing', help='Match gpt_pack (4x FLOP factor).'
        ),
    ] = True,
    warmup: Annotated[
        int, typer.Option('--warmup', help='Untimed warmup iters (absorbs cudnn autotune).')
    ] = 5,
    iters: Annotated[int, typer.Option('--iters', help='Timed iters; median is reported.')] = 10,
    with_optimizer: Annotated[bool, typer.Option('--with-optimizer/--no-optimizer')] = True,
    calibrate: Annotated[
        bool,
        typer.Option('--calibrate', help='Measure dense bf16 GEMM TFLOPS as a peak cross-check.'),
    ] = False,
    peak_tflops: Annotated[
        float | None, typer.Option('--peak-tflops', help='Override peak bf16 TFLOPS.')
    ] = None,
    smoke: Annotated[bool, typer.Option('--smoke', help='CPU tiny-model run.')] = False,
) -> None:
    """Run the synthetic compute ceiling and print tokens/s, TFLOPS, MFU%."""
    if smoke:
        n_layer, n_head, n_embd, n_inner = 2, 2, 64, 256
        ctx, windows, seg_len = 128, 1, 32
        attn_implementation = 'sdpa'
        bf16 = False
        grad_checkpointing = False
        warmup, iters = 1, 3
        with_optimizer = True

    acc = Accelerator()
    dev = acc.device
    n_positions = max(2048, windows * ctx)
    config = build_gpt2_config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_inner=n_inner,
        vocab_size=vocab_size,
        n_positions=n_positions,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    if bf16 and attn_implementation == 'flash_attention_2':
        config.dtype = torch.bfloat16
    model = build_gpt2_rope(config, attn_implementation=attn_implementation)
    if bf16 and attn_implementation == 'flash_attention_2':
        model.to(torch.bfloat16)
    model.to(dev)
    model.config.use_cache = False
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    n_params = count_trainable_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3) if with_optimizer else None
    to_prepare = [model] + ([opt] if opt is not None else [])
    prepared = acc.prepare(*to_prepare)
    model = prepared[0]
    if opt is not None:
        opt = prepared[1]

    input_ids, position_ids, labels, seg_sq_sum = _synthetic_batch(
        ctx, windows, packing_mode, seg_len, vocab_size, dev
    )
    tokens = input_ids.numel()
    flops = forward_flops(  # forward only; total_flops applies the 3x/4x bwd multiplier
        n_layer=n_layer, n_embd=n_embd, n_params=n_params, tokens=tokens, seg_sq_sum=seg_sq_sum
    ) * (4 if grad_checkpointing else 3)

    def step() -> tuple[float, float, float]:
        out: dict[str, object] = {}

        def fwd() -> None:
            out['loss'] = model(input_ids=input_ids, position_ids=position_ids, labels=labels).loss

        fwd_t = _time_region(fwd, dev)
        loss = out['loss']
        assert isinstance(loss, torch.Tensor)
        bwd_t = _time_region(lambda: acc.backward(loss), dev)
        opt_t = 0.0
        if opt is not None:
            opt_t = _time_region(lambda: (opt.step(), opt.zero_grad()), dev)
        return fwd_t, bwd_t, opt_t

    # warmup (absorbs cudnn autotune, lazy init, first-step materialization)
    for _ in range(warmup):
        step()
    if dev.type == 'cuda':
        torch.cuda.synchronize()

    fwd_ts, bwd_ts, opt_ts = [], [], []
    for _ in range(iters):
        f, b, o = step()
        fwd_ts.append(f)
        bwd_ts.append(b)
        opt_ts.append(o)

    def med(xs: list[float]) -> float:
        return statistics.median(xs)

    compute_t = med([f + b for f, b in zip(fwd_ts, bwd_ts, strict=True)])
    step_t = med([f + b + o for f, b, o in zip(fwd_ts, bwd_ts, opt_ts, strict=True)])
    achieved = flops / compute_t / 1e12
    peak = (
        peak_tflops
        if peak_tflops is not None
        else (peak_bf16_tflops() if dev.type == 'cuda' else 1.0)
    )
    mfu_pct = achieved / peak * 100 if peak else float('nan')

    if acc.is_main_process:
        world = acc.num_processes
        typer.echo(
            f'[gpu_ceiling] world={world} dev={dev} attn={attn_implementation} '
            f'bf16={bf16} grad_ckpt={grad_checkpointing}'
        )
        typer.echo(
            f'  packing={packing_mode} ctx={ctx} windows={windows} T={tokens} '
            f'seg_len={seg_len} -> seg_sq_sum={seg_sq_sum}'
        )
        typer.echo(f'  params(trainable)={n_params:,}')
        typer.echo(
            f'  fwd={med(fwd_ts) * 1000:.1f}ms  bwd={med(bwd_ts) * 1000:.1f}ms  '
            f'opt={med(opt_ts) * 1000:.1f}ms  compute(fwd+bwd)={compute_t * 1000:.1f}ms  '
            f'step={step_t * 1000:.1f}ms'
        )
        typer.echo(
            f'  tokens/step={tokens}  tokens/s={tokens / compute_t:,.0f}  '
            f'flops/step={flops:.3e}  achieved={achieved:.1f} TF'
        )
        if dev.type == 'cuda':
            typer.echo(f'  peak={peak:.1f} TF  ->  MFU={mfu_pct:.1f}%')
            if calibrate:
                measured = calibrate_dense_tflops(dev)
                typer.echo(
                    f'  [calibrate] dense bf16 GEMM = {measured:.1f} TF '
                    f'(table peak {peak:.1f} TF; ratio {measured / peak:.2f})'
                )
        typer.echo(
            '  interpret: if real gpt_pack step time >> this compute time, the cost is data or '
            'sync. Compare 1-GPU vs N-GPU bwd time -> all-reduce cost.'
        )


if __name__ == '__main__':
    app()
