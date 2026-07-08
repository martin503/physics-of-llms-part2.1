"""torch.profiler for gpt_pack: real SFTTrainer path + a clean standalone loop.

Two modes:

* ``--mode real`` replicates the relevant slice of ``src/train/gpt_pack.py::train`` (same builders,
  same streaming + TRL packing, same ``SFTConfig``/``SFTTrainer``) and runs a few steps under
  ``torch.profiler``. Use this to see the FULL real path -- including any SFTTrainer framework
  overhead a manual loop misses -- as a sorted ``key_averages`` table and a Chrome/Perfetto trace.
  Eval/save/logging are disabled to keep the trace clean; runs single-GPU (the all-reduce cost is
  already measured by ``e2e_bench`` and is ~25 ms, negligible).
* ``--mode standalone`` runs a synthetic fwd+backward+opt loop (no Trainer, no data) under the
  profiler -- a clean kernel-level trace for comparison.

Open the ``.json`` in chrome://tracing or ui.perfetto.dev. Sort by ``self_cuda_time_total``:
``flash_attn_varlen_func``/``gemm`` dominant -> compute-bound; ``nccl_*``/``wait`` -> sync;
host-to-device ``memcpy`` or long idle gaps between GPU kernels -> data/CPU-bound.

Examples::

    # CPU smoke (standalone, tiny model, sdpa):
    uv run python -m src.perf.profile_gpt_pack --mode standalone --smoke --max-steps 3

    # Real SFTTrainer path under the profiler (single GPU):
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.perf.profile_gpt_pack --mode real \
        --data-dir data/igsm_raw --ctx 768 --per-device-batch-size 16 --grad-accum 16 \
        --max-steps 8 --trace-out profiles/gpt_pack_real.json

    # Standalone clean trace (packed varlen):
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.perf.profile_gpt_pack --mode standalone \
        --ctx 768 --packing-mode packed --max-steps 8
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import torch
import typer
from transformers import GPT2TokenizerFast

from src.data.igsm import VOCAB_SIZE, load_igsm_stream
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope

app = typer.Typer(add_completion=False, help='torch.profiler for gpt_pack (real + standalone).')


def _build_model(
    *,
    n_layer,
    n_head,
    n_embd,
    n_inner,
    vocab_size,
    ctx,
    attn_implementation,
    bf16,
    grad_checkpointing,
):
    config = build_gpt2_config(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_inner=n_inner,
        vocab_size=vocab_size,
        n_positions=max(2048, ctx),
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    if bf16 and attn_implementation == 'flash_attention_2':
        config.dtype = torch.bfloat16
    model = build_gpt2_rope(config, attn_implementation=attn_implementation)
    if bf16 and attn_implementation == 'flash_attention_2':
        model.to(torch.bfloat16)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.config.use_cache = False
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    return model


@app.command()
def main(
    mode: Annotated[str, typer.Option('--mode', help='real | standalone.')] = 'standalone',
    data_dir: Annotated[
        Path, typer.Option('--data-dir', help='Raw iGSM problems (real mode).')
    ] = Path('data/igsm_raw'),
    ctx: Annotated[int, typer.Option('--ctx')] = 768,
    per_device_train_batch_size: Annotated[int, typer.Option('--per-device-batch-size')] = 16,
    gradient_accumulation_steps: Annotated[int, typer.Option('--grad-accum')] = 16,
    packing_mode: Annotated[
        str, typer.Option('--packing-mode', help='standalone only: monolithic | packed.')
    ] = 'packed',
    seg_len: Annotated[
        int, typer.Option('--seg-len', help='standalone packed segment length.')
    ] = 264,
    n_layer: Annotated[int, typer.Option('--n-layer')] = 12,
    n_head: Annotated[int, typer.Option('--n-head')] = 12,
    n_embd: Annotated[int, typer.Option('--n-embd')] = 768,
    n_inner: Annotated[int, typer.Option('--n-inner')] = 3072,
    vocab_size: Annotated[int, typer.Option('--vocab-size')] = VOCAB_SIZE,
    attn_implementation: Annotated[
        str, typer.Option('--attn-implementation')
    ] = 'flash_attention_2',
    bf16: Annotated[bool, typer.Option('--bf16/--no-bf16')] = True,
    grad_checkpointing: Annotated[
        bool, typer.Option('--grad-checkpointing/--no-grad-checkpointing')
    ] = True,
    num_workers: Annotated[
        int, typer.Option('--num-workers', help='real mode DataLoader workers.')
    ] = 4,
    max_steps: Annotated[int, typer.Option('--max-steps', help='Steps to profile.')] = 8,
    trace_out: Annotated[
        Path, typer.Option('--trace-out', help='Chrome trace output path.')
    ] = Path('profiles/gpt_pack_trace.json'),
    row_limit: Annotated[int, typer.Option('--row-limit', help='key_averages table rows.')] = 30,
    smoke: Annotated[bool, typer.Option('--smoke')] = False,
) -> None:
    """Profile gpt_pack (real SFTTrainer or standalone) and dump a table + Chrome trace."""
    if smoke:
        n_layer, n_head, n_embd, n_inner = 2, 2, 64, 256
        ctx = 256
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 1
        attn_implementation = 'sdpa'
        bf16 = False
        grad_checkpointing = False
        max_steps = 3
        num_workers = 0

    trace_out = Path(trace_out)
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    activities = []
    if torch.cuda.is_available():
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]

    if mode == 'real':
        # Mirrors src/train/gpt_pack.py::train -- keep in sync. Setup is OUTSIDE the profiler so
        # the trace covers only real training steps.
        from trl import SFTConfig, SFTTrainer

        config = build_gpt2_config(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_inner=n_inner,
            vocab_size=vocab_size,
            n_positions=max(2048, ctx),
        )
        if bf16 and attn_implementation == 'flash_attention_2':
            config.dtype = torch.bfloat16
        model = build_gpt2_rope(config, attn_implementation=attn_implementation)
        if bf16 and attn_implementation == 'flash_attention_2':
            model.to(torch.bfloat16)
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        train_dataset = load_igsm_stream(data_dir)
        args = SFTConfig(
            output_dir=str(trace_out.parent / 'tmp_out'),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=max_steps,
            bf16=bf16,
            gradient_checkpointing=grad_checkpointing,
            logging_steps=max_steps,
            logging_first_step=False,
            save_strategy='no',
            eval_strategy='no',
            report_to=[],
            dataloader_num_workers=num_workers,
            dataloader_persistent_workers=num_workers > 0,
            dataloader_prefetch_factor=2 if num_workers > 0 else None,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            packing=True,
            packing_strategy='bfd',
            max_length=ctx,
            eval_packing=True,
            use_cpu=False,
        )
        trainer = SFTTrainer(
            model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer
        )
        typer.echo(f'[profile_gpt_pack] mode=real: profiling {max_steps} real SFTTrainer steps...')
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            trainer.train()
        wall = time.perf_counter() - t0
        typer.echo(
            f'[profile_gpt_pack] mode=real: {wall:.1f}s for {max_steps} steps -> '
            f'{wall / max_steps:.2f}s/step (incl. profiler overhead & step-0 warmup)'
        )
    elif mode == 'standalone':
        model = _build_model(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_inner=n_inner,
            vocab_size=vocab_size,
            ctx=ctx,
            attn_implementation=attn_implementation,
            bf16=bf16,
            grad_checkpointing=grad_checkpointing,
        )
        dev = next(model.parameters()).device
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t = per_device_train_batch_size * ctx
        input_ids = torch.randint(0, vocab_size, (1, t), device=dev, dtype=torch.long)
        labels = input_ids.clone()
        if packing_mode == 'monolithic':
            position_ids = torch.arange(t, device=dev).unsqueeze(0)
        else:
            seg = torch.arange(seg_len, device=dev)
            position_ids = seg.repeat(t // seg_len + 1)[:t].unsqueeze(0)

        def step() -> None:
            out = model(input_ids=input_ids, position_ids=position_ids, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()

        typer.echo(f'[profile_gpt_pack] mode=standalone: profiling {max_steps} synthetic steps...')
        with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
            for _ in range(max_steps):
                step()
    else:  # pragma: no cover
        raise ValueError(f'invalid mode {mode!r}')

    typer.echo(f'\n=== key_averages (sort by self_cuda_time_total, top {row_limit}) ===')
    typer.echo(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=row_limit))
    prof.export_chrome_trace(str(trace_out))
    typer.echo(
        f'\nChrome trace written to {trace_out} (open in chrome://tracing or ui.perfetto.dev).'
    )


if __name__ == '__main__':
    app()
