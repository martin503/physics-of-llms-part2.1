"""Real-data end-to-end step decomposition: fetch / fwd / bwd_intermediate / bwd_final / opt.

THE DECIDER. Mirrors ``gpt_pack``'s exact data path (raw stream -> TRL ``bfd`` pack -> padding-free
collator -> DataLoader) and runs real optimizer steps with explicit per-phase timers, so the wall
clock of one ``gpt_pack`` step is broken into where it actually goes:

* **fetch** (CPU ``perf_counter`` over ``next(loader)``) -- if ``fetch_fraction`` is large the GPU
  is data-starved; if ~0 the GPU is the bottleneck and data waits for it.
* **fwd / bwd** (CUDA events) -- the model compute. With ``--grad-accum > 1``, non-final
  microbatches run under ``accelerator.no_sync(model)`` (no all-reduce), so
  ``bwd_final - bwd_intermediate`` isolates the DDP all-reduce cost. Run single-GPU and 2-GPU at
  equal per-rank tokens: if 2-GPU ``bwd_final`` is much larger, sync dominates.
* **opt** (CUDA events) -- AdamW (memory-bandwidth-bound, excluded from MFU).

MFU uses real ``tokens`` and real ``sum(seg**2)`` from the packed dataset's ``seq_lengths`` (not
``grad_accum * ctx``), with the 4x grad-checkpointing FLOP factor.

Examples::

    # CPU smoke (tiny model, sdpa, 2 steps):
    uv run python -m src.perf.e2e_bench --smoke --data-dir data/igsm_raw_small

    # Single-GPU real decomposition (matches gpt_pack defaults):
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.perf.e2e_bench --data-dir data/igsm_raw \
        --ctx 12288 --grad-accum 16 --grad-checkpointing

    # 2-GPU DDP (compare bwd_final vs the single-GPU run -> all-reduce cost):
    CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 \
        -m src.perf.e2e_bench --data-dir data/igsm_raw --ctx 12288 --grad-accum 16
"""

from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import torch
import typer
from accelerate import Accelerator
from torch.utils.data import DataLoader
from trl import pack_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from src.data.igsm import VOCAB_SIZE, load_igsm_dataset, load_igsm_stream
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope
from src.perf.mfu import count_trainable_params, peak_bf16_tflops, total_flops

app = typer.Typer(add_completion=False, help='Real-data step decomposition (the decider).')

EOS_TOKEN_ID = 50256


class _FlopsCollator:
    """Wrap TRL's padding-free collator; also carry per-batch seg_sq_sum + tokens for MFU.

    Top-level class so it is picklable for ``multiprocessing_context='spawn'`` DataLoader workers.
    """

    def __init__(self, base: DataCollatorForLanguageModeling) -> None:
        self.base = base

    def __call__(self, examples: list[dict]) -> dict:
        seg_sq = sum(s * s for ex in examples for s in ex['seq_lengths'])
        tokens = sum(sum(ex['seq_lengths']) for ex in examples)
        batch = self.base(examples)
        batch['_seg_sq_sum'] = seg_sq
        batch['_tokens'] = tokens
        return batch


def _time_region(fn, device: torch.device) -> float:
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
    data_dir: Annotated[
        Path, typer.Option('--data-dir', help='Raw iGSM problems (unpacked).')
    ] = Path('data/igsm_raw'),
    source: Annotated[
        str, typer.Option('--source', help='stream | inmem (stream matches gpt_pack).')
    ] = 'stream',
    ctx: Annotated[int, typer.Option('--ctx', help='Packed window length.')] = 12288,
    per_device_train_batch_size: Annotated[int, typer.Option('--per-device-batch-size')] = 1,
    gradient_accumulation_steps: Annotated[int, typer.Option('--grad-accum')] = 16,
    n_layer: Annotated[int, typer.Option('--n-layer')] = 12,
    n_head: Annotated[int, typer.Option('--n-head')] = 12,
    n_embd: Annotated[int, typer.Option('--n-embd')] = 768,
    n_inner: Annotated[int, typer.Option('--n-inner')] = 3072,
    vocab_size: Annotated[int, typer.Option('--vocab-size')] = VOCAB_SIZE,
    attn_implementation: Annotated[
        str, typer.Option('--attn-implementation')
    ] = 'flash_attention_2',
    bf16: Annotated[bool, typer.Option('--bf16/--no-bf16', help='bf16 on Ampere/3090.')] = True,
    grad_checkpointing: Annotated[
        bool, typer.Option('--grad-checkpointing/--no-grad-checkpointing')
    ] = True,
    num_workers: Annotated[int, typer.Option('--num-workers')] = 4,
    prefetch_factor: Annotated[int, typer.Option('--prefetch-factor')] = 2,
    warmup: Annotated[int, typer.Option('--warmup', help='Untimed optimizer steps.')] = 1,
    iters: Annotated[
        int, typer.Option('--iters', help='Timed optimizer steps; median reported.')
    ] = 5,
    peak_tflops: Annotated[float | None, typer.Option('--peak-tflops')] = None,
    smoke: Annotated[bool, typer.Option('--smoke')] = False,
) -> None:
    """Run the real-data step decomposition and print the per-phase + MFU table."""
    if smoke:
        n_layer, n_head, n_embd, n_inner = 2, 2, 64, 256
        ctx = 512
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 2
        attn_implementation = 'sdpa'
        bf16 = False
        grad_checkpointing = False
        num_workers = 0
        warmup, iters = 0, 2

    acc = Accelerator()
    dev = acc.device
    grad_accum = gradient_accumulation_steps

    # --- data path (mirrors gpt_pack) ---
    if source == 'inmem':
        packed = pack_dataset(load_igsm_dataset(data_dir), seq_length=ctx, strategy='bfd')
    else:
        packed = pack_dataset(load_igsm_stream(data_dir), seq_length=ctx, strategy='bfd')
    collator = _FlopsCollator(
        DataCollatorForLanguageModeling(pad_token_id=EOS_TOKEN_ID, padding_free=True)
    )
    loader = DataLoader(
        packed,
        batch_size=per_device_train_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context='spawn' if num_workers > 0 else None,
    )

    # --- model (mirrors gpt_pack build) ---
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
    model.to(dev)
    model.config.use_cache = False
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    n_params = count_trainable_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Prepare only model + opt, NOT the loader: accelerate's data-loader wrapper tries to concat
    # padding-free flat (1, T) batches across ranks (T differs per rank) and crashes. Each rank
    # iterates its own stream independently; data duplication is irrelevant for timing, and the DDP
    # all-reduce still fires on gradients via acc.backward.
    model, opt = acc.prepare(model, opt)

    peak = (
        peak_tflops
        if peak_tflops is not None
        else (peak_bf16_tflops() if dev.type == 'cuda' else 1.0)
    )
    if acc.is_main_process:
        typer.echo(
            f'[e2e_bench] world={acc.num_processes} dev={dev} source={source} ctx={ctx} '
            f'pdtbs={per_device_train_batch_size} grad_accum={grad_accum} '
            f'attn={attn_implementation} bf16={bf16} grad_ckpt={grad_checkpointing} '
            f'workers={num_workers}'
        )
        typer.echo(f'  params(trainable)={n_params:,}  peak={peak:.1f} TF')

    loader_iter = iter(loader)

    def run_step() -> dict:
        fetch_t: list[float] = []
        fwd_t: list[float] = []
        bwd_t: list[float] = []
        seg_sq = 0
        tokens = 0
        opt.zero_grad()
        for i in range(grad_accum):
            t0 = time.perf_counter()
            batch = next(loader_iter)
            fetch_t.append(time.perf_counter() - t0)
            seg_sq += batch.pop('_seg_sq_sum')
            tokens += batch.pop('_tokens')
            batch = {k: v.to(dev) for k, v in batch.items()}
            sync_ctx = nullcontext() if i == grad_accum - 1 else acc.no_sync(model)
            out: dict[str, torch.Tensor] = {}

            def fwd() -> None:
                out['loss'] = model(**batch).loss / grad_accum

            with sync_ctx:
                fwd_t.append(_time_region(fwd, dev))
                loss = out['loss']
                bwd_t.append(_time_region(lambda: acc.backward(loss), dev))

        opt_t = _time_region(lambda: (opt.step(), opt.zero_grad()), dev)
        flops = total_flops(
            n_layer=n_layer,
            n_embd=n_embd,
            n_params=n_params,
            tokens=tokens,
            seg_sq_sum=seg_sq,
            grad_ckpt=grad_checkpointing,
        )
        # bwd_t[-1] is the final microbatch (runs the DDP all-reduce); the rest run under no_sync.
        return {
            'fetch': sum(fetch_t),
            'fwd': sum(fwd_t),
            'bwd_inter_total': sum(bwd_t[:-1]),
            'bwd_final': bwd_t[-1] if bwd_t else 0.0,
            'bwd_total': sum(bwd_t),
            'opt': opt_t,
            'tokens': tokens,
            'seg_sq': seg_sq,
            'flops': flops,
        }

    # warmup
    for _ in range(warmup):
        run_step()
    if dev.type == 'cuda':
        torch.cuda.synchronize()

    steps = [run_step() for _ in range(iters)]

    def med(key: str) -> float:
        return statistics.median(s[key] for s in steps)

    step_total = med('fetch') + med('fwd') + med('bwd_total') + med('opt')
    compute_t = med('fwd') + med('bwd_total')
    achieved = med('flops') / compute_t / 1e12 if compute_t else 0.0
    mfu_pct = achieved / peak * 100 if peak else float('nan')
    fetch_frac = med('fetch') / step_total if step_total else 0.0
    n_inter = max(1, grad_accum - 1)
    per_mb_inter = med('bwd_inter_total') / n_inter
    reduce_delta = max(0.0, med('bwd_final') - per_mb_inter)

    if acc.is_main_process:
        typer.echo(
            f'\n  per-step (median over {iters} steps): step_total={step_total * 1000:.0f}ms'
        )
        typer.echo(
            f'    fetch={med("fetch") * 1000:.1f}ms ({fetch_frac * 100:.1f}% of step)  '
            f'fwd={med("fwd") * 1000:.0f}ms  bwd_total={med("bwd_total") * 1000:.0f}ms  '
            f'opt={med("opt") * 1000:.1f}ms'
        )
        if grad_accum > 1:
            typer.echo(
                f'    bwd per-microbatch: intermediate(no_sync)={per_mb_inter * 1000:.1f}ms  '
                f'final(w/ all-reduce)={med("bwd_final") * 1000:.1f}ms  '
                f'reduce_delta={reduce_delta * 1000:.1f}ms'
            )
        typer.echo(
            f'    tokens/step={med("tokens")}  flops/step={med("flops"):.3e}  '
            f'achieved={achieved:.1f} TF  MFU={mfu_pct:.1f}%'
        )
        typer.echo(
            '  decision: fetch% large -> data-starved; reduce_delta large (and 2-GPU >> 1-GPU '
            'bwd_final) -> all-reduce; fwd+bwd ~ step_total with moderate MFU -> compute-capped.'
        )


if __name__ == '__main__':
    app()
