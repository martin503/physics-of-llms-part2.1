"""Train GPT-2 + RoPE on iGSM-med with on-the-fly generation + TRL packing.

Unlike :mod:`src.train.gpt` (offline, naively packed 768-token windows where adjacent
problems cross-attend), this script:

* streams **raw** iGSM problems from a pre-generated store (``--data-dir``; see
  ``python -m src.data.igsm generate --raw``) and TRL-packs them online -- GPU-bound. Without
  ``--data-dir`` it falls back to generating on the fly via :mod:`src.data.igsm_stream` (one
  distinct seed stream per rank/worker; correct, but ~30 ms/problem bottlenecks long runs);
* **packs with TRL** (best-fit-decreasing) so each packed window holds multiple
  problems, with ``position_ids`` resetting at every problem boundary and
  FlashAttention deriving ``cu_seq_lens`` from those resets -- so **packed problems do
  not attend across boundaries**;
* **shards correctly** across DDP ranks and DataLoader workers (one distinct iGSM
  seed stream per (rank, worker) -- no duplicate data, full parallelism).

Note: under TRL's padding-free collator the first token of every packed problem
(``[222]``) is masked from the loss (``-100``) -- ~1 token per ~264, negligible and
arguably more correct than ``src.train.gpt``'s no-masking naive packing.

Examples::

    # CPU smoke test (tiny model, sdpa, few steps; exercises the full TRL packing
    # pipeline but not FlashAttention, which needs CUDA). Pre-gen a tiny raw set first:
    #   uv run python -m src.data.igsm generate --raw --num-problems 2000 \
    #       --batch-size 500 --out data/igsm_raw_small
    uv run python -m src.train.gpt_pack --data-dir data/igsm_raw_small --smoke

    # Real run on 1 GPU (bfd packing REQUIRES flash_attention_2):
    CUDA_VISIBLE_DEVICES=0 uv run python -m src.train.gpt_pack --max-steps 1000

    # Multi-GPU via torchrun (Trainer auto-DDPs; world_size seeds are disjoint):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 -m src.train.gpt_pack \\
        --max-steps 5000
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from transformers import GPT2TokenizerFast, TrainerCallback

from src.data.igsm import VOCAB_SIZE, generate_problems, load_igsm_stream
from src.data.igsm_stream import build_igsm_stream, seed_offsets_for_shards
from src.model.gpt2_rope import (
    GPT2LMHeadModelWithRoPE,
    build_gpt2_config,
    build_gpt2_rope,
    recompute_rope_inv_freq,
)

DEFAULT_CONTEXT_LENGTH = 768


def _world_size() -> int:
    """Return the DDP world size from the environment (set by torchrun/accelerate)."""
    return int(os.environ.get('WORLD_SIZE', '1'))


class LogSampleCallback(TrainerCallback):
    """Periodically log a decoded iGSM sample to W&B for sanity inspection.

    Unlike :class:`src.train.gpt.LogSampleCallback`, this does **not** re-iterate the
    training DataLoader (unsafe with an infinite streaming IterableDataset + persistent
    workers). Instead it decodes one freshly generated problem at construction time.
    """

    def __init__(self, tokenizer: GPT2TokenizerFast, seed: int, every: int = 500) -> None:
        self.tokenizer = tokenizer
        self.every = every
        self._text: str | None = None
        try:
            streams = generate_problems(num_problems=2, split='train', workers=1, seed=seed)
            flat = [t for s in streams for t in s]
            self._text = tokenizer.decode(flat, skip_special_tokens=False)
        except Exception:  # noqa: BLE001 -- degrade silently if generation fails
            self._text = None

    def on_step_end(self, args, state, control, **kwargs) -> None:  # noqa: ARG002
        if not state.is_world_process_zero or self._text is None:
            return
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        import wandb

        wandb.log(
            {'train/sample_text': wandb.Html(f'<pre>{self._text[:2000]}</pre>')},
            step=state.global_step,
        )


app = typer.Typer(add_completion=False, help='Train GPT-2 + RoPE on iGSM-med (TRL packing).')


@app.command()
def train(
    output_dir: Annotated[
        Path, typer.Option('--output-dir', help='Where to write checkpoints.')
    ] = Path('models/gpt2-rope-igsm-pack'),
    init_from: Annotated[
        Path | None,
        typer.Option(
            '--init-from',
            help='Initialize model weights from a saved checkpoint dir (model weights only; '
            'fresh optimizer/scheduler, step count from 0). Use to continue training from a '
            'trainer.save_model() output, e.g. models/100k_model/gpt-rope-igsm-fixed.',
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            '--data-dir',
            help='Pre-generated RAW iGSM problems (recommended; packed online). Omit for '
            'on-the-fly generation, which is ~30 ms/problem and bottlenecks long runs.',
        ),
    ] = None,
    context_length: Annotated[
        int, typer.Option('--context-length', help='Packed window length (TRL max_length).')
    ] = DEFAULT_CONTEXT_LENGTH,
    per_device_train_batch_size: Annotated[
        int, typer.Option('--per-device-train-batch-size')
    ] = 16,
    gradient_accumulation_steps: Annotated[
        int, typer.Option('--gradient-accumulation-steps')
    ] = 16,
    learning_rate: Annotated[float, typer.Option('--learning-rate')] = 2e-3,
    lr_scheduler_type: Annotated[
        str, typer.Option('--lr-scheduler-type', help='transformers LR scheduler name.')
    ] = 'cosine_with_min_lr',
    lr_scheduler_kwargs: Annotated[
        str | None,
        typer.Option(
            '--lr-scheduler-kwargs',
            help='Scheduler kwargs as a JSON object, e.g. \'{"min_lr_rate": 0.01}\'. '
            'Defaults to {"min_lr_rate": 0.01} for cosine_with_min_lr, else {}.',
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option('--max-steps', help='Paper uses 100k; 5k is a working default.')
    ] = 5_000,
    warmup_steps: Annotated[int, typer.Option('--warmup-steps')] = 1_000,
    weight_decay: Annotated[float, typer.Option('--weight-decay')] = 0.05,
    adam_beta1: Annotated[float, typer.Option('--adam-beta1')] = 0.9,
    adam_beta2: Annotated[float, typer.Option('--adam-beta2')] = 0.98,
    adam_epsilon: Annotated[float, typer.Option('--adam-epsilon')] = 1e-8,
    bf16: Annotated[
        bool, typer.Option('--bf16/--no-bf16', help='bf16 on Ampere/3090 (paper used fp16).')
    ] = True,
    gradient_checkpointing: Annotated[
        bool,
        typer.Option(
            '--gradient-checkpointing',
            help='Recompute forward in backward (~+33%% compute). Only worth it at long context '
            "where activations don't fit; at the default ctx=768 a 124M model fits easily, so "
            'this is off by default -- enabling it costs ~1.5 s/step for nothing here.',
        ),
    ] = False,
    logging_steps: Annotated[int, typer.Option('--logging-steps')] = 20,
    save_steps: Annotated[int, typer.Option('--save-steps')] = 1_000,
    save_total_limit: Annotated[
        int | None,
        typer.Option('--save-total-limit', help='Keep at most N step checkpoints; None = all.'),
    ] = 3,
    per_device_eval_batch_size: Annotated[int, typer.Option('--per-device-eval-batch-size')] = 1,
    eval_steps: Annotated[
        int, typer.Option('--eval-steps', help='Validate every N optimizer steps.')
    ] = 1_000,
    max_eval_samples: Annotated[
        int, typer.Option('--max-eval-samples', help='Cap eval problems (stream is infinite).')
    ] = 1024,
    no_eval: Annotated[
        bool, typer.Option('--no-eval', help='Disable validation entirely.')
    ] = False,
    attn_implementation: Annotated[
        str,
        typer.Option(
            '--attn-implementation',
            help='bfd packing isolates packed samples only with flash_attention_2.',
        ),
    ] = 'flash_attention_2',
    loss_type: Annotated[
        str,
        typer.Option(
            '--loss-type',
            help="TRL loss: 'nll' (one big lm_head + CE) or 'chunked_nll' (CE in 256-token "
            "chunks). TRL defaults to 'chunked_nll' to save activation memory at huge scale, but "
            "here its 256-row matmuls are launch/tiling-bound and add ~3 s/step vs 'nll' -- so we "
            "default to 'nll' (identical math, VRAM is plentiful for a 124M model at ctx=768).",
        ),
    ] = 'nll',
    torch_compile: Annotated[
        bool,
        typer.Option(
            '--torch-compile/--no-torch-compile',
            help='OFF by default: torch.compile is INCOMPATIBLE with TRL padding-free varlen '
            'packing -- it graph-breaks on flash-varlen `max_seqlen.item()`, recompiles per layer '
            'on `module.layer_idx`, and recompiles on every distinct flat sequence length (the '
            'bfd padding-free collator emits variable-length (1, T) batches). Result: constant '
            'recompilation, ~8x SLOWER (52 s/step vs 6.5 s/step). `src.train.gpt` can compile '
            'because it uses fixed (B, 768) sdpa shapes; this path cannot.',
        ),
    ] = False,
    use_liger_kernel: Annotated[
        bool,
        typer.Option(
            '--use-liger-kernel/--no-liger-kernel',
            help='Fuse lm_head + cross-entropy (no full logits materialized). A small ~8%% '
            'speedup when on, but our custom GPT2-RoPE patcher does not expose token_accuracy, '
            'so `mean_token_accuracy` is not logged (cosmetic). Off by default for clean metrics.',
        ),
    ] = False,
    dataloader_num_workers: Annotated[
        int, typer.Option('--dataloader-num-workers', help='Parallel on-the-fly generators.')
    ] = 4,
    dataloader_prefetch_factor: Annotated[
        int, typer.Option('--dataloader-prefetch-factor', help='Batches buffered per worker.')
    ] = 2,
    report_to: Annotated[list[str] | None, typer.Option('--report-to')] = None,
    seed: Annotated[int, typer.Option('--seed')] = 0,
    smoke: Annotated[
        bool, typer.Option('--smoke', help='CPU tiny-model smoke run (overrides most args).')
    ] = False,
) -> None:
    """Train GPT-2 + RoPE on iGSM-med, TRL-packed online (bfd + flash-varlen).

    With ``--data-dir`` (recommended), streams pre-generated **raw** problems from disk and
    TRL-packs them during training -- GPU-bound. Without it, generates problems on the fly
    (one distinct seed stream per rank/worker), which is correct but ~30 ms/problem and
    bottlenecks long runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_to = report_to or []

    # Resolve LR scheduler kwargs: JSON if given, else the documented cosine default.
    if lr_scheduler_kwargs is not None:
        try:
            sched_kwargs: dict[str, Any] = json.loads(lr_scheduler_kwargs)
        except json.JSONDecodeError as e:
            raise typer.BadParameter(f'--lr-scheduler-kwargs must be a JSON object: {e}') from e
    elif lr_scheduler_type == 'cosine_with_min_lr':
        sched_kwargs = {'min_lr_rate': 0.01}
    else:
        sched_kwargs = {}

    if smoke:
        # Tiny model + few steps on CPU, sdpa (FlashAttention needs CUDA), no wandb --
        # validates the full TRL packing pipeline offline.
        config = build_gpt2_config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_inner=256,
            vocab_size=VOCAB_SIZE,
            n_positions=max(2048, context_length),
        )
        context_length = 1024  # > max iGSM-med problem length (~890) so bfd never truncates
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 1
        max_steps = 4
        bf16 = False
        gradient_checkpointing = False
        report_to = []
        logging_steps = 1
        save_steps = max_steps
        dataloader_num_workers = 0
        max_eval_samples = 16
        per_device_eval_batch_size = 2
        eval_steps = max_steps  # = 4 -> eval runs once during training, plus the final one
        attn_implementation = 'sdpa'
    else:
        config = build_gpt2_config(vocab_size=VOCAB_SIZE, n_positions=max(2048, context_length))

    if init_from is not None:
        typer.echo(f'Initializing model weights from checkpoint {init_from} (fresh optimizer)...')
        model = GPT2LMHeadModelWithRoPE.from_pretrained(
            init_from, attn_implementation=attn_implementation
        )
        recompute_rope_inv_freq(model)
        # requires_grad=False does not survive save_pretrained/from_pretrained; re-freeze the
        # dead absolute positional embedding so continued training matches from-scratch.
        model.transformer.wpe.requires_grad_(False)
        tag = 'from-checkpoint'
    else:
        if bf16 and attn_implementation == 'flash_attention_2':
            config.dtype = torch.bfloat16
        model = build_gpt2_rope(config, attn_implementation=attn_implementation)
        tag = 'smoke' if smoke else '12-12'
    if bf16 and attn_implementation == 'flash_attention_2':
        model.to(torch.bfloat16)
    dims = (
        f'{model.config.num_hidden_layers}L {model.config.n_embd}d '
        f'{model.config.num_attention_heads}h'
    )
    typer.echo(f'Building GPT-2 + RoPE ({tag}): {dims}, attn={attn_implementation}')

    # GPT-2 tokenizer: only used for pad/eos resolution + sample logging (data is
    # pre-tokenized by iGSM, so SFTTrainer skips tokenization). GPT-2 has eos (50256);
    # pad falls back to eos -- harmless under padding-free (no real padding).
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    world_size = _world_size()
    workers = dataloader_num_workers
    if data_dir is not None:
        # Fast path: stream pre-generated RAW problems and TRL-pack online. Generation
        # (~30 ms/problem) happened once offline; the GPU is the bottleneck again.
        typer.echo(f'Streaming raw problems from {data_dir} (TRL packs online)')
        train_dataset = load_igsm_stream(Path(data_dir))
    else:
        # Slow path: generate on the fly (one distinct iGSM seed stream per rank/worker).
        typer.echo(
            'WARNING: on-the-fly generation (~30 ms/problem) bottlenecks long runs -- '
            'pre-generate raw (`python -m src.data.igsm generate --raw`) and pass --data-dir.'
        )
        typer.echo(
            f'  world_size={world_size}, dataloader_num_workers={workers} '
            f'-> {seed_offsets_for_shards(world_size, workers, seed)} train seed stream(s)'
        )
        train_dataset = build_igsm_stream(
            seed_offsets_for_shards(world_size, workers, base_seed=seed), split='train'
        )

    eval_dataset = None
    if no_eval:
        typer.echo('  (validation disabled via --no-eval)')
    else:
        # Deterministic, disjoint-from-train eval streams (split='test', bins 16-22),
        # capped to a finite size -- the stream is infinite, so a cap is mandatory.
        eval_dataset = build_igsm_stream(
            seed_offsets_for_shards(world_size, workers, base_seed=1_000_000), split='test'
        ).take(max_eval_samples)
        typer.echo(f'  eval: on-the-fly test split, {max_eval_samples} problems')

    callbacks: list[TrainerCallback] = []
    if 'wandb' in report_to and not smoke:
        callbacks.append(LogSampleCallback(tokenizer, seed=seed, every=max(500, logging_steps)))

    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=sched_kwargs,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=eval_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        report_to=report_to,
        torch_compile=torch_compile,
        use_liger_kernel=use_liger_kernel,
        # Frozen unused wpe is invisible to DDP (requires_grad=False), so False is safe.
        ddp_find_unused_parameters=False,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_persistent_workers=workers > 0,
        dataloader_prefetch_factor=dataloader_prefetch_factor if workers > 0 else None,
        remove_unused_columns=False,
        seed=seed,
        use_cpu=smoke,
        # --- TRL packing ---
        packing=True,
        packing_strategy='bfd',
        max_length=context_length,
        eval_packing=True,
        # 'nll' (not TRL's 'chunked_nll' default): chunked_nll's 256-token lm_head chunks are
        # launch-bound here and cost ~3 s/step; 'nll' is the same math in one efficient matmul.
        loss_type=loss_type,
        # padding_free is auto-forced by packing + bfd (sft_trainer.py), so the collator
        # emits reset position_ids and omits attention_mask -> FlashAttention builds
        # cu_seq_lens from the resets to block cross-problem attention.
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    typer.echo('Starting training...')
    trainer.train()
    if eval_dataset is not None:
        final_metrics = trainer.evaluate()
        typer.echo(f'Final validation: loss={final_metrics["eval_loss"]:.4f}')
    trainer.save_model(str(output_dir / 'final'))
    typer.echo(f'Done. Model saved to {output_dir / "final"}')


if __name__ == '__main__':
    app()
