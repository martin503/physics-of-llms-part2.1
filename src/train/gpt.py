"""Train GPT-2 + RoPE on iGSM-med (Physics-of-LMs Part 2.1 reproduction).

Uses the HuggingFace ``Trainer`` (which wraps ``accelerate``) for multi-GPU DDP and Weights
& Biases tracking. Trains over a pre-generated, packed iGSM dataset produced by
``src.data.igsm``.

Examples::

    # CPU smoke test (tiny model, 4 steps, no GPU/wandb):
    uv run python -m src.train.gpt --data-dir data/igsm_tiny --output-dir /tmp/smoke --smoke

    # Real run on 2x3090 (Trainer auto-DDPs via torchrun):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 -m src.train.gpt \
        --data-dir data/igsm_train --output-dir models/gpt2-rope-igsm
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    GPT2TokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.data.igsm import DEFAULT_CONTEXT_LENGTH, VOCAB_SIZE, load_igsm_dataset
from src.model.gpt2_rope import build_gpt2_config, build_gpt2_rope


class PackedDataset(TorchDataset):
    """Thin ``torch.Dataset`` view over a pre-gen HF dataset's ``input_ids`` column."""

    def __init__(self, hf_dataset: Any) -> None:
        self.input_ids: list[list[int]] = hf_dataset['input_ids']
        assert self.input_ids, 'dataset is empty'
        self.length = len(self.input_ids[0])
        assert all(len(row) == self.length for row in self.input_ids), 'non-uniform window lengths'

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return {'input_ids': self.input_ids[idx]}


def make_collator(context_length: int):
    """Collate fixed-length packed int windows: ``labels = input_ids`` (no masking)."""

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
        assert input_ids.shape == (len(batch), context_length), (
            f'bad batch shape {input_ids.shape}'
        )
        return {'input_ids': input_ids, 'labels': input_ids.clone()}

    return collate


class LogSampleCallback(TrainerCallback):
    """Periodically decode+log one packed window to W&B for sanity inspection."""

    def __init__(self, tokenizer: GPT2TokenizerFast, every: int = 500) -> None:
        self.tokenizer = tokenizer
        self.every = every
        self._sample: torch.Tensor | None = None

    def on_step_end(self, args, state, control, train_dataloader=None, **kwargs) -> None:  # noqa: ARG002
        if not state.is_world_process_zero:
            return
        if state.global_step == 0 or state.global_step % self.every != 0:
            return
        if self._sample is None and train_dataloader is not None:
            try:
                self._sample = next(iter(train_dataloader))['input_ids']
            except Exception:  # noqa: BLE001 -- degrade silently if a batch can't be pulled
                self._sample = None
                return
        if self._sample is None:
            return
        import wandb

        text = self.tokenizer.decode(self._sample[0], skip_special_tokens=False)
        wandb.log(
            {'train/sample_text': wandb.Html(f'<pre>{text[:2000]}</pre>')},
            step=state.global_step,
        )


app = typer.Typer(add_completion=False, help='Train GPT-2 + RoPE on iGSM-med.')


@app.command()
def train(
    data_dir: Annotated[
        Path, typer.Option('--data-dir', help='Pre-gen packed iGSM dataset dir.')
    ] = Path('data/igsm_train'),
    output_dir: Annotated[
        Path, typer.Option('--output-dir', help='Where to write checkpoints.')
    ] = Path('models/gpt2-rope-igsm'),
    context_length: Annotated[int, typer.Option('--context-length')] = DEFAULT_CONTEXT_LENGTH,
    dataloader_num_workers: Annotated[int, typer.Option('--dataloader-num-workers')] = 4,
    per_device_train_batch_size: Annotated[
        int, typer.Option('--per-device-train-batch-size')
    ] = 16,
    gradient_accumulation_steps: Annotated[
        int, typer.Option('--gradient-accumulation-steps')
    ] = 16,
    learning_rate: Annotated[float, typer.Option('--learning-rate')] = 2e-3,
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
    gradient_checkpointing: Annotated[bool, typer.Option('--gradient-checkpointing')] = False,
    logging_steps: Annotated[int, typer.Option('--logging-steps')] = 20,
    save_steps: Annotated[int, typer.Option('--save-steps')] = 1_000,
    eval_data_dir: Annotated[
        Path, typer.Option('--eval-data-dir', help='Packed iGSM validation dataset dir.')
    ] = Path('data/igsm_val'),
    per_device_eval_batch_size: Annotated[int, typer.Option('--per-device-eval-batch-size')] = 8,
    eval_steps: Annotated[
        int, typer.Option('--eval-steps', help='Validate every N optimizer steps.')
    ] = 1_000,
    max_eval_samples: Annotated[
        int | None, typer.Option('--max-eval-samples', help='Cap val-set size (None = all).')
    ] = None,
    no_eval: Annotated[
        bool, typer.Option('--no-eval', help='Disable validation entirely.')
    ] = False,
    attn_pdrop: Annotated[float, typer.Option('--attn-pdrop')] = 0.1,
    resid_pdrop: Annotated[float, typer.Option('--resid-pdrop')] = 0.1,
    embd_pdrop: Annotated[float, typer.Option('--embd-pdrop')] = 0.1,
    max_grad_norm: Annotated[
        float, typer.Option('--max-grad-norm', help='Max gradient norm (0 = unlimited).')
    ] = 1.0,
    attn_implementation: Annotated[str, typer.Option('--attn-implementation')] = 'sdpa',
    report_to: Annotated[list[str] | None, typer.Option('--report-to')] = None,
    seed: Annotated[int, typer.Option('--seed')] = 0,
    smoke: Annotated[
        bool, typer.Option('--smoke', help='CPU tiny-model smoke run (overrides most args).')
    ] = False,
) -> None:
    """Train GPT-2 + RoPE on a pre-generated packed iGSM-med dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_to = report_to or []

    if smoke:
        # Tiny model + few steps on CPU, no wandb -- validates the full pipeline offline.
        config = build_gpt2_config(
            n_layer=2,
            n_head=2,
            n_embd=64,
            n_inner=256,
            vocab_size=VOCAB_SIZE,
            n_positions=context_length,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
        )
        per_device_train_batch_size = 2
        gradient_accumulation_steps = 1
        max_steps = 4
        bf16 = False
        report_to = []
        logging_steps = 1
        save_steps = max_steps
        dataloader_num_workers = 0
        max_eval_samples = 16  # don't iterate all val windows on CPU
        per_device_eval_batch_size = 2
        eval_steps = max_steps  # = 4 -> eval runs once during training, plus the final one
    else:
        config = build_gpt2_config(
            vocab_size=VOCAB_SIZE,
            n_positions=max(2048, context_length),
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
        )
        dataloader_num_workers = dataloader_num_workers

    typer.echo(
        f'Building GPT-2 + RoPE ({"smoke" if smoke else "12-12"}): '
        f'{config.num_hidden_layers}L {config.n_embd}d {config.num_attention_heads}h'
    )
    if bf16 and attn_implementation == 'flash_attention_2':
        config.dtype = torch.bfloat16
    model = build_gpt2_rope(config, attn_implementation=attn_implementation)
    if bf16 and attn_implementation == 'flash_attention_2':
        model.to(torch.bfloat16)

    typer.echo(f'Loading packed dataset from {data_dir}...')
    hf_dataset = load_igsm_dataset(data_dir)
    train_dataset = PackedDataset(hf_dataset)
    typer.echo(f'  {len(train_dataset)} examples of length {train_dataset.length}')

    eval_dataset: PackedDataset | None = None
    if no_eval:
        typer.echo('  (validation disabled via --no-eval)')
    else:
        try:
            eval_hf = load_igsm_dataset(eval_data_dir)  # raises FileNotFoundError if no shards
            if max_eval_samples is not None:
                eval_hf = eval_hf.select(range(min(max_eval_samples, len(eval_hf))))
            eval_dataset = PackedDataset(eval_hf)
            typer.echo(f'  eval: {len(eval_dataset)} examples of length {eval_dataset.length}')
        except FileNotFoundError:
            typer.echo(f'  (no eval data at {eval_data_dir}; skipping validation)')

    callbacks: list[TrainerCallback] = []
    if 'wandb' in report_to and not smoke:
        try:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            callbacks.append(LogSampleCallback(tokenizer, every=max(500, logging_steps)))
        except Exception as e:  # noqa: BLE001 -- tokenizer download may fail offline
            typer.echo(f'  (skipping sample logging: {e})')

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm if max_grad_norm > 0 else 1000.0,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate': 0.01},
        bf16=bf16,
        torch_compile=True,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy='steps' if eval_dataset is not None else 'no',
        eval_steps=eval_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        report_to=report_to,
        # Frozen unused wpe is invisible to DDP (requires_grad=False), so False is safe.
        ddp_find_unused_parameters=False,
        dataloader_num_workers=dataloader_num_workers,
        remove_unused_columns=False,
        seed=seed,
        use_cpu=smoke,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=make_collator(train_dataset.length),
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
