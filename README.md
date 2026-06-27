# Physics of llms repro

## Before first use

### UV

In case you do not have uv, pls make yourself a favor and [install](https://docs.astral.sh/uv/getting-started/installation).

### Env

```
uv sync --all-extras # so that we have testing, precommit and fa
make pre-commit
git submodule update --init --recursive # clones iGSM
```

## Examples

### Training

Single gpu, no eval, no wandb
```
uv run accelerate launch --num_processes 1 --mixed_precision bf16 -m src.train.gpt --data-dir data/igsm_train --output-dir models/gpt2-rope-igsm --bf16 --gradient-checkpointing --max-steps 100 --warmup-steps 100 --per-device-train-batch-size 1 --gradient-accumulation-steps 2 --attn-implementation sdpa --no-eval
```

Multi gpu, eval, wandb
```
WANDB_ENTITY=m6rcin53-marcin-mazur WANDB_PROJECT=physics_of_llms CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 --mixed_precision bf16 -m src.train.gpt --report-to wandb --data-dir data/igsm_train --output-dir models/gpt2-rope-igsm --bf16 --gradient-checkpointing --max-steps 200 --warmup-steps 100 --per-device-train-batch-size 1 --gradient-accumulation-steps 2
```

We also added flash_attention option which also casts model weights to bf16, for biggest VRAM wins, to use it set `--attn-implementation flash_attention_2`.

### Data gen

We took first 10 parquets for train and last 2 for validation
```
uv run python -m src.data.igsm generate --split train --num-problems 120000 --workers 12 --batch-size 10000 --out data/igsm_train
```

### Eval

TODO, In progress
