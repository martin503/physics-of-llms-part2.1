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
### Data gen

```
uv run python -m src.data.igsm generate --split train --num-problems 300000 --workers 12 --batch-size 100000 --out data/igsm_train_100k
```
300000 problems -> 3 shards, 131421 packed windows of length 768 at data/igsm_train_100k
  note: paper trains 100k steps x batch 512 ~= 51M windows; this finite dataset is cycled over epochs for the working version.
and it took 1h, so it would take whole day on my pc to generate it.

### Training

Single gpu, no eval, no wandb
```
uv run accelerate launch --num_processes 1 --mixed_precision bf16 -m src.train.gpt --data-dir data/igsm_train --output-dir models/gpt2-rope-igsm --bf16 --gradient-checkpointing --max-steps 100 --warmup-steps 100 --per-device-train-batch-size 1 --gradient-accumulation-steps 2 --attn-implementation sdpa --no-eval
```

Multi gpu, eval, wandb
```
WANDB_ENTITY=m6rcin53-marcin-mazur WANDB_PROJECT=physics_of_llms CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 --mixed_precision bf16 -m src.train.gpt --report-to wandb --data-dir data/igsm_train --output-dir models/gpt2-rope-igsm --bf16 --gradient-checkpointing --max-steps 200 --warmup-steps 100 --per-device-train-batch-size 1 --gradient-accumulation-steps 2
```

Full training, with exactly same setup as paper
```
WANDB_ENTITY=m6rcin53-marcin-mazur WANDB_PROJECT=physics_of_llms CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 -m src.train.gpt --report-to wandb --max-steps 100_000 --logging-steps 1_000 --save-steps 10_000 --data-dir data/igsm_train_100k
```
This takes ~130 on 2x3090

We also added flash_attention option which also casts model weights to bf16, for biggest VRAM wins, to use it set `--attn-implementation flash_attention_2`.

### SLURM (multi-GPU cluster, e.g. gruenau9/10 A100s)

Pre-flight smoke test (real GPU path, 20 steps, 1 GPU, no wandb/eval/compile) before committing to a
multi-day job:

```
sbatch smoke.sbatch
squeue -u $USER
tail -f smoke-<JOBID>.log
```
Check it printed a dropping loss and finished cleanly:
```
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

Full training (100k steps, 3x A100, matches the paper's effective batch size closely: 16 x 11 x 3 = 528 vs 512):
```
sbatch train.sbatch
squeue -u $USER
tail -f train-<JOBID>.log
```
Resume after a timeout/kill (picks up the latest checkpoint automatically):
```
sbatch --export=ALL,RESUME=1 train.sbatch
```

### Eval

To generate the data without reask run
```
uv run python -m src.data.eval --seed 0 --out data/igsm_eval  --no-reask
```

To run the eval on smallest set of problems
```
uv run python -m src.eval.run --model models/gpt2-rope-igsm/checkpoint-800 --slices med_pq_op_le15 --batch-size 64 --limit 512
```

## Full repro

### Dummy model

1. Data
```
uv run python -m src.data.eval --seed 0 --out data/igsm_eval  --no-reask
mkdir -p data/smallest_pq_eval && cp data/igsm_eval/med_pq_op_le15.parquet data/igsm_eval/med_pq_op_le15.parquet
uv run python -m src.data.pack --in data/smallest_pq_eval --out data/smallest_pq_eval_sharded --ctx 768 --mode single --shard-size 500
```

2. Training

```
WANDB_ENTITY=m6rcin53-marcin-mazur WANDB_PROJECT=physics_of_llms CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 -m src.train.gpt --report-to wandb --max-steps 100_000 --logging-steps 10 --save-steps 500 --no-bf16 --output-dir models/gpt2-rope-igsm-100k-fp32 --data-dir data/igsm-117Mproblems-shuffled-merged
```
On 2x3090 it takes ~3h

3. Eval

```
uv run python -m src.eval.run --model models/gpt2-rope-igsm/checkpoint-11000 --slices med_pq_op_le15 --batch-size 64 --limit 64
```
I had to finish after 11k steps, but it still got pretty good results
```
Figure 3 (med) -- slice accuracy:
  slice                        n    accuracy
  med_pq_op_le15              64      0.7812
```

### Full training

Unfortunately for our case mixed precision did NOT work (probably due to no QK-norm), so we decided to go all the way fp32 (not even tf32!).

```
WANDB_ENTITY=m6rcin53-marcin-mazur WANDB_PROJECT=physics_of_llms CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes 2 -m src.train.gpt --report-to wandb --max-steps 100_000 --logging-steps 50 --save-steps 500 --no-bf16 --output-dir models/gpt2-rope-igsm-100k-fp32 --data-dir data/igsm-117Mproblems-shuffled-merged --per-device-train-batch-size 16 --gradient-accumulation-steps 16 --dataloader-num-workers 2 --streaming
```
~250h on 2x3090
