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

## Data
For whole data download (~34Gb) for training/eval/probes:
```
hf download --type dataset SimulatedScience/igsm-med-120Mproblems --local-dir data/
```

If you are interested in only some parts, you can go to our HF [repo](https://huggingface.co/datasets/SimulatedScience/igsm-med-120Mproblems) and use `--include` flag.

## Pre-training

### Local

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
This takes ~130h on 2x3090

We also added flash_attention option which also casts model weights to bf16, for biggest VRAM wins, to use it set `--attn-implementation flash_attention_2`.

### SLURM (multi-GPU cluster, e.g. gruenau9/10 A100s)

Pre-flight smoke test (real GPU path, 20 steps, 1 GPU, no wandb/eval/compile) before committing to a
multi-day job:

```
sbatch scripts/runs/smoke.sbatch
squeue -u $USER
tail -f smoke-<JOBID>.log
```
Check it printed a dropping loss and finished cleanly:
```
sacct -j <JOBID> --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed
```

Full training (100k steps, 3x A100, matches the paper's effective batch size closely: 16 x 11 x 3 = 528 vs 512):
```
sbatch scripts/runs/train.sbatch
squeue -u $USER
tail -f train-<JOBID>.log
```
Resume after a timeout/kill (picks up the latest checkpoint automatically):
```
sbatch --export=ALL,RESUME=1 train.sbatch
```

### Eval

Download best model:
```
hf download SimulatedScience/gpt2-igsm-med --include "model/20260730/final/*" --local-dir models/tmp && mv models/tmp/model/20260730/final models/gpt2-igsm-med && rm -r models/tmp
```

To run the eval on smallest set of problems (make batch-size smaller in case of OOM)
```
uv run python -m src.eval.run --slices med_pq_op_le15 --batch-size 64
```

Full eval for pq
```
uv run python -m src.eval.run --batch-size 64
```

Results:


## Probing

In case you would like to


I am not sure why seed 1_000_000 when we already have the split between train and test? - its connected     to --dep-all-pairs, split doesnt matter anymore
I was thinking of adding another control model, normal pretrained gpt, cause one could say that random      init is completely different.

Run the training of the probes:
* original gpt2
* main, we need higher than 85 we are aimimng for 95, more than 2 epochs, checkpoint after each epoch

Probe can_next/nec_next if I have time
