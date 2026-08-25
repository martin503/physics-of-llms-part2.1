# Physics of llms part2.1 repro

## Intro

Purpose of this repo is to fill a small gap in physics of llms series by training the main gpt2 model (we would love to train more architectures/configurations, but we are gpu poor) used for analysis of reasoning of gsm like problems. We provide [data](https://huggingface.co/datasets/SimulatedScience/igsm-med-120Mproblems), [model](https://huggingface.co/SimulatedScience/gpt2-igsm-med) and some [probes](https://huggingface.co/SimulatedScience/gpt2-igsm-med/tree/main/probes). Unfortunately we could not get the same results as reported in the [paper](https://arxiv.org/abs/2407.20311), nevertheless we still reproduce one of the main findings:

> Do models trained solely on grade-school math problems only learn to solve these problems, or do they develop some more general intelligence?

~ **Yes!**

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
For whole data download (~34Gb) including training/eval:
```
hf download --type dataset SimulatedScience/igsm-med-120Mproblems --local-dir data/
```

If you are interested in only some parts, you can go to our HF [repo](https://huggingface.co/datasets/SimulatedScience/igsm-med-120Mproblems) and use `--include` flag.

## Pre-training

### Local

Full training, ddp=2, no eval
```
uv run accelerate launch --num_processes 2 -m src.train.gpt --max-steps 100_000 --logging-steps 10 --save-steps 10_000 --context-length 768 --no-eval --streaming --torch-compile
```

This takes ~120h on 2x3090

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

You can download our best model like that:
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

| Figure 3 slice | Paper † | Ours | Δ (pts) |
|---|---:|---:|---:|
| op≤15 (in-dist) | 99.9 | 99.66 | −0.2 |
| op=15 | 99.1 | 98.49 | −0.6 |
| op=20 | 91.8 | 70.48 | −21.3 |
| op=21 | 87.9 | 47.53 | −40.4 |
| op=22 | 84.0 | 21.26 | −62.7 |
| op=23 | 76.8 | 6.03 | −70.8 |
| op=20 (reask) | 91.6 | 79.39 | −12.2 |

## Probing

### Training

This repo currently supports 2 out of 6 probes described in the paper:
* dep(A, B) - if parameter A (recursively) depends on parameter B
* nece(A) - whether parameter A is necessary to get the answer

Both trainings below use only 2.5% of what paper suggests in appendix, but results are pretty good, so we did not push it further.

dep (takes ~1h on single 3090)
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 uv run python -m src.probe.run vprobe --target dep --model-path models/gpt2-igsm-med --data data/probes/dep/vprobe_dep_train_80k --epochs 1 --batch-size 32 --lr 1e-3 --weight-decay 0.01 --seed 0 --no-grad-checkpointing
```

nece (takes ~1h on single 3090)
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 uv run python -m src.probe.run vprobe --target nece --model-path models/gpt2-igsm-med --data data/probes/nece/vprobe_nece_train_80k --epochs 1 --batch-size 32 --lr 1e-3 --weight-decay 0.01 --balance-classes --seed 0
```

### Eval

You can find our dep probe [here](https://huggingface.co/SimulatedScience/gpt2-igsm-med/tree/main/probes/vprobe-dep).

#### dep
```
uv run python scripts/probe/eval_fig7a.py --run-dir <dep_probe_dir>
```

| Figure 7a slice | Paper † | Ours | Δ (pts) |
|---|---:|---:|---:|
| op≤15 (in-dist) | 99.7 | 98.9 | −0.8 |
| op=15 | 99.3 | 98.0 | −1.3 |
| op=20 | 100.0 | 97.8 | −2.2 |
| op=21 | 100.0 | 97.2 | −2.8 |
| op=22 | 100.0 | 97.0 | −3.0 |
| op=23 | 100.0 | 97.0 | −3.0 |

#### nece

You can find our nece probe [here](https://huggingface.co/SimulatedScience/gpt2-igsm-med/tree/main/probes/vprobe-nece).

```
uv run python scripts/probe/eval_fig7a.py --run-dir <nece_probe_dir>
```

| Figure 7a slice | Paper † | Ours | Δ (pts) |
|---|---:|---:|---:|
| op≤15 (in-dist) | 99.8 | 99.7 | −0.1 |
| op=15 | 99.8 | 99.3 | −0.5 |
| op=20 | 98.7 | 95.9 | −2.8 |
| op=21 | 97.9 | 93.7 | −4.2 |
| op=22 | 96.9 | 92.0 | −4.9 |
| op=23 | 94.7 | 90.2 | −4.5 |

## Acknowledgements

- [iGSM repo](https://github.com/facebookresearch/iGSM)
- [YT playlist](https://youtube.com/playlist?list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp&si=t9UUgMmF5lEJ2ozN)
