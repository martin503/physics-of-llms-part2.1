#!/bin/bash
# Pre-flight smoke test: real GPU path, 20 steps, 1 A100. Confirms the model trains and
# (crucially) that the dataset loads within memory on a compute node before the 4-day run.
# Submit from the repo root:  sbatch smoke.sbatch   (then: tail -f smoke-<JOBID>.log)

#SBATCH --job-name=igsm-smoke
#SBATCH --partition=gpu                   # short job -> ordinary gpu queue is fine
#SBATCH --nodes=1
#SBATCH --gres=gpu:a10080gb:1             # 1 A100 is enough for a smoke test
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G                         # 1 rank: ~1x decoded arrow (~30-60G) + headroom
#SBATCH --time=0:30:00
#SBATCH --output=smoke-%j.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Single process, no wandb, no eval, no torch.compile (compile warmup dwarfs a 20-step run).
uv run accelerate launch --num_processes 1 \
    -m src.train.gpt \
    --max-steps 20 --warmup-steps 20 \
    --gradient-accumulation-steps 2 --no-eval \
    --output-dir models/smoke \
    --data-dir ../../../../../tmp/josebast_igsm_data/data/igsm_train_117M
