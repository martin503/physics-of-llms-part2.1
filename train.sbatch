#!/bin/bash
# Full training: GPT-2 + RoPE on iGSM-med, 100k steps, 3x A100-80GB.
# Submit from the repo root:  sbatch train.sbatch
# Resume after a kill/timeout:  sbatch --export=ALL,RESUME=1 train.sbatch

#SBATCH --job-name=igsm-gpt2-rope
#SBATCH --partition=longgpu
#SBATCH --nodes=1                        # one node, 3 GPUs on it (DDP within the box)
#SBATCH --gres=gpu:a10080gb:3            # 3x A100-80GB
#SBATCH --cpus-per-task=8                # dataloader workers + generation headroom
#SBATCH --mem=96G                        # 3 ranks share this node; each needs dataloader buffers,
                                         # pinned memory, and a torch.compile spike at step 0
#SBATCH --time=4-00:00:00                # 4 days
#SBATCH --output=train-%j.log            # %j = job id; live view with `tail -f`

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# SLURM already sets CUDA_VISIBLE_DEVICES to the 3 allocated GPUs
export WANDB_ENTITY=physics_of_lms2-1_collab
export WANDB_PROJECT=physics_of_lms2.1
export WANDB_MODE=offline

# Optional resume: `sbatch --export=ALL,RESUME=1 train.sbatch` picks up the latest checkpoint.
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then
    RESUME_FLAG="--resume"
fi

# run with eval and WandB logging on 3 GPUs
uv run accelerate launch --num_processes 3 --mixed_precision bf16 \
    -m src.train.gpt --report-to wandb \
    --max-steps 100_000 --logging-steps 1_000 --save-steps 10_000 \
    --eval-data-dir data/igsm_val --eval-steps 1000 --max-eval-samples 2000 \
    --gradient-accumulation-steps 11 --torch-compile --streaming \
    --data-dir ../../../../../tmp/josebast_igsm_data/data/igsm_train_117M \
    $RESUME_FLAG
