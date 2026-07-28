#!/usr/bin/env bash
# LR staircase DOWN from 1e-5: gbs64, CONSTANT LR, no warmup, no schedule.
# Find the floor where eq15 stops saturating (and le15 stays ~0.99).
# Args: [max_steps=200] [save_steps=50]
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WANDB_ENTITY:=m6rcin53-marcin-mazur}"
: "${WANDB_PROJECT:=physics_of_llms}"
DATA=data/igsm_raw_10M_eq11_15_packed768
INIT=models/100k_model/gpt-rope-igsm-fixed
MAX_STEPS=${1:-200}
SAVE_STEPS=${2:-50}
LRS=${3:-"8e-6 6e-6 4e-6 2e-6"}

for LR in $LRS; do
  TAG=$(echo "$LR" | tr -d '-')   # 8e6, 6e6, 4e6, 2e6
  OUT=sweeps/disc_gbs64_lr${TAG}
  mkdir -p "$OUT"
  echo "===== LR=$LR -> $OUT (max=$MAX_STEPS save=$SAVE_STEPS) ====="
  CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" \
  uv run accelerate launch --num_processes 2 -m src.train.gpt \
    --report-to wandb --max-steps "$MAX_STEPS" --logging-steps 10 --save-steps "$SAVE_STEPS" --save-total-limit 999 --save-only-model \
    --context-length 768 --warmup-steps 0 --lr-scheduler-type constant \
    --no-eval --data-dir "$DATA" \
    --per-device-train-batch-size 32 --gradient-accumulation-steps 1 --streaming \
    --init-from "$INIT" --learning-rate "$LR" --output-dir "$OUT"
done
echo "LR_STAIR_TRAIN_DONE"
