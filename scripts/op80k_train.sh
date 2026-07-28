#!/usr/bin/env bash
# Continue training from the PLASTIC 80k checkpoint (NOT the converged 100k model),
# to test whether a less-converged model develops op>15 length generalization (eq20)
# better than SFT on the converged 100k did.
#
# lr2e-4 (~the LR the original 100k cosine was AT at step 80k), gbs256 (per_dev16 x
# accum8) ctx2048, warmup100 + cosine_with_min_lr. min_lr is FLOORED at ~7e-5 = the
# value the original 100k schedule (lr2e-3, warmup1k, min_lr_rate0.01) reaches at step
# 90k (~6.94e-5); we never anneal below where the 100k model would be over the 80k->90k
# segment we're effectively replaying. With peak 2e-4 that floor is min_lr_rate=0.35.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${WANDB_ENTITY:=m6rcin53-marcin-mazur}"
: "${WANDB_PROJECT:=physics_of_llms}"
DATA=${DATA:-data/igsm_raw_30M_eq1_15_packed2048}
INIT=${INIT:-models/100k_model/checkpoint-80000-fixed}
OUT=${OUT:-sweeps/disc_80k_op1_15_lr2e4_gbs256_ctx2048}
LR=${LR:-2e-4}
MAX_STEPS=${1:-10000}
SAVE_STEPS=${2:-1000}
# floor = lr the original 100k cosine reaches at step 90k ~= 6.94e-5; at peak 2e-4 -> 0.35.
MIN_LR_RATE=${MIN_LR_RATE:-0.35}

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" \
uv run accelerate launch --num_processes 2 -m src.train.gpt \
  --report-to wandb --max-steps "$MAX_STEPS" --logging-steps 10 \
  --save-steps "$SAVE_STEPS" --save-total-limit 999 --save-only-model --gradient-checkpointing \
  --context-length 2048 --warmup-steps 100 --lr-scheduler-type cosine_with_min_lr \
  --lr-scheduler-kwargs "{\"min_lr_rate\": $MIN_LR_RATE}" \
  --no-eval --data-dir "$DATA" \
  --per-device-train-batch-size 16 --gradient-accumulation-steps 8 --streaming \
  --init-from "$INIT" --learning-rate "$LR" --output-dir "$OUT"
echo "OP80K_TRAIN_DONE"
