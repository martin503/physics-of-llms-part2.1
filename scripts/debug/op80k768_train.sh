#!/usr/bin/env bash
# Re-run the 80k PLASTICITY experiment at ctx768 / gbs512 / fa2, full 10k.
#
# Same idea as op80k_train.sh (continue from the plastic 80k checkpoint, lr2e-4, warmup100,
# cosine_with_min_lr floored at 7e-5 via min_lr_rate=0.35 = the value the original 100k cosine
# reaches at step 90k), but with the user-requested changes:
#   - ctx768 data (data/igsm_raw_30M_eq1_15_packed768) -- the natural context for this checkpoint
#     (base 80k was ctx768-trained), no RoPE extrapolation, far more token-efficient.
#   - gbs512 (per_dev64 x accum4 x 2 GPUs). per_dev64 is the speed target; if it OOMs, relaunch
#     with PER_DEV=32 ACCUM=8 (the proven ctx768 footprint).
#   - flash_attention_2 for speed/VRAM. NOTE: in this codebase fa2 triggers model.to(bfloat16)
#     (src/train/gpt.py), i.e. PURE bf16 (bf16 weights + bf16 AdamW state, no fp32 master) -- the
#     diagnosed-broken-without-QK-norm mode. Health-watch the first ~200 steps; if it floors /
#     spikes / NaNs, relaunch with ATTN=sdpa (the proven op80k mixed-precision mode).
#   - full 10k, save every 500 (brackets the eq20 transient, which at 393k tok/step lands in
#     steps ~300-1300, gone by ~2700).
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${WANDB_ENTITY:=m6rcin53-marcin-mazur}"
: "${WANDB_PROJECT:=physics_of_llms}"
DATA=${DATA:-data/igsm_raw_30M_eq1_15_packed768}
INIT=${INIT:-models/100k_model/checkpoint-80000-fixed}
OUT=${OUT:-sweeps/disc_80k_op1_15_lr2e4_gbs512_ctx768_fa2}
LR=${LR:-2e-4}
MAX_STEPS=${MAX_STEPS:-10000}
SAVE_STEPS=${SAVE_STEPS:-500}
# floor = lr the original 100k cosine reaches at step 90k ~= 6.94e-5; at peak 2e-4 -> 0.35.
MIN_LR_RATE=${MIN_LR_RATE:-0.35}
# fa2 + pure-bf16 config (override for the sdpa+mixed fallback: ATTN=sdpa PER_DEV=32 ACCUM=8 [GRAD_CKPT=1]).
ATTN=${ATTN:-flash_attention_2}
PER_DEV=${PER_DEV:-32}
ACCUM=${ACCUM:-8}
GRAD_CKPT_FLAG=""
if [ "${GRAD_CKPT:-0}" = "1" ]; then GRAD_CKPT_FLAG="--gradient-checkpointing"; fi

mkdir -p "$OUT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" \
uv run accelerate launch --num_processes 2 -m src.train.gpt \
  --report-to wandb --max-steps "$MAX_STEPS" --logging-steps 10 \
  --save-steps "$SAVE_STEPS" --save-total-limit 999 --save-only-model $GRAD_CKPT_FLAG \
  --attn-implementation "$ATTN" \
  --context-length 768 --warmup-steps 100 --lr-scheduler-type cosine_with_min_lr \
  --lr-scheduler-kwargs "{\"min_lr_rate\": $MIN_LR_RATE}" \
  --no-eval --data-dir "$DATA" \
  --per-device-train-batch-size "$PER_DEV" --gradient-accumulation-steps "$ACCUM" --streaming \
  --init-from "$INIT" --learning-rate "$LR" --output-dir "$OUT"
echo "OP80K768_TRAIN_DONE"
