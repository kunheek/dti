#!/bin/bash

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU="6"
MODEL="sana1.5_4.8b"
DESC="camera"
TOTAL_STEPS=1000

echo "============================================"
echo " SANA Experiment Pipeline"
echo " GPU:       $GPU"
echo " Model:     $MODEL"
echo " Desc:      $DESC"
echo " Steps:     $TOTAL_STEPS"
echo "============================================"

# ── 1. TI + eval ────────────────────────────────────────────────────
echo ""
echo ">>> Running TI ..."
uv run python exps/ti_sana.py \
  -g "$GPU" \
  -m "$MODEL" \
  --total_steps "$TOTAL_STEPS" \
  --desc "$DESC" \
  --instances colorful_sneaker dog dog2 dog3 dog5 dog6 dog7 dog8 duck_toy fancy_boot grey_sloth_plushie monster_toy pink_sunglasses poop_emoji

# Re-evaluate TI with embeddings rescaled to mean vocab norm.
TI_EXPDIR="outputs/ti-${MODEL}"
if [[ -n "$DESC" ]]; then
  TI_EXPDIR="${TI_EXPDIR}-${DESC}"
fi

uv run python scripts/evaluate.py \
  -g $GPU \
  -e=$TI_EXPDIR \
  --checkpoint=500 \
  --rescale=mean \
  --out_dir=images_rescaled

uv run python scripts/evaluate.py \
  -g $GPU \
  -e=$TI_EXPDIR \
  --checkpoint=$TOTAL_STEPS \
  --rescale=mean \
  --out_dir=images_rescaled
