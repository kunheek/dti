#!/bin/bash

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU="5"
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

# ── 1. Ours (DTI) + eval ────────────────────────────────────────────────────
echo ""
echo ">>> Running Ours (DTI) ..."
uv run python exps/ours_sana.py \
  -g "$GPU" \
  -m "$MODEL" \
  --total_steps "$TOTAL_STEPS" \
  --desc "$DESC" \
  --instances colorful_sneaker dog dog2 dog3 dog5 dog6 dog7 dog8 duck_toy fancy_boot grey_sloth_plushie monster_toy pink_sunglasses
