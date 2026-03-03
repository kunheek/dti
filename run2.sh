#!/bin/bash

set -e

# ── Defaults ─────────────────────────────────────────────────────────────────
GPU="2"
MODEL="sana1.5_1.6b"
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
  --instances dog6 dog7 dog8 duck_toy fancy_boot grey_sloth_plushie monster_toy pink_sunglasses poop_emoji rc_car red_cartoon robot_toy shiny_sneaker teapot vase wolf_plushie
