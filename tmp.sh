#!/bin/bash

set -e

uv run python scripts/evaluate.py \
  -g 4 \
  -e=outputs/ti-sana1.5_4.8b-camera \
  --checkpoint=500 \
  --rescale=mean \
  --out_dir=images_rescaled

uv run python scripts/evaluate.py \
  -g 4 \
  -e=outputs/ti-sana1.5_4.8b-camera \
  --checkpoint=1000 \
  --rescale=mean \
  --out_dir=images_rescaled
