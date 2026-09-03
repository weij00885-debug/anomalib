#!/usr/bin/env bash
# Run Dinomaly + DINOv2-giant on the face dataset.
#
# Usage:
#   bash run_dinomaly_giant.sh          # full run (5000 steps)
#   bash run_dinomaly_giant.sh 20       # quick smoke test (20 steps)
# Early stopping is disabled by default (see train_dinomaly_face.py --help
# if you want to enable it with a validation interval).
set -e

STEPS="${1:-5000}"

python train_dinomaly_face.py \
    --encoder-name vit_giant_patch14_reg4_dinov2 \
    --train-batch-size 1 \
    --eval-batch-size 1 \
    --model-precision float16 \
    --max-steps "${STEPS}"
