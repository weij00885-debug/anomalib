#!/usr/bin/env bash
# Run Dinomaly + DINOv2-giant on the face-cropped (128px) CASIA-FASD frames.
#
# The mtcnn_128 directory already follows the official CASIA-FASD label
# semantics: normal/ = real frames (1, 2, HR_1), abnormal/ = attack frames
# (3..8, HR_2..4), test/ = held-out real frames.
#
# Usage:
#   bash run_dinomaly_mtcnn128.sh          # full run (5000 steps)
#   bash run_dinomaly_mtcnn128.sh 20       # quick smoke test (20 steps)
set -e

STEPS="${1:-5000}"

python train_dinomaly_face.py \
    --root /mnt/c/Users/heqi/Desktop/CASIA-FASD/casia-fasd_mtcnn_128 \
    --name casia_mtcnn128 \
    --normal-dir normal \
    --abnormal-dir abnormal \
    --normal-test-dir test \
    --encoder-name vit_giant_patch14_reg4_dinov2 \
    --image-size 126 \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --model-precision float16 \
    --max-steps "${STEPS}"
