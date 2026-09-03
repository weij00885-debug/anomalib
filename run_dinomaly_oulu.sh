#!/usr/bin/env bash
# Run Dinomaly + DINOv2-giant on the organized OULU-NPU face subset.
#
# Note: this local OULU-NPU copy only contains 6 subjects per split
# (train/test/dev), so this is a quick sanity run, not the official
# subject-disjoint protocol benchmark.
#
# Usage:
#   bash run_dinomaly_oulu.sh          # full run (5000 steps)
#   bash run_dinomaly_oulu.sh 20       # quick smoke test (20 steps)
set -e

STEPS="${1:-5000}"

python train_dinomaly_face.py \
    --root /mnt/d/BaiduNetdiskDownload/OULU-NPU_organized \
    --name oulu_npu \
    --normal-dir train/normal \
    --abnormal-dir test/abnormal \
    --normal-test-dir test/normal \
    --encoder-name vit_giant_patch14_reg4_dinov2 \
    --image-size 224 \
    --train-batch-size 8 \
    --eval-batch-size 8 \
    --model-precision float16 \
    --max-steps "${STEPS}"
