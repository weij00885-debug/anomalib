#!/usr/bin/env bash
# Run DFM on OULU-NPU using GPU 1.
# Sets CUDA_VISIBLE_DEVICES=1 so DFM (which forces devices=1) uses physical GPU1.
set -e

export CUDA_VISIBLE_DEVICES=1

cd "$(dirname "$0")"

python train_oulu_npu_dfm.py \
  --src-root /mnt/d/BaiduNetdiskDownload/OULU-NPU \
  --data-root /mnt/d/BaiduNetdiskDownload/OULU-NPU_organized \
  --accelerator gpu \
  --pooling-kernel 1 \
  --pca-level 0.99 \
  --image-size 224 \
  --batch-size 16