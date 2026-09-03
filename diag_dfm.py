"""Quick DFM diagnostic: does the FRE score separate normal vs abnormal?

Fits DFM on the first few train batches, then scores up to N test images per
class and prints score stats + a hand-computed ROC-AUC. No full training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib.data import Folder
from anomalib.models.image import Dfm
from anomalib.pre_processing import PreProcessor

AUC_README = "placeholder"


def roc_auc(scores: list[float], labels: list[int]) -> float:
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(labels, scores)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/d/BaiduNetdiskDownload/OULU-NPU_organized"))
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--layer", type=str, default="layer3")
    parser.add_argument("--pooling-kernel", type=int, default=1)
    parser.add_argument("--pca-level", type=float, default=0.99)
    parser.add_argument("--score-type", type=str, default="fre")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fit-batches", type=int, default=4, help="train batches used to fit PCA")
    parser.add_argument("--per-class", type=int, default=300, help="max test images scored per class")
    args = parser.parse_args()

    image_size = (args.image_size, args.image_size)
    pre = PreProcessor(transform=Compose([
        Resize(image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]))
    dm = Folder(
        name="oulu",
        root=str(args.data_root),
        normal_dir="train/normal",
        abnormal_dir="test/abnormal",
        normal_test_dir="test/normal",
        train_batch_size=8,
        eval_batch_size=8,
        num_workers=0,
        val_split_mode="from_train",
        val_split_ratio=0.1,
    )
    dm.setup()
    train_dl = dm.train_dataloader()
    test_dl = dm.test_dataloader()

    model = Dfm(
        backbone=args.backbone, layer=args.layer, pre_trained=True,
        pooling_kernel_size=args.pooling_kernel, pca_level=args.pca_level,
        score_type=args.score_type, pre_processor=pre,
    )
    model.eval()
    dm_model = model.model

    # --- fit on a slice of train features ---
    mem = []
    for i, batch in enumerate(train_dl):
        if i >= args.fit_batches:
            break
        with torch.no_grad():
            feats, shape = dm_model.get_features(batch.image)
        mem.append(feats)
    memory = torch.cat(mem, dim=0)
    print(f"[fit] memory bank: {tuple(memory.shape)}")
    dm_model.memory_bank = [memory]
    dm_model.pca_model.fit(memory.float())
    ncomp = int(dm_model.pca_model.num_components)
    print(f"[fit] PCA components kept: {ncomp}")
    mem_total = None
    del mem, memory

    # --- score test images per class ---
    scores_n, scores_a = [], []
    with torch.no_grad():
        for batch in test_dl:
            imgs = batch.image
            gt = batch.gt_label
            out = dm_model(imgs)
            sc = out.pred_score.cpu().tolist()
            lbl = gt.cpu().tolist()
            for s, l in zip(sc, lbl):
                if l == 0 and len(scores_n) < args.per_class:
                    scores_n.append(s)
                elif l == 1 and len(scores_a) < args.per_class:
                    scores_a.append(s)
            if len(scores_n) >= args.per_class and len(scores_a) >= args.per_class:
                break

    print(f"[test] normal n={len(scores_n)}, min={min(scores_n):.4f}, max={max(scores_n):.4f}, mean={sum(scores_n)/len(scores_n):.4f}")
    print(f"[test] abnormal n={len(scores_a)}, min={min(scores_a):.4f}, max={max(scores_a):.4f}, mean={sum(scores_a)/len(scores_a):.4f}")
    import statistics
    print(f"[test] normal median={statistics.median(scores_n):.4f}, abnormal median={statistics.median(scores_a):.4f}")

    all_scores = scores_n + scores_a
    all_labels = [0] * len(scores_n) + [1] * len(scores_a)
    auc = roc_auc(all_scores, all_labels)
    print(f"[result] hand-computed ROC-AUC = {auc:.4f}")


if __name__ == "__main__":
    main()