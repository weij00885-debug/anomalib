"""Train DFM (Deep Feature Modeling) on the OULU-NPU face PAD dataset.

DFM fits a PCA + Gaussian distribution on deep features extracted from a
pretrained CNN backbone, then detects attacks (spoofs) as outliers of the
"live" (normal) distribution. OULU-NPU is a face presentation-attack dataset.

Feature-dimension knobs
-----------------------
The embedding is ``channels x spatial_patches``. Two main levers enlarge it:

  * ``--pooling-kernel 1``  removes spatial average-pooling so every patch
    stays as a feature (e.g. resnet50/layer3 on 224x224 gives ~200k dims
    vs ~9k for the default pooling=4).
  * ``--pca-level``         raised toward 1.0 keeps more PCA components,
    i.e. a larger effective low-rank feature dimension.
  * ``--backbone``/``--layer`` select a wider/deeper net -> more channels.
  * ``--image-size``        larger input -> more spatial patches.

Data layout expects (produced automatically from a raw OULU-NPU root when
``--src-root`` is given)::

    data_root/
    |-- train/normal/     # lives only (one-class training)
    |-- test/normal/      # lives for evaluation
    `-- test/abnormal/    # attack / spoof frames for evaluation

Typical usage::

    python train_oulu_npu_dfm.py \\
        --src-root /mnt/d/BaiduNetdiskDownload/OULU-NPU \\
        --data-root /mnt/d/BaiduNetdiskDownload/OULU-NPU_organized \\
        --pooling-kernel 1 --pca-level 0.99 --backbone resnet50 \\
        --image-size 224 --batch-size 16 --accelerator gpu
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models.image import Dfm
from anomalib.pre_processing import PreProcessor


def print_sep(title: str = "") -> None:
    """Print a visible section separator."""
    bar = "=" * 70
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)


def organize_oulu_npu(dataset_root: Path, src_root: Path, samples_per_folder: int) -> None:
    """Organize raw OULU-NPU (a_b_c_d folders) into the Folder layout.

    Raw structure::

        src_root/train/train/{1_1_01_1, ...}/frames.jpg   (d-th field: 1 = live)
        src_root/test/test/...
        src_root/dev/dev/...

    Output::

        dataset_root/train/normal
        dataset_root/test/normal
        dataset_root/test/abnormal
    """
    dst_root = Path(dataset_root)
    roots = (dst_root / "train" / "normal", dst_root / "test" / "normal", dst_root / "test" / "abnormal")

    def _count(path: Path) -> int:
        return sum(1 for _ in path.glob("*"))

    if all(r.exists() for r in roots) and all(_count(r) > 0 for r in roots):
        print(
            "  [info] dataset already organized: "
            f"train_normal={_count(roots[0])}, test_normal={_count(roots[1])}, test_abnormal={_count(roots[2])}"
        )
        return

    def _copy_split(split_src: Path, dst_normal: Path, dst_abnormal: Path | None, name: str) -> None:
        dst_normal.mkdir(parents=True, exist_ok=True)
        if dst_abnormal is not None:
            dst_abnormal.mkdir(parents=True, exist_ok=True)
        live = attack = 0
        folders = [f for f in split_src.iterdir() if f.is_dir()] if split_src.is_dir() else []
        for folder in folders:
            parts = folder.name.split("_")
            if len(parts) != 4:
                continue
            is_live = parts[3] == "1"  # last field: 1 = live (normal), others = attack
            if dst_abnormal is None and not is_live:
                continue  # training split keeps only live samples (one-class)
            imgs = sorted([p for p in folder.glob("*.jpg")] + [p for p in folder.glob("*.png")])
            if samples_per_folder > 0:
                imgs = imgs[:samples_per_folder]
            target = dst_normal if is_live else dst_abnormal
            for i, img in enumerate(imgs):
                shutil.copy2(img, target / f"{folder.name}_{i:03d}.jpg")
                if is_live:
                    live += 1
                else:
                    attack += 1
        print(f"    [{name}]: live={live}, attack={attack}")

    print("  Organizing OULU-NPU (train/test/dev splits)...")
    for split in ("train", "test", "dev"):
        split_src = src_root / split / split
        if not split_src.is_dir():
            continue
        if split == "train":
            _copy_split(split_src, dst_root / "train" / "normal", None, "train (live only)")
        else:
            _copy_split(split_src, dst_root / split / "normal", dst_root / split / "abnormal", split)


def feature_info(model: Dfm, image_size: tuple[int, int]) -> None:
    """Print the flattened feature dimension for the current configuration.

    The probe must not leave the model in eval mode, otherwise the subsequent
    training step would take the inference (scoring) path and crash.
    """
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size[0], image_size[1])
            feats, shape = model.model.get_features(dummy)
        print(f"      feature map shape  : {list(shape)}")
        print(f"      flattened feature dim : {feats.shape[1]:,}")
    finally:
        if was_training:
            model.train()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train DFM on OULU-NPU face PAD (one-class, live-only training)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- data ---
    parser.add_argument("--data-root", type=Path, default=Path("./datasets/oulu_npu_org"),
                        help="Organized dataset root (train/normal, test/normal, test/abnormal)")
    parser.add_argument("--src-root", type=Path, default=None,
                        help="Raw OULU-NPU root (optional); auto-organizes if organization is missing.")
    parser.add_argument("--samples-per-folder", type=int, default=5,
                        help="Frames sampled per video folder (0 = all).")

    # --- DFM / feature-dimension knobs ---
    parser.add_argument("--backbone", type=str, default="resnet50",
                        help="timm backbone; wider/deeper nets have more channels -> larger features")
    parser.add_argument("--layer", type=str, default="layer3",
                        help="Backbone feature layer used for the embedding")
    parser.add_argument("--pooling-kernel", type=int, default=1,
                        help="Spatial avg-pool kernel. 1 = no pooling (enlarged features); "
                             "higher values shrink the vector but use less RAM.")
    parser.add_argument("--pca-level", type=float, default=0.99,
                        help="PCA variance ratio kept (DFM default 0.97); raise toward 1.0 to keep more components.")
    parser.add_argument("--score-type", type=str, default="fre", choices=["fre", "nll"],
                        help="fre = feature reconstruction error, nll = negative log-likelihood")
    parser.add_argument("--no-pre-trained", action="store_true",
                        help="Disable ImageNet-pretrained backbone weights (random init)")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square input size; larger image -> more spatial patches -> larger feature vector")

    # --- training / hardware ---
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data-loading workers")
    parser.add_argument("--devices", type=int, default=1, help="DFM supports a single device only")
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cpu", "gpu"],
                        help="Training accelerator")
    parser.add_argument("--result-dir", type=Path, default=Path("./results/oulu_npu_dfm"))
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of train set used as validation (from_train)")

    args = parser.parse_args()

    # --- organize (optional) ---
    if args.src_root is not None:
        print_sep("Organizing OULU-NPU")
        organize_oulu_npu(args.data_root, Path(args.src_root), args.samples_per_folder)

    image_size = (args.image_size, args.image_size)

    # --- datamodule ---
    print_sep("Loading dataset")
    pre_processor = PreProcessor(
        transform=Compose([
            Resize(image_size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    datamodule = Folder(
        name="oulu_npu",
        root=str(args.data_root),
        normal_dir="train/normal",
        abnormal_dir="test/abnormal",
        normal_test_dir="test/normal",
        mask_dir=None,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split_mode="from_train",
        val_split_ratio=args.val_ratio,
    )
    print(f"  dataset root   : {args.data_root}")
    print(f"  image size     : {image_size}")

    # --- model (DFM with enlarged features) ---
    print_sep("Building DFM model")
    model = Dfm(
        backbone=args.backbone,
        layer=args.layer,
        pre_trained=not args.no_pre_trained,
        pooling_kernel_size=args.pooling_kernel,
        pca_level=args.pca_level,
        score_type=args.score_type,
        pre_processor=pre_processor,
    )
    print(f"  backbone       : {args.backbone} (layer={args.layer})")
    print(f"  pooling_kernel : {args.pooling_kernel}  (1 = no pooling, enlarged feature dim)")
    print(f"  pca_level      : {args.pca_level}")
    print(f"  score_type     : {args.score_type}")
    print("  [feature dimension]")
    feature_info(model, image_size)

    # --- engine ---
    print_sep("Engine")
    engine = Engine(
        accelerator=args.accelerator,
        devices=1,  # DFM trains on a single device
        default_root_dir=str(args.result_dir),
        gradient_clip_val=0,  # no gradient-based optimization in DFM
    )
    print(f"  accelerator    : {args.accelerator}")
    print("  devices        : 1 (DFM supports a single device)")
    print(f"  result dir     : {args.result_dir}")

    # --- fit + test ---
    print_sep("Training (single epoch: PCA + Gaussian fit)")
    engine.fit(model=model, datamodule=datamodule)

    print_sep("Testing")
    results = engine.test(model=model, datamodule=datamodule)

    print_sep("Results")
    if results:
        for row in results:
            for key, val in row.items():
                print(f"  {key}: {val}")
    print_sep()


if __name__ == "__main__":
    main()