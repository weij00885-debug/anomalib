"""Train anomalib Dinomaly on the wj/data rgb face anti-spoofing datasets.

Adapted from train_dinomaly_face.py. Each dataset lives at
<data-root>/<dataset>/<mode>/ with already-labeled {normal, abnormal} folders.
The mixed 	est folder is NOT used (labels are unreliable for some datasets):
test-normal images are carved out of 
ormal via --test-split-ratio, and
abnormal are used as test anomalies.

Easiest knobs:
  --dataset      which dataset       (default CASIA_mtcnn_colors)
  --max-steps    training steps      (default 5000)
  --devices      number of GPUs      (default 1)
  --accelerator  gpu / cpu           (default gpu)

NOTE: you run inside WSL, so a Windows path C:/Users/... is /mnt/c/Users/...

This script is a thin, experiment-friendly wrapper around the anomalib
``Dinomaly`` model + ``Folder`` datamodule.  It exposes the two knobs that
control *what* the decoder is asked to reconstruct and *how* the comparison
groups are formed:

1. ``--target-layers``: which DINOv2 block outputs are collected (the encoder
   "target" features), e.g. ``2 3 4 5 6 7 8 9``.
2. ``--fuse-mode`` / ``--fuse-encoder-json`` / ``--fuse-decoder-json``: how
   those per-layer features are grouped before the cosine-similarity loss /
   anomaly map.  ``en[i]`` is compared with ``de[i]``, so the encoder and
   decoder group counts must match.

The default dataset layout matches the organized OULU-NPU face
anti-spoofing data used in this workspace::

    <root>/
    ├── train/
    │   ├── normal/       # live faces -> training (normal class only)
    │   └── abnormal/     # attack faces (not used for Dinomaly training)
    └── test/
        ├── normal/       # live faces for evaluation
        └── abnormal/     # attack faces for evaluation

Typical usage::

    # Default OULU-NPU run (DINOv2-giant = maximum feature dim).
    python train_dinomaly_face.py

    # Cheaper encoder + smaller batch on a GPU-limited machine
    python train_dinomaly_face.py --encoder-name vit_small_patch14_reg4_dinov2 \
        --train-batch-size 8

    # Ablation: extract 8 deeper blocks, and compare all 8 layers at once
    python train_dinomaly_face.py --target-layers 4 5 6 7 8 9 10 11 \
        --fuse-mode single

    # Ablation: per-layer (strict) comparison instead of the official 2 groups
    python train_dinomaly_face.py --fuse-mode per-layer

    # Fully custom grouping (groups are positions inside the extracted list)
    python train_dinomaly_face.py \
        --fuse-encoder-json '[[0, 1], [2, 3], [4, 5], [6, 7]]'

The model/engine/datamodule arguments mirror the values already used by
``train_casia_fasd.py`` and the official ``dinomaly.yaml`` config where
possible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall
from torchmetrics.utilities.data import dim_zero_cat
from lightning.pytorch.callbacks import EarlyStopping
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.pre_processing import PreProcessor
from anomalib.post_processing import PostProcessor
from anomalib.metrics import AUROC, AnomalibMetric, Evaluator, F1Score

# Official Dinomaly hyperparameters used as defaults (see examples/configs/model/dinomaly.yaml).
DEFAULT_ENCODER = "vit_giant_patch14_reg4_dinov2"
DEFAULT_TARGET_LAYERS = [6, 10, 14, 18, 22, 26, 30, 34]
DEFAULT_BOTTLENECK_DROPOUT = 0.2
DEFAULT_DECODER_DEPTH = 8
DEFAULT_MAX_STEPS = 5000
DEFAULT_GRADIENT_CLIP = 0.1
DEFAULT_EARLY_STOP_PATIENCE = 20

# Default face dataset (organized OULU-NPU) used in this workspace.
DEFAULT_ROOT = "/mnt/c/Users/heqi/Desktop/wj/data"  # WSL path of C:\Users\heqi\Desktop\wj\data
DEFAULT_DATASET = "CASIA_mtcnn_colors"
DEFAULT_MODE = "rgb"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_groups(mode: str, n: int) -> list[list[int]]:
    """Build a list-of-lists grouping over `
`` consecutive feature indices.

    The modes cover the interesting comparison strategies for Dinomaly:

    - ``half``: official default, two contiguous halves ([[0..3], [4..7]] for
      n=8) -> loose low-level vs high-level supervision.
    - ``single``: fuse every layer into one group -> loosest constraint.
    - ``pairs``: adjacent pairs -> intermediate granularity.
    - ``per-layer``: one group per layer -> strictest (layer-wise) comparison.
    - ``alternating``: odd/even layers -> mixes abstraction levels in each group.

    Args:
        mode (str): Grouping strategy name.
        n (int): Number of available indices (0 .. n-1).

    Returns:
        list[list[int]]: Group indices, e.g. ``[[0, 1, 2, 3], [4, 5, 6, 7]]``.

    Raises:
        ValueError: If the mode is unknown or `
`` is too small for it.
    """
    if n < 1:
        msg = f"Cannot build groups over {n} layers."
        raise ValueError(msg)

    if mode == "single":
        return [list(range(n))]
    if mode == "half":
        if n < 2:
            msg = f"'half' grouping needs at least 2 layers, got {n}."
            raise ValueError(msg)
        half = n // 2
        return [list(range(0, half)), list(range(half, n))]
    if mode == "pairs":
        return [list(range(i, min(i + 2, n))) for i in range(0, n, 2)]
    if mode == "per-layer":
        return [[i] for i in range(n)]
    if mode == "alternating":
        if n < 2:
            msg = f"'alternating' grouping needs at least 2 layers, got {n}."
            raise ValueError(msg)
        return [list(range(0, n, 2)), list(range(1, n, 2))]
    msg = f"Unknown grouping mode: {mode}"
    raise ValueError(msg)


def validate_groups(groups: list[list[int]], n_layers: int, label: str) -> None:
    """Validate that every group index lies in the valid range and groups are non-empty.

    Args:
        groups (list[list[int]]): Group definitions.
        n_layers (int): Number of available features (valid indices are 0..n_layers-1).
        label (str): Side being validated, for error messages ("encoder"/"decoder").

    Raises:
        ValueError: If the grouping is not usable with `
_layers`` features.
    """
    if not groups:
        msg = f"{label} grouping must contain at least one group."
        raise ValueError(msg)
    for group in groups:
        if not group:
            msg = f"{label} grouping contains an empty group: {groups}"
            raise ValueError(msg)
        for index in group:
            if not 0 <= index < n_layers:
                msg = (
                    f"{label} group index {index} is out of range for {n_layers} "
                    f"features. Check --target-layers / --decoder-depth / grouping."
                )
                raise ValueError(msg)


def build_pre_processor(image_size: int, crop_size: int | None) -> PreProcessor:
    """Build the Dinomaly pre-processor for face images.

    By default the image is resized directly (no center crop) because a crop
    may remove informative face borders.  Pass ``--crop-size`` to enable the
    official 448 -> 392 pipeline instead.

    Args:
        image_size (int): Square resize size (must be a multiple of 14 for the
            ViT patch size).
        crop_size (int | None): Optional center-crop size. ``None`` disables crop.

    Returns:
        PreProcessor: Anomalib pre-processor holding the transform chain.
    """
    if image_size % 14 != 0:
        msg = f"--image-size must be a multiple of 14 (patch size), got {image_size}."
        raise ValueError(msg)
    if crop_size is not None and crop_size % 14 != 0:
        msg = f"--crop-size must be a multiple of 14, got {crop_size}."
        raise ValueError(msg)
    if crop_size is not None and crop_size > image_size:
        msg = f"--crop-size ({crop_size}) cannot exceed --image-size ({image_size})."
        raise ValueError(msg)

    transforms = [Resize((image_size, image_size))]
    if crop_size is not None:
        transforms.append(CenterCrop(crop_size))
    transforms.append(Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return PreProcessor(transform=Compose(transforms))


class Accuracy(AnomalibMetric, BinaryAccuracy):
    """Image-level accuracy over ``pred_label`` vs ``gt_label``."""


class Precision(AnomalibMetric, BinaryPrecision):
    """Image-level precision over ``pred_label`` vs ``gt_label``."""


class Recall(AnomalibMetric, BinaryRecall):
    """Image-level recall over ``pred_label`` vs ``gt_label``."""


class HTER(AnomalibMetric, BinaryConfusionMatrix):
    """Half Total Error Rate ``(FPR + FNR) / 2`` at the deployed threshold.

    All classifier-style metrics (Accuracy/F1/Precision/Recall/HTER) share the
    same decision threshold: the F1-adaptive threshold learned by the default
    ``PostProcessor`` from the training/validation split.
    """

    def compute(self) -> torch.Tensor:
        """Return HTER from the accumulated binary confusion matrix."""
        tn, fp, fn, tp = super().compute().float().reshape(-1)
        fpr = fp / (fp + tn + 1e-12)
        fnr = fn / (fn + tp + 1e-12)
        return (fpr + fnr) / 2.0


def build_evaluator() -> Evaluator:
    """Build the image-level metric set reported for face PAD.

    Validation metrics only contain ``image_AUROC`` because ``pred_label`` is
    produced by post-processing only during testing.

    Returns:
        Evaluator: Anomalib evaluator carrying the image-level metrics.
    """
    val_metrics = [AUROC(fields=["pred_score", "gt_label"], prefix="image_")]
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
        Accuracy(fields=["pred_label", "gt_label"], prefix="image_"),
        F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
        Precision(fields=["pred_label", "gt_label"], prefix="image_"),
        Recall(fields=["pred_label", "gt_label"], prefix="image_"),
        HTER(fields=["pred_label", "gt_label"], prefix="image_"),
    ]
    return Evaluator(val_metrics=val_metrics, test_metrics=test_metrics)

class _HTERAdaptiveThreshold(Metric):
    """Adaptive threshold that minimizes HTER = (FPR + FNR) / 2 on validation data."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds if preds.ndim == 1 else preds.flatten()
        target = target if target.ndim == 1 else target.flatten()
        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> torch.Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target).long()
        n_pos = (target == 1).sum().item()
        n_neg = (target == 0).sum().item()
        if n_pos == 0:
            return preds.max().detach()
        if n_neg == 0:
            return preds.min().detach()

        order = torch.argsort(preds)
        s = preds[order]
        t = target[order]
        is_pos = (t == 1).to(preds.dtype)
        is_neg = (t == 0).to(preds.dtype)
        # cum_*[i] = count among the first i (sorted) samples
        cum_pos = torch.cat([torch.zeros(1, device=preds.device), torch.cumsum(is_pos, 0)])
        cum_neg = torch.cat([torch.zeros(1, device=preds.device), torch.cumsum(is_neg, 0)])

        candidates = torch.unique(s)
        le = torch.searchsorted(s, candidates, right=True)  # number of scores <= candidate
        pos_le = cum_pos[le].float()
        neg_le = cum_neg[le].float()
        fpr = (n_neg - neg_le) / n_neg
        fnr = pos_le / n_pos
        hter = (fpr + fnr) / 2.0
        best = hter == hter.min()
        arg = best.nonzero().squeeze(1).max().item()
        return candidates[arg].detach()


class HTERAdaptiveThreshold(AnomalibMetric, _HTERAdaptiveThreshold):  # type: ignore[misc]
    """AnomalibMetric wrapper for the HTER-minimizing threshold."""


class HTERPostProcessor(PostProcessor):
    """PostProcessor that selects the threshold minimizing HTER on validation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_threshold_metric = HTERAdaptiveThreshold(fields=["pred_score", "gt_label"], strict=False)

def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train anomalib Dinomaly on a face (anti-spoofing) dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data (wj/data/<dataset>/<mode>)")
    data.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Directory holding the <dataset> folders.")
    data.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=["CASIA_mtcnn_colors", "LCC_colors", "NUAA_mtcnn_colors", "REPLAY_mtcnn_colors"],
        help="Which dataset to train on.",
    )
    data.add_argument("--mode", type=str, default=DEFAULT_MODE, help="Color-mode subfolder (rgb/hsv/lab/ycbcr/yuv).")
    data.add_argument("--train-batch-size", type=int, default=2)
    data.add_argument("--eval-batch-size", type=int, default=2)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument(
        "--test-split-ratio",
        type=float,
        default=0.2,
        help="Fraction of normal images carved out as test-normal (test folder is not used; labels unreliable).",
    )
    data.add_argument("--val-split-ratio", type=float, default=0.2, help="Fraction of test set used as validation.")

    model = parser.add_argument_group("model / feature-extraction knobs")
    model.add_argument("--encoder-name", type=str, default=DEFAULT_ENCODER)
    model.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="DINOv2 block indices to extract (positions in this list become grouping indices 0..N-1).",
    )
    model.add_argument(
        "--fuse-mode",
        type=str,
        default="half",
        choices=["half", "single", "pairs", "per-layer", "alternating"],
        help="Encoder-side grouping strategy applied before comparison (decoder mirrors it by default).",
    )
    model.add_argument(
        "--fuse-encoder-json",
        type=str,
        default=None,
        help="Exact encoder grouping, e.g. '[[0, 1, 2, 3], [4, 5, 6, 7]]'. Overrides --fuse-mode.",
    )
    model.add_argument(
        "--fuse-decoder-json",
        type=str,
        default=None,
        help="Exact decoder grouping. Defaults to mirroring the encoder grouping (official behaviour).",
    )
    model.add_argument("--bottleneck-dropout", type=float, default=DEFAULT_BOTTLENECK_DROPOUT)
    model.add_argument("--decoder-depth", type=int, default=DEFAULT_DECODER_DEPTH)
    model.add_argument("--remove-class-token", action="store_true")
    model.add_argument("--use-context-recentering", action="store_true")
    model.add_argument("--image-size", type=int, default=392, help="Resize size (multiple of 14).")
    model.add_argument("--crop-size", type=int, default=None, help="Optional center crop (multiple of 14).")
    model.add_argument(
        "--model-precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Model weights precision (float16 is stored as bfloat16 by Dinomaly).",
    )

    train = parser.add_argument_group("training / engine")
    train.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    train.add_argument("--max-epochs", type=int, default=None, help="Alternative stop criterion.")
    train.add_argument("--gradient-clip-val", type=float, default=DEFAULT_GRADIENT_CLIP)
    train.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    train.add_argument("--accelerator", type=str, default="gpu", choices=["auto", "gpu", "cpu", "xpu"])
    train.add_argument("--devices", type=int, default=1)
    train.add_argument("--results-dir", type=str, default=None)
    train.add_argument("--resume-ckpt", type=str, default=None, help="Path to a checkpoint to resume from.")
    train.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.target_layers is None:
        args.target_layers = list(DEFAULT_TARGET_LAYERS)

    if args.remove_class_token and args.use_context_recentering:
        parser.error("--remove-class-token and --use-context-recentering are mutually exclusive.")
    if len(set(args.target_layers)) != len(args.target_layers):
        parser.error("--target-layers must not contain duplicate block indices.")
    if args.decoder_depth <= 1:
        parser.error("--decoder-depth must be greater than 1.")
    if args.max_epochs is None and args.max_steps <= 0:
        parser.error("Provide either --max-steps (>0) or --max-epochs (>0).")
    return args


def main() -> None:
    """Run the full fit + test pipeline for the configured experiment."""
    args = parse_args()
    pl.seed_everything(args.seed)

    root = Path(args.root)
    dataset_dir = root / args.dataset / args.mode
    normal_train = dataset_dir / "normal"
    abnormal_dir = dataset_dir / "abnormal"
    if not normal_train.is_dir():
        msg = f"Normal training dir not found: {normal_train}. Check --root / --dataset / --mode."
        raise FileNotFoundError(msg)
    if not abnormal_dir.is_dir():
        msg = f"Abnormal dir not found: {abnormal_dir}. Anomaly evaluation needs abnormal images."
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # 1. Grouping strategy (comparison-side knobs)
    # ------------------------------------------------------------------
    n_target = len(args.target_layers)
    if args.fuse_encoder_json is not None:
        encoder_groups = json.loads(args.fuse_encoder_json)
    else:
        encoder_groups = build_groups(args.fuse_mode, n_target)
    validate_groups(encoder_groups, n_target, "encoder")

    if args.fuse_decoder_json is not None:
        decoder_groups = json.loads(args.fuse_decoder_json)
        validate_groups(decoder_groups, args.decoder_depth, "decoder")
    else:
        # Official Dinomaly uses the same grouping structure on both sides.
        decoder_groups = [list(group) for group in encoder_groups]
        validate_groups(decoder_groups, args.decoder_depth, "decoder")

    if len(encoder_groups) != len(decoder_groups):
        msg = (
            f"Encoder ({len(encoder_groups)} groups) and decoder ({len(decoder_groups)} groups) "
            "must have the same number of groups because en[i] is compared with de[i]."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    datamodule = Folder(
        name=f"{args.dataset}_{args.mode}",
        root=str(dataset_dir),
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_test_dir=None,
        mask_dir=None,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        test_split_ratio=args.test_split_ratio,
        val_split_mode="from_test",
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    pre_processor = build_pre_processor(args.image_size, args.crop_size)
    model = Dinomaly(
        encoder_name=args.encoder_name,
        bottleneck_dropout=args.bottleneck_dropout,
        decoder_depth=args.decoder_depth,
        target_layers=args.target_layers,
        fuse_layer_encoder=encoder_groups,
        fuse_layer_decoder=decoder_groups,
        remove_class_token=args.remove_class_token,
        use_context_recentering=args.use_context_recentering,
        precision=args.model_precision,
        pre_processor=pre_processor,
        post_processor=HTERPostProcessor(),
        evaluator=build_evaluator(),
    )

    # ------------------------------------------------------------------
    # 4. Engine + callbacks
    # ------------------------------------------------------------------
    callbacks = []
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="image_AUROC",
                mode="max",
                patience=args.early_stop_patience,
            )
        )

    engine_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "gradient_clip_val": args.gradient_clip_val,
        "callbacks": callbacks,
    }
    if args.max_steps > 0:
        engine_kwargs["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        engine_kwargs["max_epochs"] = args.max_epochs
    if args.results_dir is not None:
        engine_kwargs["default_root_dir"] = args.results_dir

    engine = Engine(**engine_kwargs)

    # ------------------------------------------------------------------
    # 5. Print an unambiguous experiment summary
    # ------------------------------------------------------------------
    def describe_group(side: str, groups: list[list[int]], labels: list[int]) -> str:
        """Render one side's grouping as human-readable mean-of-blocks."""
        return "; ".join(
            f"{side}[{g_idx}] = mean({', '.join(f'E{labels[i]}' for i in group)})"
            for g_idx, group in enumerate(groups)
        )

    print("=" * 72)
    print("Dinomaly face-training experiment")
    print("=" * 72)
    print(f"  encoder            : {args.encoder_name}")
    print(f"  extracted blocks   : {args.target_layers}")
    print(f"  encoder grouping   : {describe_group('en', encoder_groups, args.target_layers)}")
    print(f"  decoder grouping   : {describe_group('de', decoder_groups, list(range(args.decoder_depth))[::-1])}")
    print(f"  comparison pairs   : en[{len(encoder_groups)}] groups <-> de[{len(decoder_groups)}] groups by index")
    print(f"  bottleneck dropout : {args.bottleneck_dropout}")
    print(f"  decoder depth      : {args.decoder_depth}")
    print(f"  image / crop       : {args.image_size} / {args.crop_size or 'none'}")
    print(f"  dataset            : {args.dataset}/{args.mode} ({dataset_dir})")
    print(f"  batch (train/eval) : {args.train_batch_size} / {args.eval_batch_size}")
    print(f"  val split          : from_test (ratio {args.val_split_ratio})")
    print(f"  max steps / epochs : {args.max_steps} / {args.max_epochs}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 6. Fit + test
    # ------------------------------------------------------------------
    engine.fit(model=model, datamodule=datamodule, ckpt_path=args.resume_ckpt)

    if engine.trainer.log_dir is not None:
        print(f"\nLogs/checkpoints written to: {engine.trainer.log_dir}")

    print("\n" + "=" * 72)
    print("Evaluating on the test set ...")
    print("=" * 72)
    results = engine.test(model=model, datamodule=datamodule)

    print("\n" + "=" * 72)
    print("Test results")
    print("=" * 72)
    for idx, result in enumerate(results, start=1):
        print(f"\n  Test set {idx}:")
        for key, value in result.items():
            print(f"    {key}: {value}")
    print("\nDone.")


if __name__ == "__main__":
    main()
