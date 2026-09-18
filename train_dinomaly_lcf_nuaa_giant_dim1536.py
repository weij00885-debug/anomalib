# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train Giant Dinomaly + LCF on NUAA (1536-dimensional features).

This is the NUAA adaptation of ``train_dinomaly_lcf_replay_attack_giant_dim1536.py`` and is in
the same family as ``train_dinomaly_lcf_face.py`` (OULU-NPU + LCF). It retains the two
LCF knobs from the reference script:

    --use-lcf           Enable Layer-Conditional Fusion in :class:`DinomalyModel`
                        (drop-in replacement for ``_fuse_feature = mean()``).
    --lcf-dropout       Dropout applied to LCF attention weights.

See ``src/anomalib/models/image/dinomaly/components/lcf.py`` for the LCF implementation
and ``train_dinomaly_lcf_lcc_dim1024.py`` for the LCC_colors equivalent.

Expected dataset layout (mirrors LCC_colors)::

    <root>/NUAA_mtcnn_colors/<mode>/normal      # live/normal faces -> training
    <root>/NUAA_mtcnn_colors/<mode>/test        # held-out live/normal faces -> test normals
    <root>/NUAA_mtcnn_colors/<mode>/abnormal    # spoof/abnormal images -> test anomalies

The provided ``<mode>/test`` directory is used as the normal test set. All
``normal`` images remain available for one-class training, and all ``abnormal``
images are used as test anomalies.

Examples::

    # Default RGB NUAA run, baseline (no LCF).
    python train_dinomaly_lcf_nuaa_giant_dim1536.py

    # Same but with LCF enabled.
    python train_dinomaly_lcf_nuaa_giant_dim1536.py --use-lcf --lcf-dropout 0.1

    # Run another color representation with LCF.
    python train_dinomaly_lcf_nuaa_giant_dim1536.py --mode hsv --use-lcf

    # Override the Giant defaults with matching Large encoder layers.
    python train_dinomaly_lcf_nuaa_giant_dim1536.py \
        --encoder-name vit_large_patch14_reg4_dinov2 \
        --target-layers 4 6 8 10 12 14 16 18
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torchmetrics import Metric
from torchmetrics.classification import BinaryAccuracy, BinaryConfusionMatrix, BinaryPrecision, BinaryRecall
from torchmetrics.utilities.data import dim_zero_cat
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, Resize

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.metrics import AUROC, AnomalibMetric, Evaluator, F1Score
from anomalib.models import Dinomaly
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor

DEFAULT_ENCODER = "vit_giant_patch14_reg4_dinov2"
DEFAULT_TARGET_LAYERS = [6, 10, 14, 18, 22, 26, 30, 34]
DEFAULT_BOTTLENECK_DROPOUT = 0.2
DEFAULT_DECODER_DEPTH = 8
DEFAULT_MAX_STEPS = 5000
DEFAULT_GRADIENT_CLIP = 0.1
DEFAULT_EARLY_STOP_PATIENCE = 20

# Select the supplied data directory in both native Windows and WSL.  WSL mounts
# the Windows C: drive at /mnt/c, whereas a native Windows Python interpreter
# needs the drive-letter form.
WINDOWS_DATA_ROOT = Path(r"C:\Users\heqi\Desktop\wj\data")
WSL_DATA_ROOT = Path("/mnt/c/Users/heqi/Desktop/wj/data")
DEFAULT_ROOT = str(WSL_DATA_ROOT if WSL_DATA_ROOT.is_dir() else WINDOWS_DATA_ROOT)
DEFAULT_MODE = "rgb"
DATASET_NAME = "NUAA_mtcnn_colors"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_groups(mode: str, n: int) -> list[list[int]]:
    """Build a list-of-lists grouping over n consecutive feature indices.

    Args:
        mode: Grouping strategy name.
        n: Number of available indices (0 .. n-1).

    Returns:
        Group indices, e.g. ``[[0, 1, 2, 3], [4, 5, 6, 7]]``.

    Raises:
        ValueError: If the mode is unknown or n is too small for it.
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
    """Validate that grouping indices are in range and groups are non-empty."""
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
    """Build the Dinomaly pre-processor.

    The image is first resized square and optionally center-cropped. Sizes must
    be multiples of 14 because Dinomaly uses a ViT patch size of 14.
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
    """Half Total Error Rate ``(FPR + FNR) / 2`` at the deployed threshold."""

    def compute(self) -> torch.Tensor:
        """Return HTER from the accumulated binary confusion matrix."""
        tn, fp, fn, tp = super().compute().float().reshape(-1)
        fpr = fp / (fp + tn + 1e-12)
        fnr = fn / (fn + tp + 1e-12)
        return (fpr + fnr) / 2.0


def build_evaluator() -> Evaluator:
    """Build the image-level metric set reported for NUAA face anti-spoofing."""
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

        cum_pos = torch.cat([torch.zeros(1, device=preds.device), torch.cumsum(is_pos, 0)])
        cum_neg = torch.cat([torch.zeros(1, device=preds.device), torch.cumsum(is_neg, 0)])

        candidates = torch.unique(s)
        le = torch.searchsorted(s, candidates, right=True)
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
        description="Train anomalib Dinomaly + LCF on the NUAA dataset (NUAA_mtcnn_colors).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("data")
    data.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Directory holding the NUAA_mtcnn_colors dataset.")
    data.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        choices=["rgb", "hsv", "lab", "ycbcr", "yuv"],
        help="Color-mode subfolder under NUAA_mtcnn_colors.",
    )
    data.add_argument("--train-batch-size", type=int, default=2)
    data.add_argument("--eval-batch-size", type=int, default=2)
    data.add_argument("--num-workers", type=int, default=4)
    data.add_argument("--val-split-ratio", type=float, default=0.2, help="Fraction of test set used as validation.")

    model = parser.add_argument_group("model / feature-extraction knobs")
    model.add_argument("--encoder-name", type=str, default=DEFAULT_ENCODER)
    model.add_argument(
        "--target-layers",
        type=int,
        nargs="+",
        default=None,
        help="DINOv2 block indices to extract. Positions in this list become grouping indices 0..N-1.",
    )
    model.add_argument(
        "--fuse-mode",
        type=str,
        default="half",
        choices=["half", "single", "pairs", "per-layer", "alternating"],
        help="Encoder-side grouping strategy. Decoder mirrors it by default.",
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
        help="Exact decoder grouping. Defaults to mirroring the encoder grouping.",
    )
    model.add_argument("--bottleneck-dropout", type=float, default=DEFAULT_BOTTLENECK_DROPOUT)
    model.add_argument("--decoder-depth", type=int, default=DEFAULT_DECODER_DEPTH)
    model.add_argument("--remove-class-token", action="store_true")
    model.add_argument("--use-context-recentering", action="store_true")
    model.add_argument(
        "--use-lcf",
        action="store_true",
        help="Enable Layer-Conditional Fusion (drop-in replacement for _fuse_feature=mean()).",
    )
    model.add_argument("--lcf-dropout", type=float, default=0.0, help="Dropout on LCF attention weights.")
    model.add_argument("--image-size", type=int, default=392, help="Resize size. Must be a multiple of 14.")
    model.add_argument("--crop-size", type=int, default=None, help="Optional center crop. Must be a multiple of 14.")
    model.add_argument(
        "--model-precision",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Model weights precision. float16 is stored as bfloat16 by Dinomaly.",
    )

    train = parser.add_argument_group("training / engine")
    train.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    train.add_argument("--max-epochs", type=int, default=None, help="Alternative stop criterion.")
    train.add_argument("--gradient-clip-val", type=float, default=DEFAULT_GRADIENT_CLIP)
    train.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE)
    train.add_argument("--accelerator", type=str, default="gpu", choices=["auto", "gpu", "cpu", "xpu"])
    train.add_argument("--devices", type=int, default=1)
    train.add_argument("--results-dir", type=str, default=None, help="Override the default `results/Dinomaly/...` directory.")
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
    if not 0.0 < args.val_split_ratio < 1.0:
        parser.error("--val-split-ratio must be between 0 and 1.")
    return args


def main() -> None:
    """Run the full fit + test pipeline for NUAA_mtcnn_colors."""
    args = parse_args()
    pl.seed_everything(args.seed)

    dataset_dir = Path(args.root) / DATASET_NAME / args.mode
    normal_dir = dataset_dir / "normal"
    normal_test_dir = dataset_dir / "test"
    abnormal_dir = dataset_dir / "abnormal"

    if not normal_dir.is_dir():
        msg = f"Normal training dir not found: {normal_dir}. Point --root at the directory holding NUAA_mtcnn_colors."
        raise FileNotFoundError(msg)
    if not normal_test_dir.is_dir():
        msg = f"Normal test dir not found: {normal_test_dir}. Expected NUAA_mtcnn_colors/<mode>/test."
        raise FileNotFoundError(msg)
    if not abnormal_dir.is_dir():
        msg = f"Abnormal test dir not found: {abnormal_dir}. Expected NUAA_mtcnn_colors/<mode>/abnormal."
        raise FileNotFoundError(msg)

    # ------------------------------------------------------------------
    # 1. Grouping strategy.
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
        decoder_groups = [list(group) for group in encoder_groups]
        validate_groups(decoder_groups, args.decoder_depth, "decoder")

    if len(encoder_groups) != len(decoder_groups):
        msg = (
            f"Encoder ({len(encoder_groups)} groups) and decoder ({len(decoder_groups)} groups) "
            "must have the same number of groups because en[i] is compared with de[i]."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 2. Data.
    # ------------------------------------------------------------------
    datamodule = Folder(
        name=f"{DATASET_NAME}_{args.mode}",
        root=str(dataset_dir),
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_test_dir="test",
        mask_dir=None,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        val_split_mode="from_test",
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 3. Model.
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
        use_lcf=args.use_lcf,
        lcf_dropout=args.lcf_dropout,
        precision=args.model_precision,
        pre_processor=pre_processor,
        post_processor=HTERPostProcessor(),
        evaluator=build_evaluator(),
    )

    # ------------------------------------------------------------------
    # 4. Engine + callbacks.
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

    results_dir = Path(args.results_dir) if args.results_dir else Path("results") / "Dinomaly" / f"{DATASET_NAME}_{args.mode}"

    engine_kwargs = {
        "accelerator": args.accelerator,
        "devices": args.devices,
        "gradient_clip_val": args.gradient_clip_val,
        "callbacks": callbacks,
        "default_root_dir": str(results_dir),
    }
    if args.max_steps > 0:
        engine_kwargs["max_steps"] = args.max_steps
    if args.max_epochs is not None:
        engine_kwargs["max_epochs"] = args.max_epochs

    engine = Engine(**engine_kwargs)

    # ------------------------------------------------------------------
    # 5. Print experiment summary.
    # ------------------------------------------------------------------
    def describe_group(side: str, groups: list[list[int]], labels: list[int]) -> str:
        """Render one side's grouping as human-readable mean-of-blocks."""
        return "; ".join(
            f"{side}[{g_idx}] = mean({', '.join(f'E{labels[i]}' for i in group)})"
            for g_idx, group in enumerate(groups)
        )

    print("=" * 72)
    print("Dinomaly NUAA Giant + LCF training experiment")
    print("=" * 72)
    print(f"  dataset            : {DATASET_NAME}/{args.mode} ({dataset_dir})")
    print(f"  encoder            : {args.encoder_name}")
    print(f"  extracted blocks   : {args.target_layers}")
    print(f"  encoder grouping   : {describe_group('en', encoder_groups, args.target_layers)}")
    print(f"  decoder grouping   : {describe_group('de', decoder_groups, list(range(args.decoder_depth))[::-1])}")
    print(f"  comparison pairs   : en[{len(encoder_groups)}] groups <-> de[{len(decoder_groups)}] groups by index")
    print(f"  bottleneck dropout : {args.bottleneck_dropout}")
    print(f"  decoder depth      : {args.decoder_depth}")
    print(f"  LCF (layer attn)   : enabled={args.use_lcf}, dropout={args.lcf_dropout}")
    print(f"  image / crop       : {args.image_size} / {args.crop_size or 'none'}")
    print(f"  batch (train/eval) : {args.train_batch_size} / {args.eval_batch_size}")
    print("  test split         : provided test (normal) + abnormal directories")
    print(f"  val split          : from_test (ratio {args.val_split_ratio})")
    print(f"  max steps / epochs : {args.max_steps} / {args.max_epochs}")
    print(f"  results dir        : {results_dir}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 6. Fit + test.
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
