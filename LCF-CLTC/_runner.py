# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared legacy-protocol runner for LCF + detached CLTC face experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize, Resize
from tqdm.auto import tqdm

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.models.image.dinomaly.torch_model import DINO_ARCHITECTURES
from anomalib.pre_processing import PreProcessor

if TYPE_CHECKING:
    from anomalib.data.datasets import FolderDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset-specific defaults for a thin executable wrapper."""

    name: str
    slug: str
    encoder_name: str
    dedicated_normal_test: bool
    cltc_loss_weight: float = 0.1
    cltc_score_weight: float = 1.0


def build_groups(count: int) -> list[list[int]]:
    """Build the standard two-half Dinomaly grouping."""
    if count < 2:
        msg = "At least two target layers are required."
        raise ValueError(msg)
    middle = count // 2
    return [list(range(middle)), list(range(middle, count))]


def build_preprocessor(image_size: int) -> PreProcessor:
    """Build square ImageNet preprocessing for DINOv2."""
    if image_size <= 0 or image_size % 14:
        msg = f"Image size must be a positive multiple of 14, got {image_size}."
        raise ValueError(msg)
    return PreProcessor(
        transform=Compose([
            Resize((image_size, image_size)),
            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    )


def architecture_layers(encoder_name: str) -> list[int]:
    """Return the repository defaults for a named DINOv2 architecture."""
    for name, config in DINO_ARCHITECTURES.items():
        if name in encoder_name:
            return list(config["target_layers"])
    msg = f"Unknown encoder architecture: {encoder_name}."
    raise ValueError(msg)


def check_disjoint(datasets: dict[str, FolderDataset]) -> None:
    """Reject exact image-path overlap among train, validation and test."""
    seen: dict[Path, str] = {}
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            msg = f"The {split} subset contains no images."
            raise ValueError(msg)
        for value in dataset.samples.image_path:
            path = Path(value).resolve()
            if path in seen:
                msg = f"Image overlaps {seen[path]} and {split}: {path}"
                raise ValueError(msg)
            seen[path] = split


def write_manifest(path: Path, dataset: FolderDataset) -> str:
    """Write a stable sample manifest and return its SHA256."""
    text = dataset.samples[["image_path", "label_index"]].sort_values("image_path").to_csv(index=False)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


@torch.inference_mode()
def collect_scores(
    model: Dinomaly,
    dataset: FolderDataset,
    batch_size: int,
    num_workers: int,
    label: str,
) -> pd.DataFrame:
    """Collect internal scores with the same preprocessing as the old scripts."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    rows: list[dict[str, str | int | float]] = []
    for batch in tqdm(loader, desc=label):
        images = model.pre_processor(batch.image.to(model.device))
        branches = model.model.predict_branch_scores(images)
        values = {name: scores.float().cpu().tolist() for name, scores in branches.items()}
        for index, image_path in enumerate(batch.image_path):
            rows.append({
                "image_path": image_path,
                "label": int(batch.gt_label[index]),
                **{name: branch[index] for name, branch in values.items()},
            })
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.drop(columns=["image_path", "label"]).to_numpy()).all():
        msg = f"Non-finite scores encountered in {label}."
        raise ValueError(msg)
    return frame


def hter_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Match the old face scripts' validation threshold that minimizes HTER."""
    candidates = np.unique(scores)
    best_threshold = float(candidates[0])
    best_hter = float("inf")
    for threshold in candidates:
        prediction = scores > threshold
        false_positive_rate = prediction[labels == 0].mean()
        false_negative_rate = (~prediction[labels == 1]).mean()
        value = (false_positive_rate + false_negative_rate) / 2
        if value <= best_hter:
            best_hter = float(value)
            best_threshold = float(threshold)
    return best_threshold


def summarize_fused(test: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
    """Report only final LCF+CLTC using the legacy face-script metrics."""
    labels = test.label.to_numpy()
    validation_labels = validation.label.to_numpy()
    if set(labels) != {0, 1} or set(validation_labels) != {0, 1}:
        msg = "Validation and test subsets must both contain real and attack images."
        raise ValueError(msg)
    scores = test.fused.to_numpy()
    threshold = hter_threshold(validation_labels, validation.fused.to_numpy())
    prediction = scores > threshold
    false_positive_rate = float(prediction[labels == 0].mean())
    false_negative_rate = float((~prediction[labels == 1]).mean())
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_Accuracy": float(accuracy_score(labels, prediction)),
        "image_F1Score": float(f1_score(labels, prediction, zero_division=0)),
        "image_Precision": float(precision_score(labels, prediction, zero_division=0)),
        "image_Recall": float(recall_score(labels, prediction, zero_division=0)),
        "image_HTER": (false_positive_rate + false_negative_rate) / 2,
    }


def parse_args(spec: DatasetSpec) -> argparse.Namespace:
    """Parse shared arguments with dataset-specific defaults."""
    parser = argparse.ArgumentParser(
        description=f"Train LCF + detached CLTC on {spec.name} with its legacy split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/c/Users/heqi/Desktop/wj/data"))
    parser.add_argument("--mode", choices=["rgb", "hsv", "lab", "ycbcr", "yuv"], default="rgb")
    parser.add_argument("--encoder-name", default=spec.encoder_name)
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--model-precision", choices=["float32", "float16"], default="float16")
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split-ratio", type=float, default=0.2)
    parser.add_argument("--normal-test-ratio", type=float, default=0.2)
    parser.add_argument("--cltc-hidden-dim", type=int, default=128)
    parser.add_argument("--cltc-loss-weight", type=float, default=spec.cltc_loss_weight)
    parser.add_argument("--cltc-score-weight", type=float, default=spec.cltc_score_weight)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "xpu"], default="gpu")
    parser.add_argument("--results-dir", type=Path, default=Path(f"results/{spec.slug}_lcf_cltc_legacy_s42"))
    args = parser.parse_args()
    positive = (args.image_size, args.train_batch_size, args.eval_batch_size, args.max_steps, args.cltc_hidden_dim)
    if min(positive) <= 0 or args.num_workers < 0:
        parser.error("Image size, batch sizes, steps and hidden dimension must be positive; workers nonnegative.")
    if not 0 < args.val_split_ratio < 1 or not 0 < args.normal_test_ratio < 1:
        parser.error("Split ratios must be between zero and one.")
    return args


def run(spec: DatasetSpec) -> None:
    """Train and evaluate final LCF+CLTC using the dataset's old split."""
    args = parse_args(spec)
    pl.seed_everything(args.seed, workers=True)
    dataset_dir = args.data_root / spec.name / args.mode
    required = [dataset_dir / "normal", dataset_dir / "abnormal"]
    if spec.dedicated_normal_test:
        required.append(dataset_dir / "test")
    for directory in required:
        if not directory.is_dir():
            msg = f"Missing required directory: {directory}"
            raise FileNotFoundError(msg)

    data = Folder(
        name=f"{spec.name}_{args.mode}",
        root=dataset_dir,
        normal_dir="normal",
        abnormal_dir="abnormal",
        normal_test_dir="test" if spec.dedicated_normal_test else None,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        test_split_ratio=args.normal_test_ratio,
        val_split_mode="from_test",
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
    )
    data.setup()
    subsets = {"train": data.train_data, "validation": data.val_data, "test": data.test_data}
    check_disjoint(subsets)
    for split in ("validation", "test"):
        if set(subsets[split].samples.label_index) != {0, 1}:
            msg = f"The {split} subset must contain both real and attack images."
            raise ValueError(msg)

    args.results_dir.mkdir(parents=True, exist_ok=False)
    protocol = (
        "legacy_dedicated_normal_test_from_test_val"
        if spec.dedicated_normal_test
        else "legacy_normal_holdout_from_test_val"
    )
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        dataset=spec.name,
        protocol=protocol,
        use_lcf=True,
        use_cltc=True,
        cltc_detach=True,
        score_fusion="after_pooling",
    )
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    hashes = {
        name: write_manifest(args.results_dir / f"{name}_manifest.csv", dataset)
        for name, dataset in subsets.items()
    }

    target_layers = architecture_layers(args.encoder_name)
    groups = build_groups(len(target_layers))
    preprocessor = build_preprocessor(args.image_size)
    model = Dinomaly(
        encoder_name=args.encoder_name,
        target_layers=target_layers,
        decoder_depth=8,
        bottleneck_dropout=0.2,
        fuse_layer_encoder=groups,
        fuse_layer_decoder=groups,
        use_lcf=True,
        use_cltc=True,
        cltc_hidden_dim=args.cltc_hidden_dim,
        cltc_loss_weight=args.cltc_loss_weight,
        cltc_score_weight=args.cltc_score_weight,
        precision=args.model_precision,
        pre_processor=preprocessor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    engine = Engine(
        accelerator=args.accelerator,
        devices=1,
        max_steps=args.max_steps,
        max_epochs=-1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        gradient_clip_val=0.1,
        default_root_dir=args.results_dir,
    )
    print(f"Dataset={spec.name}/{args.mode}; encoder={args.encoder_name}")
    print(f"LCF=True; CLTC=True; detach=True; alpha={args.cltc_score_weight}")
    print(f"Legacy split train/validation/test={len(data.train_data)}/{len(data.val_data)}/{len(data.test_data)}")
    engine.fit(model=model, datamodule=data)
    model.to(engine.trainer.strategy.root_device).eval()

    validation = collect_scores(
        model,
        data.val_data,
        args.eval_batch_size,
        args.num_workers,
        "Scoring legacy validation split",
    )
    head = model.model.trajectory_head
    normal_validation = validation.loc[validation.label == 0, ["reconstruction", "trajectory"]]
    head.fit_score_scales(torch.tensor(normal_validation.to_numpy()))
    reconstruction_scale, trajectory_scale = head.score_scales.cpu().tolist()
    validation["fused"] = (
        validation.reconstruction / reconstruction_scale
        + args.cltc_score_weight * validation.trajectory / trajectory_scale
    )
    validation[["image_path", "label", "fused"]].to_csv(
        args.results_dir / "validation_scores.csv",
        index=False,
    )
    engine.trainer.save_checkpoint(args.results_dir / "calibrated_last.ckpt")

    test = collect_scores(
        model,
        data.test_data,
        args.eval_batch_size,
        args.num_workers,
        "Testing final LCF+CLTC score",
    )
    test[["image_path", "label", "fused"]].to_csv(args.results_dir / "test_scores.csv", index=False)
    report = {
        "metrics": summarize_fused(test, validation),
        "manifest_sha256": hashes,
        "score_scales": head.score_scales.cpu().tolist(),
        "protocol": protocol,
        "note": "Final LCF+CLTC fused score only; split and metrics match the dataset's old LCF script.",
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Outputs: {args.results_dir.resolve()}")
