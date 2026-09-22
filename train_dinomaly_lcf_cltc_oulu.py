# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Train detached CLTC on OULU-NPU using the original LCF script's data split.

As in ``train_dinomaly_lcf_face.py``, 20% of the folder test set is sampled as
validation data and the remaining 80% is used for final testing. This is a
frame-level folder experiment, not an implementation of official OULU protocols.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from train_dinomaly_lcf_face import DEFAULT_ENCODER, DEFAULT_ROOT, build_groups, build_pre_processor, validate_groups

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Dinomaly
from anomalib.models.image.dinomaly.torch_model import DINO_ARCHITECTURES

if TYPE_CHECKING:
    from anomalib.data.datasets import FolderDataset


def parse_args() -> argparse.Namespace:
    """Parse a fixed-step, real-only training experiment."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--normal-dir", default="train/normal")
    parser.add_argument("--normal-test-dir", default="test/normal")
    parser.add_argument("--abnormal-dir", default="test/abnormal")
    parser.add_argument(
        "--val-split-ratio",
        type=float,
        default=0.2,
        help="Fraction sampled label-aware from test, matching the old LCF script.",
    )
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER)
    parser.add_argument("--target-layers", type=int, nargs="+", default=None)
    parser.add_argument("--decoder-depth", type=int, default=8)
    parser.add_argument("--fuse-mode", choices=["half", "single", "pairs", "per-layer", "alternating"], default="half")
    parser.add_argument("--bottleneck-dropout", type=float, default=0.2)
    parser.add_argument("--lcf-dropout", type=float, default=0.0)
    parser.add_argument("--cltc-hidden-dim", type=int, default=128)
    parser.add_argument("--cltc-loss-weight", type=float, default=0.1)
    parser.add_argument("--cltc-score-weight", type=float, default=1.0, help="Fix before evaluating test data.")
    parser.add_argument("--image-size", type=int, default=392)
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--remove-class-token", action="store_true")
    parser.add_argument("--use-context-recentering", action="store_true")
    parser.add_argument(
        "--model-precision",
        choices=["float32", "float16"],
        default="float32",
        help="float16 uses BF16 base weights; the CLTC head remains FP32.",
    )
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=0.1,
        help="Match the old LCF script's gradient clipping.",
    )
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "xpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("results/oulu_lcf_cltc"))
    parser.add_argument(
        "--resume-ckpt",
        type=Path,
        default=None,
        help="Resume the same CLTC setup; not an old LCF checkpoint.",
    )
    args = parser.parse_args()
    if min(args.max_steps, args.train_batch_size, args.eval_batch_size, args.cltc_hidden_dim) <= 0:
        parser.error("Steps, batch sizes and hidden dimension must be positive.")
    if args.gradient_clip_val < 0:
        parser.error("--gradient-clip-val must be nonnegative.")
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative.")
    if not 0 < args.val_split_ratio < 1:
        parser.error("--val-split-ratio must be between zero and one.")
    if args.remove_class_token and args.use_context_recentering:
        parser.error("Token removal and context recentering are mutually exclusive.")
    if args.target_layers is None:
        for name, config in DINO_ARCHITECTURES.items():
            if name in args.encoder_name:
                args.target_layers = list(config["target_layers"])
                break
        else:
            parser.error("Unknown encoder architecture; supply --target-layers explicitly.")
    if len(args.target_layers) < 2 or args.target_layers != sorted(set(args.target_layers)):
        parser.error("Use at least two unique target layers in increasing order.")
    return args


def check_disjoint(datasets: dict[str, FolderDataset]) -> None:
    """Reject overlapping image paths between train, validation and test sets."""
    seen: dict[Path, str] = {}
    for split, dataset in datasets.items():
        if len(dataset) == 0:
            msg = f"The {split} dataset contains no images."
            raise ValueError(msg)
        for image_path in dataset.samples.image_path:
            path = Path(image_path).resolve()
            if path in seen:
                msg = f"Image overlaps {seen[path]} and {split}: {path}"
                raise ValueError(msg)
            seen[path] = split


@torch.inference_mode()
def collect_scores(model: Dinomaly, dataset: FolderDataset, args: argparse.Namespace, label: str) -> pd.DataFrame:
    """Collect internal scores using the same preprocessing as the old runner."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    rows = []
    for batch in tqdm(loader, desc=label):
        images = model.pre_processor(batch.image.to(model.device))
        branches = model.model.predict_branch_scores(images)
        values = {name: scores.float().cpu().tolist() for name, scores in branches.items()}
        for index, path in enumerate(batch.image_path):
            rows.append({
                "image_path": path,
                "label": int(batch.gt_label[index]),
                **{name: scores[index] for name, scores in values.items()},
            })
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame.drop(columns=["image_path", "label"]).to_numpy()).all():
        msg = f"Non-finite scores encountered in {label}."
        raise ValueError(msg)
    return frame


def summarize_scores(test: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Report only the final LCF+CLTC score with the old script's metrics.

    The decision threshold maximizes F1 on the validation subset, matching the
    default anomalib post-processor used by ``train_dinomaly_lcf_face.py``.
    Attack is label 1, so attack acceptance is the false-negative rate.
    """
    labels = test.label.to_numpy()
    if set(labels) != {0, 1}:
        msg = "Test evaluation requires both real (0) and attack (1) images."
        raise ValueError(msg)
    validation_labels = validation.label.to_numpy()
    if set(validation_labels) != {0, 1}:
        msg = "Validation threshold estimation requires both real (0) and attack (1) images."
        raise ValueError(msg)
    scores = test.fused.to_numpy()
    val_scores = validation.fused.to_numpy()
    val_precision, val_recall, thresholds = precision_recall_curve(validation_labels, val_scores)
    val_f1 = 2 * val_precision[:-1] * val_recall[:-1] / (val_precision[:-1] + val_recall[:-1] + 1e-10)
    threshold = float(thresholds[int(np.argmax(val_f1))])
    attack_prediction = scores > threshold
    false_negative_rate = float((~attack_prediction[labels == 1]).mean())
    false_positive_rate = float(attack_prediction[labels == 0].mean())
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_Accuracy": float(accuracy_score(labels, attack_prediction)),
        "image_F1Score": float(f1_score(labels, attack_prediction, zero_division=0)),
        "image_Precision": float(precision_score(labels, attack_prediction, zero_division=0)),
        "image_Recall": float(recall_score(labels, attack_prediction, zero_division=0)),
        "image_HTER": (false_positive_rate + false_negative_rate) / 2,
    }


def main() -> None:
    """Fit a detached head and evaluate with the original LCF folder split."""
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    groups = build_groups(args.fuse_mode, len(args.target_layers))
    validate_groups(groups, args.decoder_depth, "decoder")
    if args.decoder_depth <= 1:
        msg = "Decoder depth must be greater than one."
        raise ValueError(msg)
    for directory in (args.normal_dir, args.normal_test_dir, args.abnormal_dir):
        if not (args.root / directory).is_dir():
            msg = f"Missing directory: {args.root / directory}"
            raise FileNotFoundError(msg)
    pre_processor = build_pre_processor(args.image_size, args.crop_size)
    data = Folder(
        name="oulu_npu",
        root=args.root,
        normal_dir=args.normal_dir,
        normal_test_dir=args.normal_test_dir,
        abnormal_dir=args.abnormal_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_split_mode="from_test",
        val_split_ratio=args.val_split_ratio,
        test_split_mode="from_dir",
        seed=args.seed,
    )
    # Materialize the exact deterministic split before training so it can be
    # recorded. Lightning repeats the same seed-controlled setup during fit.
    data.setup()
    datasets = {"train": data.train_data, "validation": data.val_data, "test": data.test_data}
    check_disjoint(datasets)
    for split in ("validation", "test"):
        if set(datasets[split].samples.label_index) != {0, 1}:
            msg = f"Both real and attack images are required in the {split} split."
            raise ValueError(msg)
    # Refuse to overwrite an earlier experiment's scores/checkpoints.
    args.results_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(
        protocol="legacy_lcf_from_test_20pct_frame_level_folder",
        checkpoint_selection="last_fixed_step",
        use_lcf=True,
        use_cltc=True,
        cltc_detach=True,
        calibration="validation_normal_q95_scales_and_validation_f1_threshold",
        score_fusion="after_pooling",
    )
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    hashes = {}
    for split, dataset in datasets.items():
        manifest = dataset.samples[["image_path", "label_index"]].sort_values("image_path").to_csv(index=False)
        (args.results_dir / f"{split}_manifest.csv").write_text(manifest, encoding="utf-8")
        hashes[split] = hashlib.sha256(manifest.encode()).hexdigest()
    model = Dinomaly(
        encoder_name=args.encoder_name,
        target_layers=args.target_layers,
        decoder_depth=args.decoder_depth,
        bottleneck_dropout=args.bottleneck_dropout,
        fuse_layer_encoder=groups,
        fuse_layer_decoder=groups,
        remove_class_token=args.remove_class_token,
        use_context_recentering=args.use_context_recentering,
        use_lcf=True,
        lcf_dropout=args.lcf_dropout,
        use_cltc=True,
        cltc_hidden_dim=args.cltc_hidden_dim,
        cltc_loss_weight=args.cltc_loss_weight,
        cltc_score_weight=args.cltc_score_weight,
        precision=args.model_precision,
        pre_processor=pre_processor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    engine = Engine(
        accelerator=args.accelerator,
        devices=1,
        max_steps=args.max_steps,
        max_epochs=-1,
        gradient_clip_val=args.gradient_clip_val,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        default_root_dir=args.results_dir,
    )
    print(f"LCF=True; CLTC=True; detach=True; alpha={args.cltc_score_weight}")
    print(
        f"Legacy LCF data split: validation is {args.val_split_ratio:.0%} of test (label-aware); "
        "final evaluation uses the remainder.",
    )
    engine.fit(model=model, datamodule=data, ckpt_path=args.resume_ckpt)
    # Lightning may move the model to CPU during teardown; use the training device again.
    model.to(engine.trainer.strategy.root_device).eval()
    validation = collect_scores(model, data.val_data, args, "Scoring legacy validation split")
    head = model.model.trajectory_head
    if head is not None:
        normal_validation = validation.loc[validation.label == 0, ["reconstruction", "trajectory"]]
        head.fit_score_scales(torch.tensor(normal_validation.to_numpy()))
        rec_scale, traj_scale = head.score_scales.cpu().tolist()
        validation["fused"] = (
            validation.reconstruction / rec_scale + args.cltc_score_weight * validation.trajectory / traj_scale
        )
    validation_output = validation[["image_path", "label", "fused"]]
    validation_output.to_csv(args.results_dir / "validation_scores.csv", index=False)
    engine.trainer.save_checkpoint(args.results_dir / "calibrated_last.ckpt")
    test = collect_scores(model, data.test_data, args, "Testing LCF+CLTC on legacy test remainder")
    test_output = test[["image_path", "label", "fused"]]
    test_output.to_csv(args.results_dir / "test_scores.csv", index=False)
    metrics = summarize_scores(test, validation)
    report = {
        "metrics": metrics,
        "manifest_sha256": hashes,
        "protocol": config["protocol"],
        "score_scales": head.score_scales.cpu().tolist() if head is not None else None,
        "note": (
            "Final LCF+CLTC fused score only. Matches the old LCF script's from_test split and metrics, "
            "but remains frame-level pooled attacks; "
            "not official OULU video/protocol metrics."
        ),
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Outputs: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
