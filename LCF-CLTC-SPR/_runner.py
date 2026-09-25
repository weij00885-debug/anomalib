# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SPR against archived LCF+CLTC results, without fitting a model."""

# ruff: file-ignore[module-import-not-at-top-of-file]

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

# Make direct script execution independent of editable-install/cwd assumptions.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

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
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize
from tqdm.auto import tqdm
from train_dinomaly_lcf_face import build_pre_processor

from anomalib.data.utils import read_image
from anomalib.models import Dinomaly
from anomalib.models.image.dinomaly.components.perturbation_recheck import SameRegionPerturbationRecheck

VIEWS = ("original", "flip", "gamma09", "gamma11")
METRIC_NAMES = ("image_AUROC", "image_Accuracy", "image_F1Score", "image_Precision", "image_Recall", "image_HTER")


@dataclass(frozen=True)
class DatasetSpec:
    """Identify an existing RGB experiment and its historical threshold policy."""

    slug: str
    dataset: str
    source_run: str
    protocol: str
    threshold_policy: str


def write_json(path: Path, value: dict) -> None:
    """Persist finite, human-readable experiment metadata."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def file_sha256(path: Path) -> str:
    """Hash a checkpoint incrementally without loading another copy into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(spec: DatasetSpec) -> argparse.Namespace:
    """Parse inference options and optional assertions about the saved source run."""
    parser = argparse.ArgumentParser(
        description=f"Evaluate LCF+CLTC+SPR on {spec.slug}; compare with already-saved original scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-run", type=Path, default=REPO_ROOT / "results" / spec.source_run)
    parser.add_argument(
        "--source-max-steps", type=int, help="Expected training steps in the saved source run (check only).",
    )
    parser.add_argument("--source-train-batch-size", type=int, help="Expected training batch size (check only).")
    parser.add_argument(
        "--source-model-precision", choices=("float16", "float32"), help="Expected saved precision (check only).",
    )
    parser.add_argument("--source-seed", type=int, help="Expected training seed (check only).")
    parser.add_argument("--source-cltc-loss-weight", type=float, help="Expected saved CLTC loss weight (check only).")
    parser.add_argument("--source-cltc-score-weight", type=float, help="Expected saved CLTC score weight (check only).")
    if spec.slug == "oulu":
        parser.add_argument("--root", type=Path, help="Optional relocated OULU-NPU_organized root.")
    else:
        parser.add_argument("--data-root", type=Path, help="Optional relocated parent of the dataset directories.")
    parser.add_argument("--device", default="cuda:0", help="Device in the CURRENT visible device namespace.")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--spr-beta", type=float, default=0.25)
    parser.add_argument("--roi-fraction", type=float, default=0.1)
    parser.add_argument("--baseline-atol", type=float, default=1e-4)
    parser.add_argument("--baseline-rtol", type=float, default=1e-3)
    parser.add_argument("--results-dir", type=Path, default=None, help="Default: a new timestamped results directory.")
    args = parser.parse_args()
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    if not np.isfinite(args.spr_beta) or not 0 <= args.spr_beta <= 1:
        parser.error("--spr-beta must be finite and in [0, 1].")
    if not np.isfinite(args.roi_fraction) or not 0 < args.roi_fraction <= 1:
        parser.error("--roi-fraction must be finite and in (0, 1].")
    for name in ("baseline_atol", "baseline_rtol"):
        if not np.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error("Baseline tolerances must be finite and nonnegative.")
    return args


def check_source_config(config: dict, args: argparse.Namespace) -> None:
    """Reject source claims that disagree with the checkpoint's training configuration."""
    fields = {
        "source_max_steps": "max_steps",
        "source_train_batch_size": "train_batch_size",
        "source_model_precision": "model_precision",
        "source_seed": "seed",
        "source_cltc_loss_weight": "cltc_loss_weight",
        "source_cltc_score_weight": "cltc_score_weight",
    }
    for option, key in fields.items():
        expected = getattr(args, option)
        if expected is None:
            continue
        saved = config.get(key)
        matches = (
            np.isclose(expected, saved, atol=1e-12, rtol=0)
            if key in {"cltc_loss_weight", "cltc_score_weight"} and saved is not None
            else expected == saved
        )
        if not matches:
            msg = (
                f"--{option.replace('_', '-')}={expected} differs from the saved source run ({key}={saved}). "
                "These source options verify a checkpoint; they do not retrain it. "
                "Select a matching --source-run to change the trained model."
            )
            raise ValueError(msg)


def remap_image_path(value: str, config: dict, args: argparse.Namespace, spec: DatasetSpec) -> Path:
    """Change only a manifest's root; never infer a new data split."""
    override = getattr(args, "root", None) if spec.slug == "oulu" else getattr(args, "data_root", None)
    if override is None:
        return Path(value)
    if spec.slug == "oulu":
        old_root = config["root"]
        new_root = override
    else:
        old_root = str(PurePosixPath(config["data_root"].replace("\\", "/")) / spec.dataset / config["mode"])
        new_root = override / spec.dataset / config["mode"]
    relative = PurePosixPath(value.replace("\\", "/")).relative_to(PurePosixPath(old_root.replace("\\", "/")))
    if ".." in relative.parts:
        msg = f"Invalid parent traversal in manifest: {value}"
        raise ValueError(msg)
    return new_root.joinpath(*relative.parts)


def attach_archived_scores(frame: pd.DataFrame, source: Path, split: str) -> None:
    """Attach archived scores in manifest order, rejecting path/label mismatches."""
    archived = pd.read_csv(source / f"{split}_scores.csv")
    if not {"image_path", "label", "fused"}.issubset(archived.columns):
        msg = f"Missing original score columns in {split}_scores.csv."
        raise ValueError(msg)
    if archived.image_path.duplicated().any() or set(archived.image_path) != set(frame.image_path):
        msg = f"Archived {split} scores do not match the manifest."
        raise ValueError(msg)
    archived = archived.set_index("image_path").loc[frame.image_path]
    if not np.array_equal(archived.label.to_numpy(), frame.label_index.to_numpy()):
        msg = f"Archived labels differ from the {split} manifest."
        raise ValueError(msg)
    if not np.isfinite(archived.fused.to_numpy()).all():
        msg = f"Non-finite archived scores in {split}."
        raise ValueError(msg)
    frame["original_fused"] = archived.fused.to_numpy()


def load_inputs(spec: DatasetSpec, args: argparse.Namespace) -> tuple[dict, dict, dict, dict]:
    """Read the original manifests, labels, configuration and archived predictions."""
    source = args.source_run.resolve()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    check_source_config(config, args)
    report = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    if config.get("protocol") != spec.protocol or report.get("protocol") != spec.protocol:
        msg = f"This wrapper requires protocol {spec.protocol}. Got {config.get('protocol')}."
        raise ValueError(msg)
    if spec.slug != "oulu" and (config.get("dataset") != spec.dataset or config.get("mode") != "rgb"):
        msg = f"Expected the saved RGB experiment for {spec.dataset}."
        raise ValueError(msg)
    if not all(config.get(key) is True for key in ("use_lcf", "use_cltc", "cltc_detach")):
        msg = "Source experiment must have LCF, detached CLTC and saved calibration enabled."
        raise ValueError(msg)
    if not (source / "calibrated_last.ckpt").is_file():
        msg = f"Missing calibrated checkpoint: {source / 'calibrated_last.ckpt'}"
        raise FileNotFoundError(msg)
    manifests, hashes, seen = {}, {}, {}
    for split in ("train", "validation", "test"):
        path = source / f"{split}_manifest.csv"
        content = path.read_text(encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode()).hexdigest()
        if hashes[split] != report["manifest_sha256"][split]:
            msg = f"Source {split} manifest differs from the archived run."
            raise ValueError(msg)
        frame = pd.read_csv(path)
        if frame.empty or not {"image_path", "label_index"}.issubset(frame.columns):
            msg = f"Malformed {split} manifest."
            raise ValueError(msg)
        if frame.image_path.isna().any() or frame.image_path.duplicated().any():
            msg = f"Missing or duplicate image paths in {split}."
            raise ValueError(msg)
        expected_labels = {0} if split == "train" else {0, 1}
        if set(frame.label_index) != expected_labels:
            msg = f"Unexpected labels in {split}; expected {expected_labels}."
            raise ValueError(msg)
        resolved = []
        for value in frame.image_path:
            path = remap_image_path(str(value), config, args, spec).resolve()
            key = str(path)
            if key in seen:
                msg = f"Image overlap between {seen[key]} and {split}: {path}"
                raise ValueError(msg)
            seen[key] = split
            if split != "train" and not path.is_file():
                msg = f"Missing image: {path}. Use the dataset root override if files moved."
                raise FileNotFoundError(msg)
            resolved.append(str(path))
        frame["resolved_image_path"] = resolved
        if split != "train":
            attach_archived_scores(frame, source, split)
        manifests[split] = frame.reset_index(drop=True)
    return config, report, manifests, hashes


class ManifestImages(Dataset):
    """Load existing manifest rows, reproducing the legacy dataset's CPU Resize."""

    def __init__(self, frame: pd.DataFrame, image_size: int) -> None:
        self.paths = frame.resolved_image_path.tolist()
        # During fit, AnomalibDataModule appends the model's antialiased Resize
        # to each dataset. The old score collector then calls pre_processor again.
        self.resize = Resize((image_size, image_size), antialias=True)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.resize(read_image(self.paths[index], as_tensor=True)), index


def apply_view(images: torch.Tensor, name: str) -> torch.Tensor:
    """Transform canonical resized RGB [0,1] BEFORE crop and normalization."""
    if name == "original":
        return images
    if name == "flip":
        return images.flip(-1)
    if name in {"gamma09", "gamma11"}:
        return images.clamp(0, 1).pow(0.9 if name == "gamma09" else 1.1)
    msg = f"Unknown SPR view: {name}"
    raise ValueError(msg)


def restore_model(source: Path, config: dict, device: torch.device) -> Dinomaly:
    """Construct from saved hyperparameters, strictly load weights, then freeze.

    Memory mapping avoids eagerly reading optimizer state into RAM. Only local
    checkpoints produced by the user's training runs should be passed here.
    """
    checkpoint = torch.load(source / "calibrated_last.ckpt", map_location="cpu", weights_only=False, mmap=True)
    hparams = dict(checkpoint["hyper_parameters"])
    for key in ("encoder_name", "cltc_hidden_dim", "cltc_score_weight", "cltc_loss_weight"):
        if hparams.get(key) != config[key]:
            msg = f"Checkpoint/config mismatch for {key}."
            raise ValueError(msg)
    for key in ("use_lcf", "use_cltc"):
        if hparams.get(key) is not True:
            msg = f"Checkpoint does not enable {key}."
            raise ValueError(msg)
    if hparams.get("precision") != config["model_precision"]:
        msg = "Checkpoint precision differs from the saved configuration."
        raise ValueError(msg)
    processor = build_pre_processor(config["image_size"], config.get("crop_size"))
    model = Dinomaly(**hparams, pre_processor=processor, post_processor=False, evaluator=False, visualizer=False)
    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.eval().requires_grad_(requires_grad=False).to(device)
    return model


def assert_original_matches(actual: np.ndarray, archived: np.ndarray, atol: float, rtol: float) -> None:
    """Stop an unfair comparison if the necessary original-view pass has drifted."""
    if not np.isfinite(actual).all() or not np.allclose(actual, archived, atol=atol, rtol=rtol):
        maximum = float(np.max(np.abs(actual - archived)))
        msg = (
            f"Original-view scores differ from the saved LCF+CLTC scores (max abs error={maximum:.6g}). "
            "Check checkpoint, preprocessing, data paths and precision before comparing SPR. "
            "Do not interpret this as an SPR improvement."
        )
        raise RuntimeError(msg)


@torch.inference_mode()
def collect_spr(
    model: Dinomaly,
    frame: pd.DataFrame,
    config: dict,
    args: argparse.Namespace,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    """Use exactly four sequential view passes and one original-selected ROI."""
    loader = DataLoader(
        ManifestImages(frame, config["image_size"]),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=model.device.type == "cuda",
    )
    module = SameRegionPerturbationRecheck(args.roi_fraction).eval()
    q_rec, q_traj = model.model.trajectory_head.score_scales
    alpha = model.model.cltc_score_weight
    rows = []
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    for raw_images, indices in tqdm(loader, desc=f"SPR {split}: 4 views/image", unit="batch"):
        images = raw_images.to(model.device, non_blocking=True)
        region_scores, view_scores = [], {}
        roi = None
        for name in VIEWS:
            processed = model.pre_processor(apply_view(images, name))
            evidence = model.model.predict_evidence(processed)
            fused_map = (
                evidence["reconstruction_patch_map"].clamp_min(0) / q_rec
                + alpha * evidence["trajectory_patch_map"].clamp_min(0) / q_traj
            )
            view_scores[name] = evidence["fused"].float().cpu().numpy()
            if name == "original":
                archived = frame.iloc[indices.tolist()].original_fused.to_numpy()
                assert_original_matches(view_scores[name], archived, args.baseline_atol, args.baseline_rtol)
                roi = module.make_roi(fused_map)
            if name == "flip":
                fused_map = fused_map.flip(-1)
            region_scores.append(module.region_score(fused_map, roi))
        values = torch.stack(region_scores, dim=1)
        summary = {name: value.cpu().numpy() for name, value in module(values).items()}
        values = values.cpu().numpy()
        for offset, index in enumerate(indices.tolist()):
            item = frame.iloc[index]
            row = {
                "image_path": item.image_path,
                "resolved_image_path": item.resolved_image_path,
                "label": int(item.label_index),
                "original_fused": float(item.original_fused),
                "original_forward": float(view_scores["original"][offset]),
                "spr_evidence": float(summary["evidence"][offset]),
                "spr_median": float(summary["median"][offset]),
                "spr_mad": float(summary["mad"][offset]),
            }
            for j, name in enumerate(VIEWS):
                row[f"region_{name}"] = float(values[offset, j])
                # Cache available scalars for a later same-budget TTA study;
                # no TTA aggregate or TTA metric is computed in this experiment.
                row[f"view_{name}_fused"] = float(view_scores[name][offset])
            rows.append(row)
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    result = pd.DataFrame(rows)
    if not np.isfinite(result.select_dtypes(include="number").to_numpy()).all():
        msg = f"Non-finite SPR evidence in {split}."
        raise ValueError(msg)
    cost = {
        "images": len(result),
        "network_forward_batches": len(loader) * len(VIEWS),
        "network_image_forwards": len(result) * len(VIEWS),
        "views_per_image": len(VIEWS),
        "wall_seconds_including_io": elapsed,
        "seconds_per_image_including_io": elapsed / len(result),
        "max_abs_original_score_error": float((result.original_forward - result.original_fused).abs().max()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device)
        if model.device.type == "cuda"
        else None,
    }
    return result, cost


def choose_threshold(labels: np.ndarray, scores: np.ndarray, policy: str) -> float:
    """Reproduce the legacy strict-greater-than decision and tie-breaking policy."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        msg = "Threshold calibration needs finite scores and both validation classes."
        raise ValueError(msg)
    if policy == "validation_f1":
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        return float(thresholds[int(np.argmax(f1))])
    if policy != "validation_hter":
        msg = f"Unknown threshold policy: {policy}"
        raise ValueError(msg)
    # Equivalent to the old ascending unique-threshold loop using <= ties,
    # but compute counts with sorted arrays rather than an O(N^2) scan.
    candidates = np.unique(scores)
    normal = np.sort(scores[labels == 0])
    attack = np.sort(scores[labels == 1])
    fpr = (len(normal) - np.searchsorted(normal, candidates, side="right")) / len(normal)
    fnr = np.searchsorted(attack, candidates, side="right") / len(attack)
    hter = (fpr + fnr) / 2
    return float(candidates[np.flatnonzero(hter == hter.min())[-1]])


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Report the same six image-level attack-positive metrics as the source runs."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    prediction = scores > threshold
    fpr = float(prediction[labels == 0].mean())
    fnr = float((~prediction[labels == 1]).mean())
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_Accuracy": float(accuracy_score(labels, prediction)),
        "image_F1Score": float(f1_score(labels, prediction, zero_division=0)),
        "image_Precision": float(precision_score(labels, prediction, zero_division=0)),
        "image_Recall": float(recall_score(labels, prediction, zero_division=0)),
        "image_HTER": (fpr + fnr) / 2,
    }


def fit_calibration(validation: pd.DataFrame, beta: float, policy: str) -> dict:
    """Fit SPR scales on validation normals and the decision threshold on validation only."""
    normal = validation.loc[validation.label == 0]
    # Use the exact archived baseline scale and baseline term, after checking
    # the new original-view forward against it. No independent baseline rerun.
    q0 = float(np.quantile(normal.original_fused.to_numpy(), 0.95))
    qe = float(np.quantile(normal.spr_evidence.to_numpy(), 0.95))
    if not np.isfinite([q0, qe]).all() or min(q0, qe) <= 1e-8:
        msg = "SPR calibration is degenerate (normal q95 <= 1e-8); do not amplify an epsilon denominator."
        raise ValueError(msg)
    calibration = {"beta": beta, "q0": q0, "qe": qe, "threshold_policy": policy}
    scores = combine_spr(validation, calibration)
    calibration["original_threshold"] = choose_threshold(
        validation.label.to_numpy(), validation.original_fused.to_numpy(), policy,
    )
    calibration["spr_threshold"] = choose_threshold(validation.label.to_numpy(), scores, policy)
    return calibration


def combine_spr(frame: pd.DataFrame, calibration: dict) -> np.ndarray:
    """Combine archived S0 with calibrated SPR evidence; beta=0 restores S0 exactly."""
    beta = calibration["beta"]
    return (1 - beta) * frame.original_fused.to_numpy() + beta * calibration[
        "q0"
    ] * frame.spr_evidence.to_numpy() / calibration["qe"]


def run(spec: DatasetSpec) -> None:
    """Evaluate a frozen source checkpoint and compare C with archived baseline metrics."""
    args = parse_args(spec)
    config, archived_report, manifests, hashes = load_inputs(spec, args)
    print(
        f"Saved source: steps={config.get('max_steps')}, train_batch={config.get('train_batch_size')}, "
        f"precision={config['model_precision']}, seed={config['seed']}, "
        f"CLTC lambda={config['cltc_loss_weight']}, alpha={config['cltc_score_weight']}; "
        f"SPR beta={args.spr_beta}, ROI={args.roi_fraction}, eval_batch={args.eval_batch_size}",
        flush=True,
    )
    saved_val, saved_test = manifests["validation"], manifests["test"]
    saved_threshold = choose_threshold(
        saved_val.label_index.to_numpy(), saved_val.original_fused.to_numpy(), spec.threshold_policy,
    )
    saved_metrics = metrics_at_threshold(
        saved_test.label_index.to_numpy(), saved_test.original_fused.to_numpy(), saved_threshold,
    )
    if any(
        not np.isclose(saved_metrics[key], archived_report["metrics"][key], atol=1e-6, rtol=0)
        for key in METRIC_NAMES
    ):
        msg = "Archived metrics do not match their saved scores under the expected threshold policy."
        raise ValueError(msg)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA is unavailable in this Python environment. Use your working torch_env Python or --device cpu."
        raise RuntimeError(msg)
    source = args.source_run.resolve()
    output = args.results_dir or REPO_ROOT / "results" / f"{spec.slug}_lcf_cltc_spr_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    print(f"Source: {source}\nOutput: {output.resolve()}", flush=True)
    print("Using archived original metrics/scores. SPR: original + flip + gamma09 + gamma11; no training.", flush=True)
    print("Hashing the source checkpoint for provenance...", flush=True)
    checkpoint_hash = file_sha256(source / "calibrated_last.ckpt")
    settings = {
        "source_run": str(source),
        "source_config": config,
        "protocol": spec.protocol,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": hashes,
        "views": list(VIEWS),
        "spr_beta": args.spr_beta,
        "roi_fraction": args.roi_fraction,
        "baseline_source": "archived_scores_and_metrics",
        "training": False,
        "vec_enabled": False,
        "tta_comparison_enabled": False,
        "view_domain": "legacy_antialiased_resized_rgb_before_normalization",
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "baseline_atol": args.baseline_atol,
        "baseline_rtol": args.baseline_rtol,
        "root_override": str(getattr(args, "root", None) or getattr(args, "data_root", None) or ""),
    }
    write_json(output / "config.json", settings)
    torch.manual_seed(config["seed"])
    print("Restoring checkpoint with strict state loading...", flush=True)
    model = restore_model(source, config, device)
    scales = model.model.trajectory_head.score_scales.detach().cpu().numpy()
    if not np.allclose(scales, archived_report["score_scales"], rtol=1e-6, atol=1e-8) or (scales <= 0).any():
        msg = "Checkpoint branch calibration differs from the archived report."
        raise ValueError(msg)
    validation, validation_cost = collect_spr(model, manifests["validation"], config, args, "validation")
    calibration = fit_calibration(validation, args.spr_beta, spec.threshold_policy)
    calibration.update(
        checkpoint_sha256=checkpoint_hash,
        validation_manifest_sha256=hashes["validation"],
        score_scales=scales.tolist(),
    )
    write_json(output / "calibration.json", calibration)
    validation["spr"] = combine_spr(validation, calibration)
    validation.to_csv(output / "validation_scores.csv", index=False)
    print("SPR calibration and threshold fixed. Evaluating the saved test manifest...", flush=True)
    test, test_cost = collect_spr(model, manifests["test"], config, args, "test")
    test["spr"] = combine_spr(test, calibration)
    test.to_csv(output / "test_scores.csv", index=False)
    original = archived_report["metrics"]
    spr_metrics = metrics_at_threshold(test.label.to_numpy(), test.spr.to_numpy(), calibration["spr_threshold"])
    differences = {key: spr_metrics[key] - original[key] for key in METRIC_NAMES}
    report = {
        "metrics": {"original_lcf_cltc": original, "lcf_cltc_spr": spr_metrics},
        "delta_spr_minus_original": differences,
        "delta_percentage_points": {key: value * 100 for key, value in differences.items()},
        "calibration": calibration,
        "protocol": spec.protocol,
        "manifest_sha256": hashes,
        "runtime": {"validation": validation_cost, "test": test_cost},
        "baseline_source": str(source / "metrics.json"),
        "note": (
            "Paired legacy frame-level evaluation against archived LCF+CLTC. "
            "No retraining, no VEC, no TTA comparison."
        ),
    }
    write_json(output / "metrics.json", report)
    comparison = pd.DataFrame({
        "metric": METRIC_NAMES,
        "original_lcf_cltc": [original[k] for k in METRIC_NAMES],
        "lcf_cltc_spr": [spr_metrics[k] for k in METRIC_NAMES],
        "delta_percentage_points": [100 * differences[k] for k in METRIC_NAMES],
    })
    comparison.to_csv(output / "comparison.csv", index=False)
    print(comparison.to_string(index=False), flush=True)
    print("AUROC/Accuracy/F1/Precision/Recall: higher is better. HTER: lower is better.")
    print(f"Outputs: {output.resolve()}")
