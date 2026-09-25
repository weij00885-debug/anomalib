# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare archived SPR with fixed four-view mean TTA using CPU score arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS = ("original", "flip", "gamma09", "gamma11")
VIEW_COLUMNS = [f"view_{view}_fused" for view in VIEWS]
METRICS = ("image_AUROC", "image_Accuracy", "image_F1Score", "image_Precision", "image_Recall", "image_HTER")


@dataclass(frozen=True)
class DatasetSpec:
    """Pin one dataset, completed SPR run and historical threshold policy."""

    slug: str
    dataset: str
    spr_run: str
    protocol: str
    threshold_policy: str


def read_json(path: Path) -> dict:
    """Read an existing experiment document."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Write finite, readable output metadata."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def choose_threshold(labels: np.ndarray, scores: np.ndarray, policy: str) -> float:
    """Match SPR's legacy threshold candidates, tie breaks and strict > predictions."""
    if set(labels) != {0, 1} or not np.isfinite(scores).all():
        msg = "Threshold selection needs both validation classes and finite scores."
        raise ValueError(msg)
    if policy == "validation_f1":
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        return float(thresholds[int(np.argmax(f1))])
    if policy != "validation_hter":
        msg = f"Unsupported threshold policy: {policy}"
        raise ValueError(msg)
    candidates = np.unique(scores)
    normal, attack = np.sort(scores[labels == 0]), np.sort(scores[labels == 1])
    fpr = (len(normal) - np.searchsorted(normal, candidates, side="right")) / len(normal)
    fnr = np.searchsorted(attack, candidates, side="right") / len(attack)
    hter = (fpr + fnr) / 2
    return float(candidates[np.flatnonzero(hter == hter.min())[-1]])


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Compute the six historical attack-positive image metrics at a fixed threshold."""
    prediction = scores > threshold
    return {
        "image_AUROC": float(roc_auc_score(labels, scores)),
        "image_Accuracy": float(accuracy_score(labels, prediction)),
        "image_F1Score": float(f1_score(labels, prediction, zero_division=0)),
        "image_Precision": float(precision_score(labels, prediction, zero_division=0)),
        "image_Recall": float(recall_score(labels, prediction, zero_division=0)),
        "image_HTER": float((prediction[labels == 0].mean() + (~prediction[labels == 1]).mean()) / 2),
    }


def mean_tta(frame: pd.DataFrame) -> np.ndarray:
    """Give each cached view equal weight, without fitting scales or view weights."""
    return frame[VIEW_COLUMNS].to_numpy(dtype=np.float64).mean(axis=1)


def blend_tta(frame: pd.DataFrame, calibration: dict) -> np.ndarray:
    """Match SPR's original-score mixture, replacing region evidence with mean TTA."""
    beta = calibration["beta"]
    return (
        (1 - beta) * frame.original_fused.to_numpy()
        + beta * calibration["q0"] * frame.tta_mean.to_numpy() / calibration["q_tta"]
    )


def fit_blended_tta(validation: pd.DataFrame, beta: float, policy: str) -> dict:
    """Fit matched-mixture scales on validation normals and its threshold on validation."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        msg = "The saved SPR beta must be finite and in [0, 1]."
        raise ValueError(msg)
    normal = validation.loc[validation.label == 0]
    q0 = float(np.quantile(normal.original_fused.to_numpy(), 0.95))
    q_tta = float(np.quantile(normal.tta_mean.to_numpy(), 0.95))
    if not np.isfinite([q0, q_tta]).all() or min(q0, q_tta) <= 1e-8:
        msg = "Matched TTA calibration is degenerate (normal q95 <= 1e-8)."
        raise ValueError(msg)
    calibration = {"beta": beta, "q0": q0, "q_tta": q_tta, "threshold_policy": policy}
    calibration["threshold"] = choose_threshold(
        validation.label.to_numpy(), blend_tta(validation, calibration), policy,
    )
    return calibration


def read_scores(path: Path, required: list[str]) -> pd.DataFrame:
    """Reject incomplete, duplicate or nonfinite score archives."""
    frame = pd.read_csv(path, float_precision="round_trip")
    if frame.empty or not {"image_path", "label", *required}.issubset(frame.columns):
        msg = f"Missing score columns or empty archive: {path}"
        raise ValueError(msg)
    if frame.image_path.isna().any() or frame.image_path.duplicated().any() or set(frame.label) != {0, 1}:
        msg = f"Invalid paths or labels: {path}"
        raise ValueError(msg)
    if not np.isfinite(frame[required].to_numpy(dtype=np.float64)).all():
        msg = f"Nonfinite scores: {path}"
        raise ValueError(msg)
    return frame


def check_close(actual: object, expected: object, message: str, atol: float = 1e-10, rtol: float = 0) -> None:
    """Fail on inconsistent saved values instead of silently mixing experiments."""
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise ValueError(message)


def validate_split(
    spr: Path, baseline: Path, split: str, manifest: pd.DataFrame, config: dict, calibration: dict,
) -> pd.DataFrame:
    """Join by image path and verify original and SPR scores against their archives."""
    frame = read_scores(spr / f"{split}_scores.csv", ["original_fused", "spr", "spr_evidence", *VIEW_COLUMNS])
    original = read_scores(baseline / f"{split}_scores.csv", ["fused"])
    for candidate in (frame, original):
        if set(candidate.image_path) != set(manifest.image_path):
            msg = f"{split} score paths differ from the original manifest."
            raise ValueError(msg)
        labels = candidate.set_index("image_path").loc[manifest.image_path, "label"].to_numpy()
        if not np.array_equal(labels, manifest.label_index.to_numpy()):
            msg = f"{split} score labels differ from the original manifest."
            raise ValueError(msg)
    frame = frame.set_index("image_path").loc[manifest.image_path].reset_index()
    archived = original.set_index("image_path").loc[frame.image_path, "fused"].to_numpy()
    check_close(frame.original_fused, archived, f"{split}: original scores changed.")
    check_close(
        frame.view_original_fused, archived, f"{split}: original-view scores drifted.",
        atol=config["baseline_atol"], rtol=config["baseline_rtol"],
    )
    reconstructed = (
        (1 - calibration["beta"]) * frame.original_fused
        + calibration["beta"] * calibration["q0"] * frame.spr_evidence / calibration["qe"]
    )
    check_close(frame.spr, reconstructed, f"{split}: cached SPR scores disagree with saved calibration.")
    return frame


def load_inputs(spec: DatasetSpec, spr: Path, baseline_override: Path | None) -> tuple[dict, dict, dict, Path, dict]:
    """Validate completed score archives without reading images or checkpoint weights."""
    config, report = read_json(spr / "config.json"), read_json(spr / "metrics.json")
    calibration = read_json(spr / "calibration.json")
    source = config["source_config"]
    if any(item.get("protocol") != spec.protocol for item in (config, report, source)):
        msg = f"This script requires protocol {spec.protocol}."
        raise ValueError(msg)
    if spec.slug != "oulu" and (source.get("dataset") != spec.dataset or source.get("mode") != "rgb"):
        msg = f"Expected an RGB SPR run for {spec.dataset}."
        raise ValueError(msg)
    if config.get("views") != list(VIEWS) or calibration != report.get("calibration"):
        msg = "Saved views/calibration do not match the completed SPR report."
        raise ValueError(msg)
    if calibration["threshold_policy"] != spec.threshold_policy or calibration["beta"] != config["spr_beta"]:
        msg = "Saved SPR threshold policy or beta is inconsistent."
        raise ValueError(msg)
    if calibration["checkpoint_sha256"] != config["checkpoint_sha256"]:
        msg = "SPR checkpoint provenance is inconsistent."
        raise ValueError(msg)
    baseline = baseline_override or Path(config["source_run"])
    if baseline_override is None and not baseline.is_dir():
        # Allow the same workspace to be opened through Windows or WSL.
        baseline = REPO_ROOT / "results" / PurePosixPath(config["source_run"].replace("\\", "/")).name
    baseline = baseline.resolve()
    original_report = read_json(baseline / "metrics.json")
    if read_json(baseline / "config.json") != source or original_report["protocol"] != spec.protocol:
        msg = "Baseline directory does not match the SPR source configuration."
        raise ValueError(msg)
    hashes = config["manifest_sha256"]
    if hashes != report["manifest_sha256"] or hashes != original_report["manifest_sha256"]:
        msg = "SPR and baseline manifest hashes differ."
        raise ValueError(msg)
    frames, seen = {}, set()
    files = [spr / name for name in ("config.json", "metrics.json", "calibration.json")]
    files.extend(baseline / name for name in ("config.json", "metrics.json"))
    for split in ("train", "validation", "test"):
        path = baseline / f"{split}_manifest.csv"
        if hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest() != hashes[split]:
            msg = f"The original {split} manifest has changed."
            raise ValueError(msg)
        manifest = pd.read_csv(path)
        labels = {0} if split == "train" else {0, 1}
        if (
            manifest.empty or manifest.image_path.isna().any() or manifest.image_path.duplicated().any()
            or set(manifest.label_index) != labels or seen.intersection(manifest.image_path)
        ):
            msg = f"Invalid or overlapping {split} manifest."
            raise ValueError(msg)
        seen.update(manifest.image_path)
        files.append(path)
        if split != "train":
            frames[split] = validate_split(spr, baseline, split, manifest, config, calibration)
            files.extend((spr / f"{split}_scores.csv", baseline / f"{split}_scores.csv"))
    for key, column in (("original_lcf_cltc", "original_fused"), ("lcf_cltc_spr", "spr")):
        threshold = calibration["original_threshold" if column == "original_fused" else "spr_threshold"]
        validation = frames["validation"]
        check_close(
            choose_threshold(validation.label.to_numpy(), validation[column].to_numpy(), spec.threshold_policy),
            threshold, f"Saved {key} threshold cannot be reproduced from validation.",
        )
        test = frames["test"]
        actual = metrics_at_threshold(test.label.to_numpy(), test[column].to_numpy(), threshold)
        check_close(
            [actual[k] for k in METRICS], [report["metrics"][key][k] for k in METRICS],
            f"Saved {key} metrics cannot be reproduced.", atol=1e-6,
        )
    check_close(
        [original_report["metrics"][k] for k in METRICS],
        [report["metrics"]["original_lcf_cltc"][k] for k in METRICS], "Original metric archives disagree.",
    )
    provenance = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return config, report, frames, baseline, provenance


def run(spec: DatasetSpec) -> None:
    """Compare plain and matched-mixture TTA with thresholds selected only on validation."""
    parser = argparse.ArgumentParser(
        description=f"Compare original / mean TTA / matched-mixture TTA / SPR for {spec.slug} using cached CSVs.",
    )
    parser.add_argument("--spr-run", type=Path, default=REPO_ROOT / "results" / spec.spr_run)
    parser.add_argument("--baseline-run", type=Path, help="Optional relocated original LCF+CLTC result directory.")
    parser.add_argument("--tta-reduction", choices=("mean",), default="mean", help="Fixed equal-weight four-view mean.")
    parser.add_argument("--results-dir", type=Path, help="Default: a new timestamped comparison directory.")
    parser.add_argument(
        "--check-inputs-only", action="store_true", help="Validate archives without evaluating TTA or writing.",
    )
    args = parser.parse_args()
    spr = args.spr_run.resolve()
    print(f"Checking completed SPR archive: {spr}", flush=True)
    config, saved, frames, baseline, provenance = load_inputs(spec, spr, args.baseline_run)
    source = config["source_config"]
    print(
        f"Source: steps={source['max_steps']}, train_batch={source['train_batch_size']}, "
        f"precision={source['model_precision']}, seed={source['seed']}, "
        f"CLTC lambda={source['cltc_loss_weight']}, alpha={source['cltc_score_weight']}; "
        f"SPR beta={config['spr_beta']}, ROI={config['roi_fraction']}",
        flush=True,
    )
    print(f"Validated: validation={len(frames['validation'])}, test={len(frames['test'])}; cached views={list(VIEWS)}")
    if args.check_inputs_only:
        print("Input checks passed. TTA metrics were not evaluated; no files written.")
        return
    output = args.results_dir or REPO_ROOT / "results" / f"{spec.slug}_spr_tta_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    validation = frames["validation"]
    validation["tta_mean"] = mean_tta(validation)
    threshold = choose_threshold(validation.label.to_numpy(), validation.tta_mean.to_numpy(), spec.threshold_policy)
    matched_calibration = fit_blended_tta(validation, config["spr_beta"], spec.threshold_policy)
    check_close(matched_calibration["q0"], saved["calibration"]["q0"], "Original q95 differs from SPR calibration.")
    validation["tta_blend"] = blend_tta(validation, matched_calibration)
    calibration = {
        "threshold_policy": spec.threshold_policy,
        "tta_threshold": threshold,
        "tta_reduction": "mean",
        "tta_view_weights": [0.25] * len(VIEWS),
        "views": list(VIEWS),
        "original_threshold": saved["calibration"]["original_threshold"],
        "spr_threshold": saved["calibration"]["spr_threshold"],
        "validation_manifest_sha256": config["manifest_sha256"]["validation"],
        "matched_tta": matched_calibration,
    }
    write_json(output / "calibration.json", calibration)
    print("TTA scales/thresholds fixed from validation. Computing test metrics from cached scores...", flush=True)
    test = frames["test"]
    test["tta_mean"] = mean_tta(test)
    test["tta_blend"] = blend_tta(test, matched_calibration)
    tta = metrics_at_threshold(test.label.to_numpy(), test.tta_mean.to_numpy(), threshold)
    matched_tta = metrics_at_threshold(
        test.label.to_numpy(), test.tta_blend.to_numpy(), matched_calibration["threshold"],
    )
    original, c_metrics = saved["metrics"]["original_lcf_cltc"], saved["metrics"]["lcf_cltc_spr"]
    comparison = pd.DataFrame({
        "metric": METRICS,
        "original_lcf_cltc": [original[k] for k in METRICS],
        "tta_mean_4views": [tta[k] for k in METRICS],
        "tta_blend_4views": [matched_tta[k] for k in METRICS],
        "lcf_cltc_spr": [c_metrics[k] for k in METRICS],
        "tta_minus_original_pp": [100 * (tta[k] - original[k]) for k in METRICS],
        "spr_minus_original_pp": [100 * (c_metrics[k] - original[k]) for k in METRICS],
        "spr_minus_tta_pp": [100 * (c_metrics[k] - tta[k]) for k in METRICS],
        "blend_minus_original_pp": [100 * (matched_tta[k] - original[k]) for k in METRICS],
        "spr_minus_blend_pp": [100 * (c_metrics[k] - matched_tta[k]) for k in METRICS],
    })
    comparison.to_csv(output / "comparison.csv", index=False)
    for split, frame in frames.items():
        frame[["image_path", "label", "original_fused", *VIEW_COLUMNS, "tta_mean", "tta_blend", "spr"]].to_csv(
            output / f"{split}_scores.csv", index=False,
        )
    write_json(output / "config.json", {
        "dataset": spec.slug, "spr_run": str(spr), "baseline_run": str(baseline),
        "source_config": source, "spr_beta": config["spr_beta"], "roi_fraction": config["roi_fraction"],
        "protocol": spec.protocol, "views": list(VIEWS), "tta_reduction": "mean",
        "input_file_sha256": provenance, "checkpoint_sha256": config["checkpoint_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "matched_tta_formula": "(1-beta)*original_fused + beta*q0*tta_mean/q_tta",
        "matched_tta_beta_source": "saved_spr_beta",
    })
    write_json(output / "metrics.json", {
        "metrics": {
            "original_lcf_cltc": original, "tta_mean_4views": tta,
            "tta_blend_4views": matched_tta, "lcf_cltc_spr": c_metrics,
        },
        "calibration": calibration, "protocol": spec.protocol,
        "delta_spr_minus_tta_percentage_points": {k: 100 * (c_metrics[k] - tta[k]) for k in METRICS},
        "delta_spr_minus_blend_percentage_points": {k: 100 * (c_metrics[k] - matched_tta[k]) for k in METRICS},
        "manifest_sha256": config["manifest_sha256"],
        "network_views_per_image": {
            "original_lcf_cltc": 1, "tta_mean_4views": 4, "tta_blend_4views": 4, "lcf_cltc_spr": 4,
        },
        "additional_network_forwards": 0,
        "note": (
            "Paired legacy frame-level comparison. Four cached views, mean and SPR-beta-matched TTA; "
            "validation-only scales/thresholds."
        ),
    })
    print(comparison.to_string(index=False))
    print("spr_minus_tta_pp: positive favors SPR except HTER, where negative favors SPR.")
    print("spr_minus_blend_pp: SPR minus TTA with matched beta and normal-q95 calibration; same sign convention.")
    print(f"Outputs: {output.resolve()}")
