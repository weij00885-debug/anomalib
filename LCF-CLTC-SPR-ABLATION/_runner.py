# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate two fixed-region SPR ablations from the existing four-view score cache."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("spr_ablation_shared", REPO_ROOT / "LCF-CLTC-SPR-TTA" / "_runner.py")
shared = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = shared
_SPEC.loader.exec_module(shared)
DatasetSpec = shared.DatasetSpec
REGION_COLUMNS = [f"region_{view}" for view in shared.VIEWS]
VARIANTS = ("median_no_penalty", "mean_no_penalty")


def region_evidence(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Verify cached FP32 region statistics and derive the two unpenalized alternatives."""
    required = [*REGION_COLUMNS, "spr_median", "spr_mad", "spr_evidence"]
    if not set(required).issubset(frame.columns):
        msg = "Ablations require all four cached region scores, spr_median, spr_mad and spr_evidence."
        raise ValueError(msg)
    values = frame[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        msg = "Cached region statistics must be finite and nonnegative."
        raise ValueError(msg)
    regions = frame[REGION_COLUMNS].to_numpy(dtype=np.float32)
    median = np.median(regions, axis=1)
    mad = np.median(np.abs(regions - median[:, None]), axis=1)
    evidence = median / (1 + mad / (np.abs(median) + np.float32(1e-8)))
    for name, actual in (("spr_median", median), ("spr_mad", mad), ("spr_evidence", evidence)):
        shared.check_close(
            actual, frame[name].to_numpy(), f"Cached {name} does not match the region scores.",
            atol=1e-7, rtol=2e-6,
        )
    return {
        # Retain the exact archived FP32 median so removing MAD is the only evidence change.
        "median_no_penalty": frame.spr_median.to_numpy(dtype=np.float64),
        "mean_no_penalty": regions.mean(axis=1, dtype=np.float32).astype(np.float64),
    }


def combine(original: np.ndarray, evidence: np.ndarray, calibration: dict) -> np.ndarray:
    """Apply the same original-score mixture as the saved complete SPR experiment."""
    beta = calibration["beta"]
    return (1 - beta) * original + beta * calibration["q0"] * evidence / calibration["qe"]


def fit_variant(validation: pd.DataFrame, evidence: np.ndarray, beta: float, policy: str) -> dict:
    """Fit one variant using validation-normal q95 scales and validation-only threshold."""
    if not np.isfinite(beta) or not 0 <= beta <= 1:
        msg = "Saved SPR beta must be finite and in [0, 1]."
        raise ValueError(msg)
    normal = validation.label.to_numpy() == 0
    q0 = float(np.quantile(validation.original_fused.to_numpy()[normal], 0.95))
    qe = float(np.quantile(evidence[normal], 0.95))
    if not np.isfinite([q0, qe]).all() or min(q0, qe) <= 1e-8:
        msg = "Ablation calibration is degenerate (normal q95 <= 1e-8)."
        raise ValueError(msg)
    calibration = {"beta": beta, "q0": q0, "qe": qe, "threshold_policy": policy}
    calibration["threshold"] = shared.choose_threshold(
        validation.label.to_numpy(), combine(validation.original_fused.to_numpy(), evidence, calibration), policy,
    )
    return calibration


def run(spec: DatasetSpec) -> None:
    """Save the two prespecified ablations alongside archived baseline and complete SPR."""
    parser = argparse.ArgumentParser(description=f"Cached same-region SPR ablations for {spec.slug}.")
    parser.add_argument("--spr-run", type=Path, default=REPO_ROOT / "results" / spec.spr_run)
    parser.add_argument("--baseline-run", type=Path, help="Optional relocated original result directory.")
    parser.add_argument("--results-dir", type=Path, help="Default: a new timestamped ablation directory.")
    args = parser.parse_args()
    spr = args.spr_run.resolve()
    config, saved, frames, baseline, provenance = shared.load_inputs(spec, spr, args.baseline_run)
    beta = config["spr_beta"]
    print(f"{spec.slug}: fixed original ROI={config['roi_fraction']}; beta={beta}; four cached views.", flush=True)
    validation = frames["validation"]
    validation_evidence = region_evidence(validation)
    calibrations = {}
    for variant, evidence in validation_evidence.items():
        fitted = fit_variant(validation, evidence, beta, spec.threshold_policy)
        shared.check_close(fitted["q0"], saved["calibration"]["q0"], "Original normal q95 differs from SPR.")
        calibrations[variant] = fitted
        validation[f"evidence_{variant}"] = evidence
        validation[variant] = combine(validation.original_fused.to_numpy(), evidence, fitted)
    output = args.results_dir or REPO_ROOT / "results" / f"{spec.slug}_spr_ablation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    shared.write_json(output / "calibration.json", {
        "variants": calibrations, "saved_complete_spr": saved["calibration"],
        "validation_manifest_sha256": config["manifest_sha256"]["validation"],
    })
    print("Both scales and thresholds fixed on validation. Computing test results...", flush=True)
    test = frames["test"]
    metrics = dict(saved["metrics"])
    for variant, evidence in region_evidence(test).items():
        test[f"evidence_{variant}"] = evidence
        test[variant] = combine(test.original_fused.to_numpy(), evidence, calibrations[variant])
        metrics[variant] = shared.metrics_at_threshold(
            test.label.to_numpy(), test[variant].to_numpy(), calibrations[variant]["threshold"],
        )
    names = ("original_lcf_cltc", "mean_no_penalty", "median_no_penalty", "lcf_cltc_spr")
    columns = {"metric": shared.METRICS}
    columns.update({name: [metrics[name][key] for key in shared.METRICS] for name in names})
    deltas = {}
    for variant in VARIANTS:
        deltas[variant] = {
            key: 100 * (metrics["lcf_cltc_spr"][key] - metrics[variant][key]) for key in shared.METRICS
        }
        columns[f"full_minus_{variant}_pp"] = [deltas[variant][key] for key in shared.METRICS]
    columns["median_minus_mean_pp"] = [
        100 * (metrics["median_no_penalty"][key] - metrics["mean_no_penalty"][key]) for key in shared.METRICS
    ]
    comparison = pd.DataFrame(columns)
    comparison.to_csv(output / "comparison.csv", index=False)
    for split, frame in frames.items():
        frame.to_csv(output / f"{split}_scores.csv", index=False)
    shared.write_json(output / "config.json", {
        "dataset": spec.slug, "spr_run": str(spr), "baseline_run": str(baseline),
        "source_config": config["source_config"], "beta": beta, "roi_fraction": config["roi_fraction"],
        "views": list(shared.VIEWS), "protocol": spec.protocol,
        "variants": {
            "median_no_penalty": "Archived same-region median, with no MAD penalty.",
            "mean_no_penalty": "FP32 mean of the same four region scores, with no MAD penalty.",
        },
        "fusion": "(1-beta)*original_fused + beta*q0*evidence/qe",
        "region_stat_check": {"atol": 1e-7, "rtol": 2e-6},
        "checkpoint_sha256": config["checkpoint_sha256"], "manifest_sha256": config["manifest_sha256"],
        "input_file_sha256": provenance,
    })
    shared.write_json(output / "metrics.json", {
        "metrics": metrics, "calibration": calibrations, "protocol": spec.protocol,
        "full_minus_ablation_percentage_points": deltas,
        "manifest_sha256": config["manifest_sha256"],
        "network_views_per_image": {name: 1 if name == "original_lcf_cltc" else 4 for name in names},
        "additional_network_forwards": 0,
        "note": (
            "Same frozen checkpoint, views, original ROI and saved beta; each alternative calibrated on validation. "
            "Full vs median isolates the MAD penalty; median vs mean compares aggregation without the penalty."
        ),
    })
    print(comparison.to_string(index=False))
    print("Positive full-minus differences favor complete SPR, except HTER where negative favors it.")
    print(f"Outputs: {output.resolve()}")
