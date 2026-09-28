# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sweep SPR ROI fractions on frozen LCF+CLTC runs with one inference pass."""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from _runner import (
    METRIC_NAMES,
    VIEWS,
    DatasetSpec,
    ManifestImages,
    apply_view,
    assert_original_matches,
    choose_threshold,
    combine_spr,
    file_sha256,
    fit_calibration,
    load_inputs,
    metrics_at_threshold,
    restore_model,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "oulu": DatasetSpec(
        "oulu", "OULU-NPU", "oulu_lcf_cltc_legacy_s42",
        "legacy_lcf_from_test_20pct_frame_level_folder", "validation_f1",
    ),
    "lcc": DatasetSpec(
        "lcc", "LCC_colors", "lcc_lcf_cltc_final_5k_b2_s42",
        "legacy_normal_holdout_from_test_val", "validation_hter",
    ),
    "nuaa": DatasetSpec(
        "nuaa", "NUAA_mtcnn_colors", "nuaa_lcf_cltc_legacy_fp32_20k_s42",
        "legacy_dedicated_normal_test_from_test_val", "validation_hter",
    ),
    "replay": DatasetSpec(
        "replay", "REPLAY_mtcnn_colors", "replay_lcf_cltc_legacy_fp32_20k_s42",
        "legacy_normal_holdout_from_test_val", "validation_hter",
    ),
}
DEFAULT_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare several SPR ROI fractions with one four-view inference pass per split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", choices=tuple(SPECS), default=list(SPECS))
    parser.add_argument("--roi-fractions", nargs="+", type=float, default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--spr-beta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--baseline-atol", type=float, default=1e-4)
    parser.add_argument("--baseline-rtol", type=float, default=1e-3)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--oulu-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    if not args.roi_fractions or any(not np.isfinite(x) or not 0 < x <= 1 for x in args.roi_fractions):
        parser.error("ROI fractions must be finite values in (0, 1].")
    if len(set(args.roi_fractions)) != len(args.roi_fractions):
        parser.error("ROI fractions must be unique.")
    if not np.isfinite(args.spr_beta) or not 0 <= args.spr_beta <= 1:
        parser.error("--spr-beta must be finite and in [0, 1].")
    if args.eval_batch_size <= 0 or args.num_workers < 0:
        parser.error("Batch size must be positive and workers nonnegative.")
    if any(not np.isfinite(x) or x < 0 for x in (args.baseline_atol, args.baseline_rtol)):
        parser.error("Baseline tolerances must be finite and nonnegative.")
    return args


def make_runner_args(spec: DatasetSpec, args: argparse.Namespace) -> SimpleNamespace:
    """Build the small namespace consumed by the existing source-run validator."""
    return SimpleNamespace(
        source_run=REPO_ROOT / "results" / spec.source_run,
        source_max_steps=None,
        source_train_batch_size=None,
        source_model_precision=None,
        source_seed=None,
        source_cltc_loss_weight=None,
        source_cltc_score_weight=None,
        root=args.oulu_root if spec.slug == "oulu" else None,
        data_root=args.data_root if spec.slug != "oulu" else None,
        baseline_atol=args.baseline_atol,
        baseline_rtol=args.baseline_rtol,
    )


@torch.inference_mode()
def collect_roi_sweep(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    config: dict,
    args: argparse.Namespace,
    runner_args: SimpleNamespace,
    fractions: tuple[float, ...],
    split: str,
) -> tuple[pd.DataFrame, dict]:
    """Run each view once, then score every ROI fraction from those same maps."""
    from anomalib.models.image.dinomaly.components.perturbation_recheck import SameRegionPerturbationRecheck

    # Reuse the legacy dataset's CPU antialiased resize, exactly as _runner does.
    loader = DataLoader(
        ManifestImages(frame, config["image_size"]),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=model.device.type == "cuda",
    )
    modules = {
        fraction: SameRegionPerturbationRecheck(fraction).eval()
        for fraction in fractions
    }
    q_rec, q_traj = model.model.trajectory_head.score_scales
    alpha = model.model.cltc_score_weight
    rows: list[dict] = []
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
        torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()

    for raw_images, indices in tqdm(loader, desc=f"{split}: 4 views, {len(fractions)} ROIs", unit="batch"):
        images = raw_images.to(model.device, non_blocking=True)
        view_maps: dict[str, torch.Tensor] = {}
        view_scores: dict[str, np.ndarray] = {}
        original_map = None
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
                assert_original_matches(
                    view_scores[name], archived, runner_args.baseline_atol, runner_args.baseline_rtol,
                )
                original_map = fused_map
            if name == "flip":
                fused_map = fused_map.flip(-1)
            view_maps[name] = fused_map

        if original_map is None:
            raise RuntimeError("Original view did not produce a patch map.")
        batch_values: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for fraction, module in modules.items():
            roi = module.make_roi(original_map)
            region_scores = torch.stack(
                [module.region_score(view_maps[name], roi) for name in VIEWS], dim=1,
            )
            summary = module(region_scores)
            batch_values[fraction] = (
                region_scores.cpu().numpy(),
                summary["evidence"].cpu().numpy(),
                summary["median"].cpu().numpy(),
                summary["mad"].cpu().numpy(),
            )

        for offset, index in enumerate(indices.tolist()):
            item = frame.iloc[index]
            row = {
                "image_path": item.image_path,
                "resolved_image_path": item.resolved_image_path,
                "label": int(item.label_index),
                "original_fused": float(item.original_fused),
                "original_forward": float(view_scores["original"][offset]),
            }
            for name in VIEWS:
                row[f"view_{name}_fused"] = float(view_scores[name][offset])
            for fraction, values in batch_values.items():
                region, evidence, median, mad = values
                pct = f"{100 * fraction:g}"
                row[f"spr_roi_{pct}_evidence"] = float(evidence[offset])
                row[f"spr_roi_{pct}_median"] = float(median[offset])
                row[f"spr_roi_{pct}_mad"] = float(mad[offset])
                for j, name in enumerate(VIEWS):
                    row[f"region_{name}_roi_{pct}"] = float(region[offset, j])
            rows.append(row)

    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    result = pd.DataFrame(rows)
    numeric = result.select_dtypes(include="number").to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite SPR evidence in {split}.")
    cost = {
        "images": len(result),
        "network_forward_batches": len(loader) * len(VIEWS),
        "network_image_forwards": len(result) * len(VIEWS),
        "views_per_image": len(VIEWS),
        "roi_fractions_scored_per_forward": list(fractions),
        "wall_seconds_including_io": elapsed,
        "seconds_per_image_including_io": elapsed / len(result),
        "max_abs_original_score_error": float((result.original_forward - result.original_fused).abs().max()),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device)
        if model.device.type == "cuda" else None,
    }
    return result, cost


def run_dataset(spec: DatasetSpec, args: argparse.Namespace, output: Path) -> list[dict]:
    runner_args = make_runner_args(spec, args)
    config, archived_report, manifests, hashes = load_inputs(spec, runner_args)
    fractions = tuple(args.roi_fractions)
    print(
        f"\n[{spec.slug}] source={runner_args.source_run}; beta={args.spr_beta}; "
        f"ROIs={[f'{100 * x:g}%' for x in fractions]}; seed={config['seed']}; "
        f"precision={config['model_precision']}; CLTC alpha={config['cltc_score_weight']}",
        flush=True,
    )
    val, test = manifests["validation"], manifests["test"]
    baseline_threshold = choose_threshold(
        val.label_index.to_numpy(), val.original_fused.to_numpy(), spec.threshold_policy,
    )
    baseline_metrics = metrics_at_threshold(
        test.label_index.to_numpy(), test.original_fused.to_numpy(), baseline_threshold,
    )
    if any(
        not np.isclose(baseline_metrics[key], archived_report["metrics"][key], atol=1e-6, rtol=0)
        for key in METRIC_NAMES
    ):
        raise ValueError(f"{spec.slug}: archived metrics do not match saved scores and threshold policy.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; use the working torch_env.")
    checkpoint_hash = file_sha256(runner_args.source_run / "calibrated_last.ckpt")
    model = restore_model(runner_args.source_run, config, device)
    scales = model.model.trajectory_head.score_scales.detach().cpu().numpy()
    if not np.allclose(scales, archived_report["score_scales"], rtol=1e-6, atol=1e-8) or (scales <= 0).any():
        raise ValueError(f"{spec.slug}: checkpoint branch calibration differs from archived report.")

    validation, validation_cost = collect_roi_sweep(
        model, val, config, args, runner_args, fractions, "validation",
    )
    test_scores, test_cost = collect_roi_sweep(
        model, test, config, args, runner_args, fractions, "test",
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    calibrations: dict[str, dict] = {}
    rows: list[dict] = []
    for fraction in fractions:
        pct = f"{100 * fraction:g}"
        val_for_fit = validation[["label", "original_fused", f"spr_roi_{pct}_evidence"]].rename(
            columns={f"spr_roi_{pct}_evidence": "spr_evidence"},
        )
        calibration = fit_calibration(val_for_fit, args.spr_beta, spec.threshold_policy)
        calibration.update(
            roi_fraction=fraction,
            checkpoint_sha256=checkpoint_hash,
            validation_manifest_sha256=hashes["validation"],
            score_scales=scales.tolist(),
        )
        calibrations[pct] = calibration
        validation[f"spr_roi_{pct}"] = combine_spr(
            val_for_fit.assign(spr_evidence=validation[f"spr_roi_{pct}_evidence"].to_numpy()), calibration,
        )
        test_scores[f"spr_roi_{pct}"] = combine_spr(
            test_scores.assign(spr_evidence=test_scores[f"spr_roi_{pct}_evidence"].to_numpy()), calibration,
        )
        val_metric = metrics_at_threshold(
            validation.label.to_numpy(), validation[f"spr_roi_{pct}"].to_numpy(), calibration["spr_threshold"],
        )
        test_metric = metrics_at_threshold(
            test_scores.label.to_numpy(), test_scores[f"spr_roi_{pct}"].to_numpy(), calibration["spr_threshold"],
        )
        rows.append({
            "dataset": spec.slug,
            "roi_fraction": fraction,
            "roi_percent": 100 * fraction,
            "beta": args.spr_beta,
            "validation_AUROC": val_metric["image_AUROC"],
            "test_AUROC": test_metric["image_AUROC"],
            **{f"test_{key}": value for key, value in test_metric.items()},
            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            "validation_qe": calibration["qe"],
            "validation_q0": calibration["q0"],
            "validation_threshold": calibration["spr_threshold"],
        })

    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "source_run": str(runner_args.source_run.resolve()),
        "source_config": config,
        "protocol": spec.protocol,
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": hashes,
        "views": list(VIEWS),
        "spr_beta": args.spr_beta,
        "roi_fractions": list(fractions),
        "baseline_source": "archived_scores_and_metrics",
        "training": False,
        "device": str(device),
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "baseline_atol": args.baseline_atol,
        "baseline_rtol": args.baseline_rtol,
        "root_override": str((args.oulu_root if spec.slug == "oulu" else args.data_root) or ""),
        "roi_selection_note": "All ROI results are exploratory if the test AUROC sweep is used to select ROI.",
    }
    write_json(output / "config.json", settings)
    write_json(output / "calibration.json", calibrations)
    validation.to_csv(output / "validation_scores.csv", index=False)
    test_scores.to_csv(output / "test_scores.csv", index=False)
    result = pd.DataFrame(rows)
    result.to_csv(output / "roi_metrics.csv", index=False)
    write_json(output / "runtime.json", {"validation": validation_cost, "test": test_cost})
    print(result[["roi_percent", "validation_AUROC", "test_AUROC", "test_image_HTER"]].to_string(index=False), flush=True)
    print(f"[{spec.slug}] saved: {output.resolve()}", flush=True)
    return rows


def main() -> None:
    args = parse_args()
    root = args.results_dir or REPO_ROOT / "results" / f"spr_roi_sweep_{datetime.now():%Y%m%d_%H%M%S_%f}"
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "run_config.json", {
        "datasets": args.datasets,
        "roi_fractions": args.roi_fractions,
        "spr_beta": args.spr_beta,
        "device": args.device,
        "created_local": datetime.now().astimezone().isoformat(),
    })
    all_rows: list[dict] = []
    for slug in args.datasets:
        all_rows.extend(run_dataset(SPECS[slug], args, root / slug))

    summary = pd.DataFrame(all_rows)
    summary.to_csv(root / "summary.csv", index=False)
    macro = summary.groupby("roi_fraction", as_index=False).agg(
        validation_AUROC_macro=("validation_AUROC", "mean"),
        test_AUROC_macro=("test_AUROC", "mean"),
        datasets=("dataset", "nunique"),
    )
    macro.to_csv(root / "macro_summary.csv", index=False)
    test_idx = int(macro.test_AUROC_macro.to_numpy().argmax())
    val_idx = int(macro.validation_AUROC_macro.to_numpy().argmax())
    selection = {
        "selection_scope": "Across datasets, unweighted macro-average AUROC.",
        "highest_observed_test_macro_roi_fraction": float(macro.iloc[test_idx].roi_fraction),
        "highest_observed_test_macro_auroc": float(macro.iloc[test_idx].test_AUROC_macro),
        "validation_selected_roi_fraction": float(macro.iloc[val_idx].roi_fraction),
        "validation_selected_macro_auroc": float(macro.iloc[val_idx].validation_AUROC_macro),
        "warning": (
            "Choosing ROI by test_AUROC_macro reuses the test sets for hyperparameter selection and is optimistic. "
            "Use validation_selected_roi_fraction as the methodologically safer choice; a new untouched test set "
            "is needed for an unbiased final estimate after selection."
        ),
    }
    write_json(root / "selection.json", selection)
    print("\nMacro-average across evaluated datasets:", flush=True)
    print(macro.to_string(index=False), flush=True)
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)
    print(f"\nAll results: {root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
