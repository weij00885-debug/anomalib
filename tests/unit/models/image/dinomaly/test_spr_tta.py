# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for cached TTA comparison, provenance and validation-only thresholds."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("spr_tta_runner_tests", ROOT / "LCF-CLTC-SPR-TTA" / "_runner.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)
ABLATION_SPEC = importlib.util.spec_from_file_location(
    "spr_ablation_runner_tests", ROOT / "LCF-CLTC-SPR-ABLATION" / "_runner.py",
)
ablation = importlib.util.module_from_spec(ABLATION_SPEC)
sys.modules[ABLATION_SPEC.name] = ablation
ABLATION_SPEC.loader.exec_module(ablation)


def test_equal_view_weights() -> None:
    """All four views contribute equally, independent of cached C scores."""
    frame = pd.DataFrame({
        key: [value, value * 10] for key, value in zip(runner.VIEW_COLUMNS, (1, 2, 4, 9), strict=True)
    })
    frame["spr"] = [100, -100]
    np.testing.assert_array_equal(runner.mean_tta(frame), [4, 40])


def test_matched_tta_scales_use_validation_normals_and_beta_zero_restores_original() -> None:
    """Attack magnitudes cannot set scales; disabling the mixture preserves every original score."""
    validation = pd.DataFrame({
        "label": [0, 0, 1, 1], "original_fused": [1.0, 2.0, 3.0, 4.0],
        "tta_mean": [2.0, 4.0, 100.0, 200.0],
    })
    fitted = runner.fit_blended_tta(validation, 0.25, "validation_hter")
    assert fitted["q0"] == pytest.approx(1.95)
    assert fitted["q_tta"] == pytest.approx(3.9)
    np.testing.assert_allclose(runner.blend_tta(validation, fitted), [1.0, 2.0, 14.75, 28.0])
    validation.loc[validation.label == 1, "tta_mean"] *= 100
    changed = runner.fit_blended_tta(validation, 0.25, "validation_hter")
    assert changed["q0"] == fitted["q0"]
    assert changed["q_tta"] == fitted["q_tta"]
    fitted["beta"] = 0
    np.testing.assert_array_equal(runner.blend_tta(validation, fitted), validation.original_fused)


def test_matched_tta_rejects_degenerate_scale() -> None:
    """A zero normal reference must fail instead of amplifying numerical noise."""
    validation = pd.DataFrame({"label": [0, 1], "original_fused": [1.0, 2.0], "tta_mean": [0.0, 3.0]})
    with pytest.raises(ValueError, match="degenerate"):
        runner.fit_blended_tta(validation, 0.25, "validation_hter")


@pytest.mark.parametrize("policy", ["validation_hter", "validation_f1"])
def test_threshold_matches_legacy_tie_handling(policy: str) -> None:
    """Retain old policies including tied candidates and the F1 boundary convention."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])
    if policy == "validation_hter":
        best, expected = np.inf, None
        for candidate in np.unique(scores):
            prediction = scores > candidate
            hter = (prediction[labels == 0].mean() + (~prediction[labels == 1]).mean()) / 2
            if hter <= best:
                best, expected = hter, candidate
    else:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-10)
        expected = thresholds[np.argmax(values)]
    assert runner.choose_threshold(labels, scores, policy) == expected


@pytest.fixture
def archives(tmp_path: Path) -> tuple:
    """Build a completed tiny SPR archive without images, weights or model imports."""
    baseline, spr = tmp_path / "baseline", tmp_path / "spr"
    baseline.mkdir()
    spr.mkdir()
    spec = runner.DatasetSpec("lcc", "LCC_colors", "unused", "legacy_normal_holdout_from_test_val", "validation_hter")
    source = {
        "protocol": spec.protocol, "dataset": spec.dataset, "mode": "rgb", "max_steps": 5000,
        "train_batch_size": 2, "model_precision": "float32", "seed": 42,
        "cltc_loss_weight": 0.1, "cltc_score_weight": 0.5,
    }
    runner.write_json(baseline / "config.json", source)
    hashes, frames = {}, {}
    for split in ("train", "validation", "test"):
        paths = [f"/images/{split}_{i}.png" for i in range(1 if split == "train" else 4)]
        labels = [0] if split == "train" else [0, 0, 1, 1]
        manifest = pd.DataFrame({"image_path": paths, "label_index": labels})
        path = baseline / f"{split}_manifest.csv"
        manifest.to_csv(path, index=False)
        hashes[split] = hashlib.sha256(path.read_text().encode()).hexdigest()
        if split == "train":
            continue
        multiplier = 1 if split == "validation" else 10
        views = np.array([[1, 2, 1, 0], [2, 4, 3, 1], [3, 6, 5, 2], [4, 8, 7, 3]]) * multiplier
        frame = pd.DataFrame(views, columns=runner.VIEW_COLUMNS)
        frame["image_path"], frame["label"] = paths, labels
        frame["original_fused"] = views[:, 0]
        frame["spr_evidence"] = 2 * views[:, 0]
        frame["spr"] = views[:, 0]
        frame.to_csv(spr / f"{split}_scores.csv", index=False)
        frame[["image_path", "label", "original_fused"]].rename(columns={"original_fused": "fused"}).to_csv(
            baseline / f"{split}_scores.csv", index=False,
        )
        frames[split] = frame
    checkpoint_hash = "a" * 64
    calibration = {
        "beta": 0.25, "q0": 1.95, "qe": 3.9, "threshold_policy": spec.threshold_policy,
        "original_threshold": 2.0, "spr_threshold": 2.0, "checkpoint_sha256": checkpoint_hash,
        "validation_manifest_sha256": hashes["validation"],
    }
    test = frames["test"]
    metrics = runner.metrics_at_threshold(test.label.to_numpy(), test.original_fused.to_numpy(), 2.0)
    runner.write_json(baseline / "metrics.json", {
        "metrics": metrics, "protocol": spec.protocol, "manifest_sha256": hashes,
    })
    runner.write_json(spr / "config.json", {
        "source_run": str(baseline), "source_config": source, "protocol": spec.protocol,
        "views": list(runner.VIEWS), "spr_beta": 0.25, "roi_fraction": 0.1,
        "checkpoint_sha256": checkpoint_hash, "manifest_sha256": hashes,
        "baseline_atol": 1e-4, "baseline_rtol": 1e-3,
    })
    runner.write_json(spr / "calibration.json", calibration)
    runner.write_json(spr / "metrics.json", {
        "calibration": calibration, "protocol": spec.protocol, "manifest_sha256": hashes,
        "metrics": {"original_lcf_cltc": metrics, "lcf_cltc_spr": metrics},
    })
    return spec, spr, baseline


def test_complete_comparison_keeps_validation_threshold(
    archives: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test scores favor a different threshold, but the output must retain validation's 2.5."""
    spec, spr, _ = archives
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare", "--spr-run", str(spr), "--results-dir", str(output)])
    runner.run(spec)
    report = json.loads((output / "metrics.json").read_text())
    assert report["calibration"]["tta_threshold"] == pytest.approx(2.5)
    assert report["metrics"]["tta_mean_4views"]["image_Accuracy"] == pytest.approx(0.5)
    matched = report["calibration"]["matched_tta"]
    assert matched["beta"] == pytest.approx(0.25)
    assert matched["q0"] == pytest.approx(1.95)
    assert matched["q_tta"] == pytest.approx(2.425)
    assert matched["threshold"] == pytest.approx(0.75 * 2 + 0.25 * 1.95 * 2.5 / 2.425)
    assert report["metrics"]["tta_blend_4views"]["image_Accuracy"] == pytest.approx(0.5)
    assert report["additional_network_forwards"] == 0
    assert report["network_views_per_image"]["tta_mean_4views"] == 4
    assert (output / "comparison.csv").is_file()
    for path, content in before.items():
        assert Path(path).read_bytes() == content


@pytest.mark.parametrize("corruption", ["path", "label", "duplicate", "nan", "manifest"])
def test_rejects_incompatible_cached_scores(archives: tuple, corruption: str) -> None:
    """Mixed rows or changed manifests must fail before any comparison is produced."""
    spec, spr, baseline = archives
    if corruption == "manifest":
        path = baseline / "test_manifest.csv"
        path.write_text(path.read_text() + "\n")
    else:
        path = spr / "test_scores.csv"
        frame = pd.read_csv(path)
        if corruption == "path":
            frame.loc[0, "image_path"] = "/different.png"
        elif corruption == "label":
            frame.loc[0, "label"] = 1
        elif corruption == "duplicate":
            frame.loc[0, "image_path"] = frame.loc[1, "image_path"]
        else:
            frame.loc[0, runner.VIEW_COLUMNS[1]] = np.nan
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"paths|labels|Invalid|Nonfinite|manifest has changed"):
        runner.load_inputs(spec, spr, None)


def test_check_only_writes_nothing_and_does_not_compute_tta(
    archives: tuple, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Allow real-archive verification while leaving the requested experiment to the user."""
    spec, spr, _ = archives
    output = tmp_path / "must_not_exist"
    monkeypatch.setattr(
        sys, "argv", ["compare", "--spr-run", str(spr), "--results-dir", str(output), "--check-inputs-only"],
    )

    def forbidden_mean(_frame: pd.DataFrame) -> None:
        pytest.fail("Input validation must not compute TTA scores.")

    monkeypatch.setattr(runner, "mean_tta", forbidden_mean)
    runner.run(spec)
    assert not output.exists()


def test_region_ablations_use_fixed_roi_and_even_median() -> None:
    """An outlying region changes the mean but not the central-pair median."""
    frame = pd.DataFrame({key: [value] for key, value in zip(ablation.REGION_COLUMNS, (1, 3, 5, 100), strict=True)})
    frame["spr_median"], frame["spr_mad"], frame["spr_evidence"] = 4.0, 2.0, 8 / 3
    frame["view_original_fused"] = 9999.0
    evidence = ablation.region_evidence(frame)
    np.testing.assert_array_equal(evidence["median_no_penalty"], [4.0])
    np.testing.assert_array_equal(evidence["mean_no_penalty"], [27.25])
    frame["spr_median"] = 3.0
    with pytest.raises(ValueError, match="spr_median"):
        ablation.region_evidence(frame)


def test_ablation_scales_use_normals_and_zero_beta_preserves_original() -> None:
    """Changing attack evidence cannot affect normal-q95 scaling."""
    frame = pd.DataFrame({"label": [0, 0, 1, 1], "original_fused": [1.0, 2.0, 3.0, 4.0]})
    evidence = np.array([2.0, 4.0, 100.0, 200.0])
    fitted = ablation.fit_variant(frame, evidence, 0.25, "validation_hter")
    assert fitted["q0"] == pytest.approx(1.95)
    assert fitted["qe"] == pytest.approx(3.9)
    evidence[2:] *= 100
    changed = ablation.fit_variant(frame, evidence, 0.25, "validation_hter")
    assert changed["qe"] == fitted["qe"]
    fitted["beta"] = 0
    np.testing.assert_array_equal(
        ablation.combine(frame.original_fused.to_numpy(), evidence, fitted), frame.original_fused,
    )


def test_both_ablations_end_to_end_use_validation_thresholds(
    archives: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both variants retain validation thresholds despite a tenfold test-score shift."""
    spec, spr, _ = archives
    for split in ("validation", "test"):
        path = spr / f"{split}_scores.csv"
        frame = pd.read_csv(path)
        for name in ablation.REGION_COLUMNS:
            frame[name] = frame.spr_evidence
        frame["spr_median"], frame["spr_mad"] = frame.spr_evidence, 0.0
        frame.to_csv(path, index=False)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "ablations"
    monkeypatch.setattr(sys, "argv", ["ablate", "--spr-run", str(spr), "--results-dir", str(output)])
    ablation.run(spec)
    report = json.loads((output / "metrics.json").read_text())
    for name in ablation.VARIANTS:
        assert report["calibration"][name]["threshold"] == pytest.approx(2.0)
        assert report["metrics"][name]["image_Accuracy"] == pytest.approx(0.5)
        assert report["network_views_per_image"][name] == 4
    assert report["additional_network_forwards"] == 0
    assert (output / "comparison.csv").is_file()
    for path, content in before.items():
        assert Path(path).read_bytes() == content
