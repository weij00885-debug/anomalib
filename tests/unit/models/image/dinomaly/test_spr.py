# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline SPR geometry, calibration and saved-experiment evaluation checks."""

import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from tests.unit.models.image.dinomaly.test_cltc import FakeEncoder, make_model
from torchvision.transforms.v2 import Resize

from anomalib.data.utils import read_image
from anomalib.models.image.dinomaly.components.perturbation_recheck import SameRegionPerturbationRecheck
from anomalib.models.image.dinomaly.lightning_model import Dinomaly
from anomalib.models.image.dinomaly.torch_model import DinomalyModel

ROOT = Path(__file__).resolve().parents[5]
SPEC = importlib.util.spec_from_file_location("spr_runner_for_tests", ROOT / "LCF-CLTC-SPR" / "_runner.py")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.fixture
def tiny_model() -> Iterator[Dinomaly]:
    """Use an offline four-patch model with real checkpoint serialization."""
    config = {"embed_dim": 16, "num_heads": 2, "target_layers": [0, 1, 2, 3]}
    with (
        patch("anomalib.models.image.dinomaly.torch_model.TimmFeatureExtractor", FakeEncoder),
        patch.object(DinomalyModel, "_get_architecture_config", return_value=config),
    ):
        torch.manual_seed(9)
        yield make_model(use_cltc=True).eval()


@pytest.mark.parametrize("use_bf16", [False, True])
def test_evidence_preserves_legacy_scores_and_single_forward(tiny_model: Dinomaly, use_bf16: bool) -> None:
    """Exporting native maps must not alter separately pooled legacy scores."""
    model = tiny_model.model
    if use_bf16:
        model.bfloat16()
        model.trajectory_head.float()
    images = torch.randn(2, 3, 28, 28)
    old = model.predict_branch_scores(images)
    with patch.object(model, "_get_outputs", wraps=model._get_outputs) as outputs:  # ruff: ignore[private-member-access]
        evidence = model.predict_evidence(images)
    assert outputs.call_count == 1
    for name in ("reconstruction", "trajectory", "fused"):
        torch.testing.assert_close(evidence[name], old[name], rtol=0, atol=0)
    assert evidence["reconstruction_patch_map"].shape == (2, 1, 2, 2)
    assert evidence["trajectory_patch_map"].shape == (2, 1, 2, 2)
    assert all(not value.requires_grad for value in evidence.values())


def test_fractional_roi_ties_and_flat_maps() -> None:
    """Ties share remaining mass rather than selecting an arbitrary location."""
    module = SameRegionPerturbationRecheck(0.5)
    source = torch.tensor([[[[4.0, 3.0, 3.0, 1.0]]], [[[2.0, 2.0, 2.0, 2.0]]]])
    weights = module.make_roi(source)
    torch.testing.assert_close(weights, torch.tensor([[[[1.0, 0.5, 0.5, 0.0]]], [[[0.5, 0.5, 0.5, 0.5]]]]))
    assert len(list(module.parameters())) == 0


def test_flip_alignment_and_fixed_roi() -> None:
    """A flipped peak must return to the original ROI; later peaks cannot reselect it."""
    module = SameRegionPerturbationRecheck(0.25)
    original = torch.tensor([[[[9.0, 1.0, 0.0, 0.0]]]])
    weights = module.make_roi(original)
    flipped = original.flip(-1)
    torch.testing.assert_close(module.region_score(flipped.flip(-1), weights), torch.tensor([9.0]))
    torch.testing.assert_close(module.region_score(flipped, weights), torch.tensor([0.0]))
    other_peak = torch.tensor([[[[2.0, 100.0, 0.0, 0.0]]]])
    torch.testing.assert_close(module.region_score(other_peak, weights), torch.tensor([2.0]))


def test_even_median_mad_and_zero_evidence() -> None:
    """Even medians average the central pair and zeros remain finite."""
    result = SameRegionPerturbationRecheck()(torch.tensor([[1.0, 3.0, 5.0, 7.0], [0.0, 0.0, 0.0, 0.0]]))
    torch.testing.assert_close(result["median"], torch.tensor([4.0, 0.0]))
    torch.testing.assert_close(result["mad"], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(result["evidence"], torch.tensor([8 / 3, 0.0]))


def test_gamma_domain_and_input_unchanged() -> None:
    """Gamma acts on RGB intensity, and views do not mutate the original batch."""
    images = torch.tensor([[[[0.0, 0.25, 1.0]]]])
    original = images.clone()
    torch.testing.assert_close(runner.apply_view(images, "gamma09"), images.pow(0.9))
    torch.testing.assert_close(runner.apply_view(images, "gamma11"), images.pow(1.1))
    torch.testing.assert_close(images, original)


def test_visible_source_options_reject_checkpoint_mismatch() -> None:
    """Displayed training arguments must never silently override a saved checkpoint."""
    config = {
        "max_steps": 5000,
        "train_batch_size": 2,
        "model_precision": "float32",
        "seed": 42,
        "cltc_loss_weight": 0.1,
        "cltc_score_weight": 0.5,
    }
    options = Namespace(**{f"source_{key}": value for key, value in config.items()})
    runner.check_source_config(config, options)
    options.source_max_steps = 10000
    with pytest.raises(ValueError, match="do not retrain"):
        runner.check_source_config(config, options)


def test_calibration_uses_normals_and_beta_zero_is_identity() -> None:
    """Attack evidence does not set q95; disabling C preserves the archived baseline."""
    validation = pd.DataFrame({
        "label": [0, 0, 1, 1],
        "original_fused": [1.0, 2.0, 3.0, 4.0],
        "spr_evidence": [2.0, 4.0, 100.0, 200.0],
    })
    calibration = runner.fit_calibration(validation, 0, "validation_hter")
    assert calibration["q0"] == pytest.approx(1.95)
    assert calibration["qe"] == pytest.approx(3.9)
    np.testing.assert_array_equal(runner.combine_spr(validation, calibration), validation.original_fused.to_numpy())
    validation.loc[validation.label == 0, "spr_evidence"] = 0
    with pytest.raises(ValueError, match="degenerate"):
        runner.fit_calibration(validation, 0.25, "validation_hter")


def test_hter_threshold_matches_legacy_ties() -> None:
    """Vectorized calibration keeps the old strict-greater-than and highest-tie rules."""
    rng = np.random.default_rng(42)
    labels = np.array([0] * 17 + [1] * 23)
    for _ in range(20):
        scores = rng.integers(0, 8, size=len(labels)).astype(float)
        best, expected = float("inf"), None
        for threshold in np.unique(scores):
            prediction = scores > threshold
            value = (prediction[labels == 0].mean() + (~prediction[labels == 1]).mean()) / 2
            if value <= best:
                best, expected = value, threshold
        assert runner.choose_threshold(labels, scores, "validation_hter") == expected


def test_original_score_drift_is_rejected() -> None:
    """A mismatched original forward cannot silently become an SPR gain."""
    with pytest.raises(RuntimeError, match="differ from the saved"):
        runner.assert_original_matches(np.array([2.0]), np.array([1.0]), 1e-4, 1e-3)


def test_archive_join_uses_paths_and_rejects_mismatched_labels(tmp_path: Path) -> None:
    """Archived prediction rows may be ordered differently from the manifest."""
    frame = pd.DataFrame({"image_path": ["real.png", "attack.png"], "label_index": [0, 1]})
    archived = pd.DataFrame({"image_path": ["attack.png", "real.png"], "label": [1, 0], "fused": [4.0, 2.0]})
    archived.to_csv(tmp_path / "test_scores.csv", index=False)
    runner.attach_archived_scores(frame, tmp_path, "test")
    assert frame.original_fused.tolist() == [2.0, 4.0]
    archived.loc[0, "label"] = 0
    archived.to_csv(tmp_path / "test_scores.csv", index=False)
    with pytest.raises(ValueError, match="labels differ"):
        runner.attach_archived_scores(frame, tmp_path, "test")


def test_saved_run_end_to_end_without_training(
    tiny_model: Dinomaly, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load a tiny checkpoint, reuse manifests/archives, calibrate, and export only C vs original."""
    source, output = tmp_path / "source", tmp_path / "spr"
    source.mkdir()
    model = tiny_model
    model.model.trajectory_head.fit_score_scales(torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
    model.pre_processor = runner.build_pre_processor(28, None)
    spec = runner.DatasetSpec("lcc", "LCC_colors", "unused", "legacy_normal_holdout_from_test_val", "validation_hter")
    config = {
        "dataset": "LCC_colors",
        "mode": "rgb",
        "data_root": str(tmp_path),
        "protocol": spec.protocol,
        "use_lcf": True,
        "use_cltc": True,
        "cltc_detach": True,
        "encoder_name": model.hparams.encoder_name,
        "cltc_hidden_dim": 128,
        "cltc_score_weight": 1.0,
        "cltc_loss_weight": 0.1,
        "model_precision": "float32",
        "image_size": 28,
        "seed": 9,
    }
    runner.write_json(source / "config.json", config)
    hashes, archived_frames = {}, {}
    for number, split in enumerate(("train", "validation", "test")):
        records, score_records = [], []
        for index, label in enumerate([0] if split == "train" else [0, 0, 1, 1]):
            path = tmp_path / f"{split}_{index}.png"
            rgb = np.random.default_rng(number * 7 + index).integers(0, 255, (35, 47, 3), dtype=np.uint8)
            Image.fromarray(rgb).save(path)
            records.append({"image_path": str(path), "label_index": label})
            if split != "train":
                tensor = Resize((28, 28), antialias=True)(read_image(path, as_tensor=True)).unsqueeze(0)
                score = model.model.predict_branch_scores(model.pre_processor(tensor))["fused"].item()
                score_records.append({"image_path": str(path), "label": label, "fused": score})
        content = pd.DataFrame(records).to_csv(index=False)
        (source / f"{split}_manifest.csv").write_text(content, encoding="utf-8")
        hashes[split] = hashlib.sha256(content.encode()).hexdigest()
        if score_records:
            archived_frames[split] = pd.DataFrame(score_records)
            archived_frames[split].to_csv(source / f"{split}_scores.csv", index=False)
    val, test = archived_frames["validation"], archived_frames["test"]
    threshold = runner.choose_threshold(val.label.to_numpy(), val.fused.to_numpy(), spec.threshold_policy)
    original_metrics = runner.metrics_at_threshold(test.label.to_numpy(), test.fused.to_numpy(), threshold)
    runner.write_json(
        source / "metrics.json",
        {"metrics": original_metrics, "protocol": spec.protocol, "manifest_sha256": hashes, "score_scales": [2.0, 4.0]},
    )
    torch.save(
        {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams)}, source / "calibrated_last.ckpt",
    )
    before = runner.file_sha256(source / "calibrated_last.ckpt")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval",
            "--source-run",
            str(source),
            "--results-dir",
            str(output),
            "--device",
            "cpu",
            "--num-workers",
            "0",
            "--eval-batch-size",
            "2",
        ],
    )
    runner.run(spec)
    report = json.loads((output / "metrics.json").read_text())
    assert set(report["metrics"]) == {"original_lcf_cltc", "lcf_cltc_spr"}
    assert report["metrics"]["original_lcf_cltc"] == original_metrics
    assert report["runtime"]["test"]["network_image_forwards"] == 16
    assert report["runtime"]["test"]["max_abs_original_score_error"] < 1e-4
    assert (output / "comparison.csv").is_file()
    result = pd.read_csv(output / "test_scores.csv")
    assert set(result.image_path) == set(test.image_path)
    assert np.isfinite(result.spr.to_numpy()).all()
    assert runner.file_sha256(source / "calibrated_last.ckpt") == before
