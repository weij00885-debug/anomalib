# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Protocol and output checks for the OULU CLTC runner, without any training."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from train_dinomaly_lcf_cltc_oulu import check_disjoint, parse_args, summarize_scores

from anomalib.data.datasets import FolderDataset


def test_metrics_use_validation_f1_threshold() -> None:
    """Threshold comes from labeled validation and attack error is false negatives."""
    validation = pd.DataFrame({
        "image_path": ["v1", "v2", "v3", "v4"],
        "label": [0, 0, 1, 1],
        "fused": [0.1, 0.2, 0.8, 0.9],
    })
    test = pd.DataFrame({
        "image_path": ["r1", "r2", "a1", "a2"],
        "label": [0, 0, 1, 1],
        "fused": [0.1, 0.3, 0.15, 0.9],
    })
    result = summarize_scores(test, validation)
    assert result["image_Recall"] == pytest.approx(0.5)
    assert result["image_HTER"] == pytest.approx(0.25)
    assert result["image_AUROC"] == pytest.approx(0.75)


def test_manifest_overlap_is_rejected(tmp_path: Path) -> None:
    """A validation directory cannot silently alias training data."""
    Image.new("RGB", (28, 28)).save(tmp_path / "face.png")
    dataset = FolderDataset(name="normal", normal_dir=tmp_path, split="train")
    with pytest.raises(ValueError, match="overlaps train and validation"):
        check_disjoint({"train": dataset, "validation": dataset})


def test_cli_encoder_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing backbone does not leave the Giant block indices in the CLI."""
    monkeypatch.setattr(sys, "argv", ["runner", "--encoder-name", "vit_base_patch14_reg4_dinov2"])
    args = parse_args()
    assert args.target_layers == [2, 3, 4, 5, 6, 7, 8, 9]
    assert args.cltc_score_weight == pytest.approx(1.0)
