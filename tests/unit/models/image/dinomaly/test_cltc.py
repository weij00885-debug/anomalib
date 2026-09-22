# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline CLTC gradient, token handling, scoring and serialization regressions."""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import lightning.pytorch as pl
import pytest
import torch
from torch import nn
from torch.nn import functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]

from anomalib.models.image.dinomaly.components import CrossLayerTrajectoryHead
from anomalib.models.image.dinomaly.lightning_model import Dinomaly
from anomalib.models.image.dinomaly.torch_model import DinomalyModel


class FakeEncoder(nn.Module):
    """Tiny deterministic encoder with CLS and four register tokens; no downloads."""

    patch_size = 14
    num_register_tokens = 4

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(1, 17).float(), requires_grad=False)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce four depth-dependent sequences with four spatial patches."""
        patches = F.avg_pool2d(images.mean(1, keepdim=True), 14).flatten(2).transpose(1, 2)
        patches = patches * self.weight
        prefix = patches.new_full((images.shape[0], 5, 16), 10)
        tokens = torch.cat([prefix, patches], dim=1)
        return {f"blocks.{i}": tokens + self.weight.roll(i) * 0.1 for i in range(4)}


@pytest.fixture
def tiny_encoder() -> Iterator[None]:
    """Replace expensive pretrained encoder and architecture dimensions."""
    config = {"embed_dim": 16, "num_heads": 2, "target_layers": [0, 1, 2, 3]}
    with (
        patch("anomalib.models.image.dinomaly.torch_model.TimmFeatureExtractor", FakeEncoder),
        patch.object(DinomalyModel, "_get_architecture_config", return_value=config),
    ):
        yield


def make_model(**kwargs) -> Dinomaly:
    """Build a CPU-only Lightning model without evaluators or data transforms."""
    return Dinomaly(
        decoder_depth=4,
        fuse_layer_encoder=[[0, 1], [2, 3]],
        fuse_layer_decoder=[[0, 1], [2, 3]],
        bottleneck_dropout=0,
        use_lcf=True,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
        **kwargs,
    )


def test_head_target_and_detach() -> None:
    """Only head parameters get gradients; target distances match known vectors."""
    head = CrossLayerTrajectoryHead(2, 3, 4)
    a = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    b = torch.tensor([[[0.0, 1.0]]], requires_grad=True)
    target = head.build_target([a, b, -b])
    torch.testing.assert_close(target, torch.tensor([[[1.0, 2.0]]]))
    assert not target.requires_grad
    F.smooth_l1_loss(head(a), target).backward()
    assert a.grad is None
    assert b.grad is None
    assert head.predictor[-1].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("options", [{}, {"remove_class_token": True}, {"use_context_recentering": True}])
@pytest.mark.usefixtures("tiny_encoder")
def test_raw_patch_targets_and_isolated_gradients(options: dict) -> None:
    """Remove prefix tokens and retain raw targets even with recentering enabled."""
    model = make_model(use_cltc=True, **options).model
    images = torch.randn(2, 3, 28, 28)
    en, de, prediction, target = model._get_outputs(images)  # ruff: ignore[private-member-access]
    raw = list(model.encoder(images).values())
    expected = model.trajectory_head.build_target([feature[:, 5:] for feature in raw])
    assert target.shape == prediction.shape == (2, 4, 3)
    torch.testing.assert_close(target, expected)
    F.smooth_l1_loss(prediction, target).backward()
    assert all(param.grad is None for param in model.lcf.parameters())
    assert all(param.grad is None for param in model.bottleneck.parameters())
    assert all(param.grad is None for param in model.decoder.parameters())
    assert any(param.grad is not None for param in model.trajectory_head.parameters())
    model.loss_fn(en, de, global_step=1).backward()
    assert any(param.grad is not None for param in model.lcf.parameters())
    assert any(param.grad is not None for param in model.decoder.parameters())


@pytest.mark.usefixtures("tiny_encoder")
def test_baseline_initialization_scores_and_gradients_unchanged() -> None:
    """CLTC preserves the common initialization and the reconstruction gradient."""
    torch.manual_seed(21)
    baseline = make_model(use_cltc=False).model
    torch.manual_seed(21)
    extended = make_model(use_cltc=True).model
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, extended.state_dict()[name], rtol=0, atol=0)
    images = torch.randn(2, 3, 28, 28)
    baseline(images, global_step=1).backward()
    extended(images, global_step=1).backward()
    extended_parameters = dict(extended.named_parameters())
    for name, parameter in baseline.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, extended_parameters[name].grad)
    baseline.eval()
    extended.eval()
    torch.testing.assert_close(baseline(images).pred_score, extended.predict_branch_scores(images)["reconstruction"])
    assert not any("trajectory" in key for key in baseline.state_dict())


@pytest.mark.usefixtures("tiny_encoder")
def test_calibrated_scores_and_checkpoint(tmp_path: Path) -> None:
    """Calibration buffers, head weights and constructor switches survive Lightning loading."""
    model = make_model(use_cltc=True).eval()
    model.model.trajectory_head.fit_score_scales(torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
    images = torch.randn(2, 3, 28, 28)
    scores = model.model.predict_branch_scores(images)
    torch.testing.assert_close(scores["fused"], scores["reconstruction"] / 2 + scores["trajectory"] / 4)
    torch.testing.assert_close(model.model(images).pred_score, scores["fused"])
    assert model.model(images).anomaly_map.shape == (2, 1, 28, 28)
    checkpoint = tmp_path / "cltc.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    # Anomalib deliberately excludes processing/evaluator components from hparams.
    restored = Dinomaly.load_from_checkpoint(
        checkpoint, weights_only=False, pre_processor=False, post_processor=False, evaluator=False, visualizer=False,
    ).eval()
    torch.testing.assert_close(restored.model.predict_branch_scores(images)["fused"], scores["fused"])


@pytest.mark.usefixtures("tiny_encoder")
def test_head_fp32_and_optimizer_membership() -> None:
    """The BF16 base keeps an FP32 head included in the actual optimizer."""
    model = make_model(use_cltc=True, precision="float16")
    assert next(model.model.lcf.parameters()).dtype == torch.bfloat16
    assert all(
        param.dtype == torch.float32 and param.requires_grad for param in model.model.trajectory_head.parameters()
    )
    model.trainer = pl.Trainer(accelerator="cpu", max_steps=2, max_epochs=-1, logger=False, enable_checkpointing=False)
    optimizers, _ = model.configure_optimizers()
    optimized = {id(param) for group in optimizers[0].param_groups for param in group["params"]}
    assert all(id(param) in optimized for param in model.model.trajectory_head.parameters())
    loss = model.model(torch.randn(2, 3, 28, 28), global_step=1)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.usefixtures("tiny_encoder")
def test_clipping_does_not_couple_head_to_lcf() -> None:
    """Large head gradients do not reduce the baseline clipping budget."""
    model = make_model(use_cltc=True)
    optimizer = torch.optim.SGD(model.trainable_modules.parameters(), lr=0.1)
    base = list(model.model.lcf.parameters())
    for parameter in base:
        parameter.grad = torch.ones_like(parameter)
    for parameter in model.model.trajectory_head.parameters():
        parameter.grad = torch.ones_like(parameter) * 1e6
    expected = [nn.Parameter(parameter.detach().clone()) for parameter in base]
    for parameter in expected:
        parameter.grad = torch.ones_like(parameter)
    nn.utils.clip_grad_norm_(expected, 0.1)
    model.configure_gradient_clipping(optimizer, 0.1, "norm")
    for left, right in zip(base, expected, strict=True):
        torch.testing.assert_close(left.grad, right.grad)


@pytest.mark.parametrize("values", [torch.empty(0, 2), torch.ones(3), torch.tensor([[float("nan"), 1.0]])])
def test_invalid_calibration(values: torch.Tensor) -> None:
    """Reject malformed calibration rather than silently corrupting scores."""
    with pytest.raises(ValueError, match=r"Expected nonempty|finite and nonnegative"):
        CrossLayerTrajectoryHead(4, 3).fit_score_scales(values)
