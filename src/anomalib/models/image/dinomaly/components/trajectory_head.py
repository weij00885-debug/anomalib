# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experimental detached cross-layer trajectory prediction for Dinomaly."""

from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F  # ruff: ignore[lowercase-imported-as-non-lowercase]


class CrossLayerTrajectoryHead(nn.Module):
    """Predict adjacent encoder-layer cosine distances from detached fused tokens.

    The target is computed from raw frozen encoder patch tokens, before context
    recentering or LCF. Neither the target nor predictor input receives gradients.
    This is a research hypothesis, not a guaranteed attack detector.

    Args:
        embed_dim (int): Input feature dimension.
        num_layers (int): Number of extracted encoder layers, at least two.
        hidden_dim (int): Hidden predictor dimension. Defaults to 128.

    Raises:
        ValueError: If dimensions are nonpositive or fewer than two layers are supplied.
    """

    def __init__(self, embed_dim: int, num_layers: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if min(embed_dim, hidden_dim) <= 0 or num_layers < 2:
            msg = "CLTC requires positive dimensions and at least two encoder layers."
            raise ValueError(msg)
        self.num_layers = num_layers
        self.predictor = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers - 1),
        )
        # Persistent global score scales, fitted on independent normal samples.
        self.register_buffer("score_scales", torch.ones(2, dtype=torch.float32))

    def build_target(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Construct a fixed target from adjacent encoder layers.

        Args:
            features (list[torch.Tensor]): L raw patch sequences of shape [B, N, C].

        Returns:
            torch.Tensor: Detached FP32 cosine distances with shape [B, N, L-1].

        Raises:
            ValueError: If the layer count differs from the configured count.
        """
        if len(features) != self.num_layers:
            msg = f"Expected {self.num_layers} layers, received {len(features)}."
            raise ValueError(msg)
        with torch.no_grad(), torch.autocast(device_type=features[0].device.type, enabled=False):
            return torch.stack(
                [
                    (1 - F.cosine_similarity(left.float(), right.float(), dim=-1)).clamp(0, 2)
                    for left, right in pairwise(features)
                ],
                dim=-1,
            )

    def forward(self, fused_patches: torch.Tensor) -> torch.Tensor:
        """Predict a trajectory without passing gradients to the fused input.

        Args:
            fused_patches (torch.Tensor): LCF patch features of shape [B, N, C].

        Returns:
            torch.Tensor: FP32 predicted distances of shape [B, N, L-1].
        """
        with torch.autocast(device_type=fused_patches.device.type, enabled=False):
            return self.predictor(fused_patches.detach().float())

    @torch.no_grad()
    def fit_score_scales(self, normal_scores: torch.Tensor) -> None:
        """Fit two q95 scales from independent normal image scores.

        Args:
            normal_scores (torch.Tensor): [M, 2] scores, reconstruction then trajectory.

        Raises:
            ValueError: If scores are empty, malformed, negative or non-finite.
        """
        if normal_scores.ndim != 2 or normal_scores.shape[1] != 2 or normal_scores.shape[0] == 0:
            msg = "Expected nonempty normal scores of shape [M, 2]."
            raise ValueError(msg)
        if not torch.isfinite(normal_scores).all() or (normal_scores < 0).any():
            msg = "Calibration scores must be finite and nonnegative."
            raise ValueError(msg)
        scales = torch.quantile(normal_scores.float(), 0.95, dim=0).clamp_min(1e-8)
        self.score_scales.copy_(scales.to(self.score_scales))
