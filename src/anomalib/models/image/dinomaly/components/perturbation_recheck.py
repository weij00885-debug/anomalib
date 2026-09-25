# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Parameter-free same-region perturbation recheck (SPR) for frozen checkpoints."""

from math import isfinite

import torch
from torch import nn


class SameRegionPerturbationRecheck(nn.Module):
    """Select one original-image ROI and summarize aligned view evidence.

    This component has no learned parameters. The caller aligns each view map
    to the original coordinates and calibrates the resulting image evidence on
    validation normals before combining it with the legacy image score.

    Args:
        roi_fraction: Fraction of native patches selected from the original map.
        epsilon: Denominator floor for the relative view dispersion.
    """

    def __init__(self, roi_fraction: float = 0.1, epsilon: float = 1e-8) -> None:
        super().__init__()
        if not isfinite(roi_fraction) or not 0 < roi_fraction <= 1:
            msg = "roi_fraction must be finite and in (0, 1]."
            raise ValueError(msg)
        if not isfinite(epsilon) or epsilon <= 0:
            msg = "epsilon must be positive and finite."
            raise ValueError(msg)
        self.roi_fraction = roi_fraction
        self.epsilon = epsilon

    def make_roi(self, original_map: torch.Tensor) -> torch.Tensor:
        """Return fractional top-k weights, with tied values sharing the boundary.

        Args:
            original_map: Finite original evidence maps of shape ``[B, 1, H, W]``.

        Returns:
            Nonnegative FP32 weights of the same shape, summing to k per image.
            A constant map receives uniform weights, avoiding arbitrary indices.
        """
        if original_map.ndim != 4 or original_map.shape[1] != 1 or original_map.numel() == 0:
            msg = "Expected a nonempty [B, 1, H, W] original map."
            raise ValueError(msg)
        if not torch.isfinite(original_map).all():
            msg = "Original maps must be finite."
            raise ValueError(msg)
        flat = original_map.float().flatten(1)
        k = max(1, int(flat.shape[1] * self.roi_fraction))
        cutoff = flat.topk(k, dim=1).values[:, -1:]
        above = (flat > cutoff).float()
        tied = (flat == cutoff).float()
        remaining = k - above.sum(1, keepdim=True)
        weights = above + tied * remaining / tied.sum(1, keepdim=True)
        return weights.reshape_as(original_map)

    @staticmethod
    def region_score(aligned_map: torch.Tensor, roi_weights: torch.Tensor) -> torch.Tensor:
        """Average an aligned view over the ORIGINAL image's fixed ROI.

        Args:
            aligned_map: Nonnegative evidence maps with shape ``[B, 1, H, W]``.
            roi_weights: Weights created once by ``make_roi`` for the original view.

        Returns:
            FP32 region scores of shape ``[B]``.
        """
        if aligned_map.shape != roi_weights.shape:
            msg = "Aligned map and original ROI shapes must match."
            raise ValueError(msg)
        if not torch.isfinite(aligned_map).all() or not torch.isfinite(roi_weights).all():
            msg = "Maps and ROI weights must be finite."
            raise ValueError(msg)
        mass = roi_weights.flatten(1).sum(1)
        if (roi_weights < 0).any() or (mass <= 0).any():
            msg = "ROI weights must be nonnegative with positive mass."
            raise ValueError(msg)
        return (aligned_map.float() * roi_weights).flatten(1).sum(1) / mass

    def forward(self, view_region_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        """Combine fixed-region observations using median and median deviation.

        Args:
            view_region_scores: Finite nonnegative scores of shape ``[B, V]``.

        Returns:
            ``median``, ``mad`` and ``evidence`` vectors. Even-sized medians use
            the mean of the central pair, rather than PyTorch's lower median.
        """
        if view_region_scores.ndim != 2 or min(view_region_scores.shape) < 1:
            msg = "Expected nonempty [B, V] region scores."
            raise ValueError(msg)
        values = view_region_scores.float()
        if not torch.isfinite(values).all() or (values < 0).any():
            msg = "View region scores must be finite and nonnegative."
            raise ValueError(msg)
        median = torch.quantile(values, 0.5, dim=1)
        mad = torch.quantile((values - median[:, None]).abs(), 0.5, dim=1)
        evidence = median / (1 + mad / (median.abs() + self.epsilon))
        return {"median": median, "mad": mad, "evidence": evidence}
