# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Layer-Conditional Fusion (LCF) for the Dinomaly model.

This module implements a drop-in replacement for Dinomaly's
``_fuse_feature = mean()`` that uses **per-patch attention** over a group of
DINOv2 layers. Each patch token learns its own attention weights over the input
layers, so the fusion becomes **input-conditioned** rather than a fixed average.

Why this matters for face anti-spoofing
---------------------------------------
FAS anomalies live at different layer granularities of DINOv2:

  - high-frequency texture anomalies (paper grain, screen pixels): blocks 2-3
  - mid-frequency structural anomalies (mask edges, cut-outs):     blocks 5-6
  - low-frequency semantic anomalies (deepfake, identity shifts): blocks 8-9

Plain ``mean()`` averages these signals with equal weight, which dilutes
subtle local anomalies (e.g. paper-edge seams). LCF lets the model attend more
to whichever layer carries the strongest anomaly signal **per patch**.

Usage
-----
LCF is created by ``DinomalyModel`` when ``use_lcf=True`` and is called from
``DinomalyModel._fuse_feature`` instead of the original static method.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerConditionalFusion(nn.Module):
    """Per-patch attention over a group of encoder/decoder layers.

    The number of fused layers is determined dynamically from the input
    list, so the same module instance works for both the "fuse all N
    encoder layers" call (e.g. ``N=8``) and the per-group "fuse M layers"
    calls (e.g. ``M=4``).

    Args:
        embed_dim (int): Embedding dimension (e.g. ``768`` for vit_base,
            ``1024`` for vit_large, ``384`` for vit_small).
        dropout (float): Dropout rate applied to the attention weights.

    Example:
        >>> lcf = LayerConditionalFusion(embed_dim=768)
        >>> feat_list = [torch.randn(2, 784, 768) for _ in range(4)]
        >>> out = lcf(feat_list)
        >>> out.shape
        torch.Size([2, 784, 768])
    """

    def __init__(self, embed_dim: int = 768, dropout: float = 0.0) -> None:
        super().__init__()
        # The number of fused layers is determined dynamically from the input
        # list length in ``forward``; the same instance works for both the
        # step-3 "fuse all N layers" call and the step-6 "fuse per-group M
        # layers" calls without any hint or warning.
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Xavier init matches DinomalyMLP/LinearAttention style.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        """Fuse a list of per-layer features with input-conditioned attention.

        Args:
            feat_list: list of ``L`` tensors, each of shape ``(B, N, D)``,
                where ``L`` is inferred from the list length.

        Returns:
            Tensor of shape ``(B, N, D)`` that has fused information from
            all ``L`` layers with per-patch, input-dependent layer weights.
        """
        if len(feat_list) < 1:
            msg = f"LayerConditionalFusion expects >=1 layers, got {len(feat_list)}."
            raise ValueError(msg)
        # (B, N, L, D)
        stacked = torch.stack(feat_list, dim=2)
        _, _, _, D = stacked.shape

        # Input-conditioned per-patch query: mean across layers as a summary
        avg_feat = stacked.mean(dim=2)             # (B, N, D)
        q = self.query_proj(avg_feat).unsqueeze(2)  # (B, N, 1, D)

        # Layer-wise keys and values
        k = self.key_proj(stacked)  # (B, N, L, D)
        v = self.value_proj(stacked)  # (B, N, L, D)

        # Attention scores: (B, N, 1, D) x (B, N, D, L) -> (B, N, 1, L)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)             # (B, N, 1, L)
        attn = self.dropout(attn)

        # Weighted sum: (B, N, 1, L) x (B, N, L, D) -> (B, N, 1, D)
        out = torch.matmul(attn, v).squeeze(2)     # (B, N, D)
        return self.out_proj(out)


__all__ = ["LayerConditionalFusion"]
