# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Components module for Dinomaly model.

This module provides all the necessary components for the Dinomaly Vision Transformer
architecture including layers, utilities, and vision transformer implementations.
"""

# Layer components
from .layers import Block, DinomalyMLP, LinearAttention, MemEffAttention

# Training-related classes: Loss, Optimizer and scheduler
from .lcf import LayerConditionalFusion
from .loss import CosineHardMiningLoss
from .optimizer import StableAdamW, WarmCosineScheduler
from .trajectory_head import CrossLayerTrajectoryHead

__all__ = [
    "CrossLayerTrajectoryHead",
    # Layers
    "Block",
    "DinomalyMLP",
    "LinearAttention",
    "MemEffAttention",
    # LCF (Layer-Conditional Fusion) - Dinomaly fused-feature attention
    "LayerConditionalFusion",
    # Utils
    "StableAdamW",
    "WarmCosineScheduler",
    "CosineHardMiningLoss",
]
