# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run detached CLTC with the established ViT-Giant Replay-Attack configuration."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            name="REPLAY_mtcnn_colors",
            slug="replay",
            encoder_name="vit_giant_patch14_reg4_dinov2",
            dedicated_normal_test=False,
        ),
    )
