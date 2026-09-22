# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run detached CLTC with the established ViT-Giant NUAA configuration."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            name="NUAA_mtcnn_colors",
            slug="nuaa",
            encoder_name="vit_giant_patch14_reg4_dinov2",
            dedicated_normal_test=True,
        ),
    )
