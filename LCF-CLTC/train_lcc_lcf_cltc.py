# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run detached CLTC with the established ViT-Large LCC-FASD configuration."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            name="LCC_colors",
            slug="lcc",
            encoder_name="vit_large_patch14_reg4_dinov2",
            dedicated_normal_test=False,
            cltc_loss_weight=0.5,
            cltc_score_weight=0.25,
        ),
    )
