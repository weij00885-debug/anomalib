# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SPR using the saved NUAA LCF+CLTC checkpoint and manifests."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "nuaa",
            "NUAA_mtcnn_colors",
            "nuaa_lcf_cltc_legacy_fp32_20k_s42",
            "legacy_dedicated_normal_test_from_test_val",
            "validation_hter",
        ),
    )
