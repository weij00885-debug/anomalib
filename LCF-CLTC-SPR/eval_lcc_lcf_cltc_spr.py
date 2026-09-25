# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SPR using the saved LCC LCF+CLTC checkpoint and manifests."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "lcc",
            "LCC_colors",
            "lcc_lcf_cltc_final_5k_b2_s42",
            "legacy_normal_holdout_from_test_val",
            "validation_hter",
        ),
    )
