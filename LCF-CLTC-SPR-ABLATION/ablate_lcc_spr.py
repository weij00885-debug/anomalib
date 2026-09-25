# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run both cached same-region SPR ablations for lcc."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "lcc",
            "LCC_colors",
            "lcc_lcf_cltc_spr_20260925_092943_946978",
            "legacy_normal_holdout_from_test_val",
            "validation_hter",
        ),
    )
