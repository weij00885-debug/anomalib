# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Compare cached original, mean TTA and SPR results for nuaa."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "nuaa",
            "NUAA_mtcnn_colors",
            "nuaa_lcf_cltc_spr_20260924_164610_163966",
            "legacy_dedicated_normal_test_from_test_val",
            "validation_hter",
        ),
    )
