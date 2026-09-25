# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SPR using the saved OULU LCF+CLTC checkpoint and legacy manifests."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "oulu",
            "OULU-NPU",
            "oulu_lcf_cltc_legacy_s42",
            "legacy_lcf_from_test_20pct_frame_level_folder",
            "validation_f1",
        ),
    )
