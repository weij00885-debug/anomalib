# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Evaluate SPR using the saved Replay LCF+CLTC checkpoint and manifests."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "replay",
            "REPLAY_mtcnn_colors",
            "replay_lcf_cltc_legacy_fp32_20k_s42",
            "legacy_normal_holdout_from_test_val",
            "validation_hter",
        ),
    )
