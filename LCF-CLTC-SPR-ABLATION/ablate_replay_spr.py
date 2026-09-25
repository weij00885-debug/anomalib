# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run both cached same-region SPR ablations for replay."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "replay",
            "REPLAY_mtcnn_colors",
            "replay_lcf_cltc_spr_20260925_101212_401331",
            "legacy_normal_holdout_from_test_val",
            "validation_hter",
        ),
    )
