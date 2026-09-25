# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run both cached same-region SPR ablations for oulu."""

from _runner import DatasetSpec, run

if __name__ == "__main__":
    run(
        DatasetSpec(
            "oulu",
            "OULU-NPU",
            "oulu_lcf_cltc_spr_20260925_010043_662042",
            "legacy_lcf_from_test_20pct_frame_level_folder",
            "validation_f1",
        ),
    )
