# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The refreshed record must retain auditable samples for every measured query."""

import json
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

RECORD = Path(__file__).with_name("canonical-mtp-forward-refresh-32d50c27.json")


@pytest.mark.parametrize("round_index", range(6))
def test_every_forward_summary_recomputes_from_its_raw_samples(round_index):
    data = json.loads(RECORD.read_text())
    assert len(data["rounds"]) == 6
    record = data["rounds"][round_index]
    assert record["order"] == (["baseline", "candidate"] if round_index % 2 == 0 else ["candidate", "baseline"])
    for side in ("baseline", "candidate"):
        result = record["results"][side]
        identity = result["identity"]
        assert identity["constructor_sha256"] == data["provenance"]["constructor_sha256"][side]
        assert identity["native_sha256"] == data["provenance"]["native_sha256"]
        assert set(result["builds"]) == {"nano", "control"}
        for model in result["builds"].values():
            assert set(model["queries"]) == {"decode", "mixed_ragged", "prefill_uniform", "prefill_ragged"}
            for query in model["queries"].values():
                samples = query["warm_samples_us"]
                assert len(samples) == 100
                assert all(math.isfinite(value) and value > 0 for value in samples)
                assert query["predicted_ms"] > 0
                assert query["first_query_us"] > 0
                warm = query["warm"]
                assert warm["iterations"] == 100
                assert warm["call_mean_us"] == statistics.fmean(samples)
                assert warm["call_median_us"] == statistics.median(samples)
                assert warm["call_min_us"] == min(samples)
                assert warm["call_max_us"] == max(samples)
                for percentile in (50, 90, 99):
                    assert warm[f"call_p{percentile}_us"] == pytest.approx(np.percentile(samples, percentile))
    # The non-Mamba control must return the same predictions in both builds,
    # regardless of host timing variation.
    for name, query in record["results"]["baseline"]["builds"]["control"]["queries"].items():
        assert (
            query["predicted_ms"]
            == record["results"]["candidate"]["builds"]["control"]["queries"][name]["predicted_ms"]
        )
