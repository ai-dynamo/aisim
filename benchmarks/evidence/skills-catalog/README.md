<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Catalog skill repair evidence

This is **local, source-only, synthetic consumer evidence** for PR #937.
It is not a native prediction, an agent evaluation, a GPU measurement or hosted
CI. No numerical accuracy or winning deployment is claimed. The historical
observed-result claims were removed from the skills and eval ground truths;
workload inputs, user requirements and illustrative arithmetic remain.

## Producer and fixtures

`producer-contract.json` records the metric whitelist inspected in
[`release/0.14.0` at 77d82652b2f83fd67afefc3141d69a5f06e630be](https://github.com/ai-dynamo/aisim/blob/77d82652b2f83fd67afefc3141d69a5f06e630be/python/aisim/src/aisim/sweeper/score.py).
The release has no `aiperf_metrics.py` module and its candidate metrics contain
neither p90 field, `num_requests`, nor `mean_itl_ms`. Both summarizers now use
only whitelisted metrics. The request count in the every-request E2E eval is
compared to the configured 320 requests, not an absent candidate metric.

`consumer-fixture.json` is hand-authored synthetic data with explicit expected
fields. It contains a prediction-report shape and a corresponding recommendation
ledger. Its values do not come from a model or hardware benchmark. Tests run
both summarizer copies in text and JSON modes, feed the prediction report
through the checked-out `make_candidate` producer, test export and malformed
artifacts, and exercise ladder classification and headroom JSON parsing.
The synthetic configuration labels are not runtime/backend validation evidence.

The existing `mixed-lengths.jsonl` is a synthetic alternating-length trace.
Its deterministic recipe was reconstructed from the committed bytes; it is not
a production trace. Regeneration and anomaly verification are available with:

```bash
python3 skills/aisim-deployment-search/evals/fixtures/generate_mixed_lengths.py
```

The adjacent `mixed-lengths.summary.json` records the SHA-256, 200 records,
timestamp range/step, length distributions, hash-ID range, uniqueness and
anomaly counts. `--write` regenerates the fixture and summary together.

## Reproduce local checks

Use a Python environment with pytest, pytest-asyncio, pydantic, PyYAML and NumPy.
Set `PYTHON` to its interpreter and `SCRATCH` to a fresh directory outside the
checkout. These tests do not import or build the native runtime.

```bash
export PYTHONPATH=python/aisim/src
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -m pytest tests/sweeper/test_catalog_size_ladder.py \
  tests/sweeper/test_catalog_recommendation_summary.py tests/sweeper/test_score.py \
  -q -p no:cacheprovider
"$PYTHON" benchmarks/evidence/skills-catalog/collect_evidence.py \
  --scratch "$SCRATCH/historical-controls" > "$SCRATCH/behavior-evidence.json"
```

`behavior-evidence.json` saves both SLA checkers' results and before/after test
outputs for bdb0a4a0 (arrival windows), 92c2564e (probe timeout), 16da056e
(effective override validation), and f49e7925 (first-budget validation).
Historical failures are intentional negative controls: the same current tests
fail on the pre-fix script and pass on the repaired script. Reruns require a
fresh scratch directory. Each archived script is read with `git show`; no
published history is rewritten.

The fallback and AISim's `aggregate_sla_violations` agree on expected verdicts
at inclusive bounds, just above bounds, and with absent/unqualified numeric
samples. Diagnostic wording differs for missing or zero samples. The fallback
also rejects numeric strings and booleans; these are outside the numeric
prediction-report contract. Agreement validates SLA classification only, not
the accuracy of predictions. Additional tests cover non-finite values.

`local-validation.json` captures command output, environment and the source
commit tested. It distinguishes local checks and fake-runner controls from
hosted CI. The first development test run had 15 test-setup failures because
its new oracle case combined mutually exclusive E2E and TTFT/TPOT bounds. The
case now constructs valid SLA modes and passes; production validation was not
weakened. Initial lint findings in new files were corrected with `pairwise`
and formatting. The system Ruff 0.15.4 also flagged unchanged files; the
repository-pinned Ruff 0.14.1 passes the full paths. A missing documentation
parser dependency in the isolated environment was installed and its check
rerun successfully. Full development logs are retained in the repair run's scratch
directory.

## Remaining external evidence

On 2026-11-19 UTC, `https://pypi.org/pypi/aisim/0.14.0/json` returned HTTP 404.
The agent eval suites have not run against that public wheel. Publishing the
release and running the catalog eval/signing workflow remain external work;
these local tests do not substitute for them. No eval-result file is invented.

The live PR description still claims that the skills were exercised end to end
on CPU. This repair cannot edit PR metadata. The supervising worker or maintainer
must remove that claim or supply its original exact commands and artifacts.
Final-head NVSkills/Full CI and fresh technical reviews/reapproval belong to the
controller; earlier green Fast CI is evidence only for its own commit.
