# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentX accuracy metrics: unsigned request-average and per-request quantile errors, and step errors.

Request metrics use the uniquely matched requests of one run. ``e2e`` compares mean latency,
``|mean AISim / mean silicon - 1|``; ``quantile`` averages ``|AISim pN / silicon pN - 1|`` over
p5, p10, ..., p95. Both are percentages with a bootstrap standard error over resampled requests.
Silicon TPOT under 1 ms is burst delivery, not decode pacing, so it is excluded and counted.
A zero AISim prediction is kept: replay emits zero ITL when tokens share a timestamp.

Step metrics are the MAPE of predicted versus measured forward-pass time, kept as sums so lanes pool
exactly; the standard error is std/sqrt(n) over steps.
"""

import math
import random

QUANTILES = tuple(range(5, 96, 5))
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
MIN_SILICON_TPOT_MS = 1.0
REQUEST_METRICS = ("tpot_ms", "ttft_ms")
STEP_KINDS = ("decode", "prefill")


def percentile(ordered, q):
    """Linear interpolation between closest ranks (NumPy's default) over a sorted sequence."""
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def e2e_error(silicon, predicted):
    return abs(sum(predicted) / sum(silicon) - 1) * 100


def quantile_error(silicon, predicted):
    silicon, predicted = sorted(silicon), sorted(predicted)
    return sum(abs(percentile(predicted, q) / percentile(silicon, q) - 1) for q in QUANTILES) * 100 / len(QUANTILES)


def _stderr(values):
    average = sum(values) / len(values)
    return math.sqrt(sum((v - average) ** 2 for v in values) / (len(values) - 1))


def scored_pairs(pairs, metric):
    """Matched pairs with finite, non-negative values and a positive silicon denominator.

    A zero prediction is valid (replay emits zero ITL when tokens share a timestamp). TPOT first excludes
    and counts silicon burst delivery under 1 ms, including zero, before requiring the denominator.
    """
    rows, excluded = [], 0
    for p in pairs:
        silicon, predicted = p["silicon_" + metric], p["predicted_" + metric]
        if p.get("status", "matched") != "matched" or not (_finite(silicon) and _finite(predicted)):
            continue
        if silicon < 0 or predicted < 0:
            continue
        if metric == "tpot_ms" and silicon < MIN_SILICON_TPOT_MS:
            excluded += 1
        elif p.get(metric + "_ape") is not None:
            rows.append((silicon, predicted))
    return rows, excluded


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def point_metrics(pairs, metric):
    """Point estimates only; validation recomputes these from the published pairs."""
    rows, excluded = scored_pairs(pairs, metric)
    result = {"n": len(rows), "excluded": excluded, "e2e": None, "quantile": None}
    if rows:
        silicon, predicted = [r[0] for r in rows], [r[1] for r in rows]
        result.update(e2e=e2e_error(silicon, predicted), quantile=quantile_error(silicon, predicted))
    return result


def request_metrics(pairs):
    """Per-run request metrics with bootstrap standard errors (fixed seed, deterministic)."""
    generator = random.Random(BOOTSTRAP_SEED)
    result = {}
    for metric in REQUEST_METRICS:
        rows, _ = scored_pairs(pairs, metric)
        values = point_metrics(pairs, metric)
        values.update(e2e_se=None, quantile_se=None)
        if len(rows) > 1:
            e2e, quantile = [], []
            for _ in range(BOOTSTRAP_RESAMPLES):
                sample = [rows[generator.randrange(len(rows))] for _ in rows]
                silicon, predicted = [r[0] for r in sample], [r[1] for r in sample]
                e2e.append(e2e_error(silicon, predicted))
                quantile.append(quantile_error(silicon, predicted))
            values.update(e2e_se=_stderr(e2e), quantile_se=_stderr(quantile))
        result[metric] = values
    return result


def step_accumulator(errors):
    """Pool-ready sums of absolute percentage errors."""
    return {"n": len(errors), "sum_ape": math.fsum(errors), "sum_sq_ape": math.fsum(e * e for e in errors)}


def step_summary(accumulator):
    """MAPE and std/sqrt(n) from pooled sums; None below two steps."""
    n = accumulator["n"]
    if n < 2:
        return None
    mape = accumulator["sum_ape"] / n
    variance = max(accumulator["sum_sq_ape"] - n * mape * mape, 0.0) / (n - 1)
    return {"mape": mape, "se": math.sqrt(variance / n), "n": n}
