# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the numbers that AISim code cites from the SGLang engine probe against the committed files.

Standard library only. Run it with no arguments:

    python benchmarks/evidence/collector/sglbench-engine-probe-20261113/verify.py

It reads ``results/<run>/results.json``, ``provenance.json`` and ``phase0_spike.py`` next to this file. It prints
one line per group of assertions and a last line with the totals. The exit status is 0 when every assertion
holds and 1 otherwise; failed assertions are printed under their group.

Most groups check what the result files record. Three do not. The group "saved driver" checks constants of
``phase0_spike.py`` that no result file records; README.md ("Limits") says why the saved driver is not shown to
be the one that wrote every file. The groups "files" and "provenance.json" check the hashes, the run records and
the file times that provenance.json holds.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SNAPSHOT = {
    "Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}

# run: (SGLang version, model, page size, TP size, KV pool tokens, checks in the file)
RUNS = {
    "v0522-q06-p1": ("0.5.22", "Qwen3-0.6B", 1, 1, 1165358, "1,2,3,4,5,6"),
    "v0522-q06-p64": ("0.5.22", "Qwen3-0.6B", 64, 1, 1165312, "1,2,3,4,5,6"),
    "v0522-q06-p1-over": ("0.5.22", "Qwen3-0.6B", 1, 1, 1165358, "1,9"),
    "v0522-q06-p64-over": ("0.5.22", "Qwen3-0.6B", 64, 1, 1165312, "1,9"),
    "v0522-q06-p64-over2": ("0.5.22", "Qwen3-0.6B", 64, 1, 1165312, "1,9"),
    "v0522-q8-p1": ("0.5.22", "Qwen3-8B", 1, 1, 803762, "1,2,3,4,5,6,8"),
    "v0522-q8-p64": ("0.5.22", "Qwen3-8B", 64, 1, 803712, "1,2,3,4,5,8"),
    "v0522-q8-tp2": ("0.5.22", "Qwen3-8B", 1, 2, 1704996, "1,2,3,4"),
    "v0523-q06-p1-a": ("0.5.23", "Qwen3-0.6B", 1, 1, 1160904, "1,2,3"),
    "v0523-q06-p1-b": ("0.5.23", "Qwen3-0.6B", 1, 1, 1160904, "1,4,5,6"),
    "v0523-q06-p64": ("0.5.23", "Qwen3-0.6B", 64, 1, 1160896, "1,2,3,4,5,6"),
    "v0523-q06-p1-over": ("0.5.23", "Qwen3-0.6B", 1, 1, 1160904, "1,9"),
    "v0523-q06-p64-over": ("0.5.23", "Qwen3-0.6B", 64, 1, 1160896, "1,9"),
    "v0523-q06-p64-over2": ("0.5.23", "Qwen3-0.6B", 64, 1, 1160896, "1,9"),
    "v0523-q8-p1": ("0.5.23", "Qwen3-8B", 1, 1, 800297, "1,2,3,4,5,6,8"),
    "v0523-q8-p64": ("0.5.23", "Qwen3-8B", 64, 1, 800256, "1,2,3,4,5,8"),
    "v0523-q8-tp2": ("0.5.23", "Qwen3-8B", 1, 2, 1698074, "1,2,3,4"),
}

# The limits every scheduler header reports alike: I, R, M and C in the names of sglbench_grid.Limits.
MAX_REQ_INPUT_LEN = 40954
MAX_RUNNING_REQUESTS = 4096
MAX_PREFILL_TOKENS = 16384
CHUNKED_PREFILL_SIZE = 8192

# The two limits records of the tests (H200_PAGE_1 and H200_PAGE_64 in test_fpm_sglbench_grid.py), in the field
# order of sglbench_grid.Limits: page size, I, KV pool, R, M, C.
LIMITS_PAGE_1 = (1, 40954, 1165358, 4096, 16384, 8192)
LIMITS_PAGE_64 = (64, 40954, 1165312, 4096, 16384, 8192)
LIMIT_FIELDS = (
    "page_size",
    "max_req_input_len",
    "max_total_num_tokens",
    "max_running_requests",
    "max_prefill_tokens",
    "chunked_prefill_size",
)

HEADER_KEYS = {
    "file",
    "type",
    "pid",
    "sglang_version",
    "class",
    "mro",
    "install",
    "ranks",
    "input_blocker",
    "enable_overlap",
    *LIMIT_FIELDS,
}

# Words that would name the GPU, the image, the torch build, the radix-tree core or a graph size. README.md says
# that no result file holds any of them.
NOT_RECORDED = (b"h200", b"gpu", b"cuda", b"nvidia", b"torch", b"docker", b"lmsysorg", b"radix", b"graph")

SEVEN_TOKEN_RUN = "EXTEND DECODE DECODE DECODE DECODE DECODE DECODE"

# check_5 stages these points in this order (batch, context).
REUSE_CONTEXTS = (4096, 2048, 1024, 512, 256, 128, 64, 32)
REUSE_POINTS = [(batch, context) for batch in (4, 2, 1) for context in REUSE_CONTEXTS][:20]
# Runs in which one of the four cold seeds of check_5 found one token cached (4092 computed, not 4093).
ONE_TOKEN_MATCH = ("v0522-q06-p1", "v0522-q8-p1", "v0523-q8-p1")


class Group:
    """One named group of assertions. It records failures and does not raise."""

    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.failures: list[str] = []

    def equal(self, got, want, what: str) -> None:
        self.count += 1
        if got != want:
            self.failures.append(f"{what}: got {got!r}, want {want!r}")

    def true(self, condition, what: str) -> None:
        self.count += 1
        if not condition:
            self.failures.append(what)


def load(run: str) -> dict:
    return json.loads((HERE / "results" / run / "results.json").read_text(encoding="utf-8"))


def runs_with(key: str) -> list[str]:
    return [run for run in RUNS if key[len("check_") :] in RUNS[run][5].split(",")]


def argument(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def pct(value: float, reference: float) -> float:
    return (value - reference) / reference * 100.0


def tenths_down(value: float) -> int:
    return math.floor(value * 10)


def tenths_up(value: float) -> int:
    return math.ceil(value * 10)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conditions(runs: list[str]) -> set[tuple]:
    """The (SGLang version, model, page size, TP size) of each run."""
    return {RUNS[run][:4] for run in runs}


# Both tags, both models and both page sizes at TP 1; and the same with Qwen3-8B at page size 1 and TP 2 added.
TP_1 = {(v, m, p, 1) for v in ("0.5.22", "0.5.23") for m in ("Qwen3-0.6B", "Qwen3-8B") for p in (1, 64)}
TP_1_AND_2 = TP_1 | {(v, "Qwen3-8B", 1, 2) for v in ("0.5.22", "0.5.23")}


# ------------------------------------------------------------------------------------------------- the groups


def group_files(g: Group) -> None:
    """The folder holds the 17 runs and the files provenance.json lists, with the hashes it gives."""
    found = sorted(path.name for path in (HERE / "results").iterdir() if path.is_dir())
    g.equal(found, sorted(RUNS), "run folders under results/")
    g.equal(len(RUNS), 17, "number of runs")
    for run in RUNS:
        names = sorted(path.name for path in (HERE / "results" / run).iterdir())
        g.equal(names, ["results.json"], f"{run}: files in the run folder")
    listed = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))["sha256"]
    present = sorted(
        path.relative_to(HERE).as_posix()
        for path in HERE.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
    )
    g.equal(sorted([*listed, "provenance.json"]), present, "files listed in provenance.json against the folder")
    for name, digest in listed.items():
        g.equal(sha256(HERE / name), digest, f"sha256 of {name}")


def group_conditions(g: Group) -> None:
    """Each run is what its name says: SGLang version, model, page size, TP size, checks; no fault."""
    for run, (version, model, page, tp_size, _pool, checks) in RUNS.items():
        data = load(run)
        present = sorted(key for key in data if key.startswith("check_"))
        g.equal(present, [f"check_{key}" for key in checks.split(",")], f"{run}: checks in the file")
        other = sorted(key for key in data if not key.startswith("check_"))
        g.equal(other, ["argv", "faults", "headers", "recorder_dir", "sglang_argv"], f"{run}: other top-level keys")
        g.equal(data["faults"], [], f"{run}: recorder faults")
        g.equal(argument(data["argv"], "--checks"), checks, f"{run}: --checks argument")
        path = argument(data["sglang_argv"], "--model-path") or ""
        g.true(path.endswith(f"models--Qwen--{model}/snapshots/{SNAPSHOT[model]}"), f"{run}: model path names {model}")
        extra = {1: [], 64: ["--page-size", "64"]}[page] + ([] if tp_size == 1 else ["--tp-size", "2"])
        g.equal(data["sglang_argv"][2:], extra, f"{run}: SGLang arguments after the model path")
        g.equal(len(data["headers"]), tp_size, f"{run}: one header per scheduler process")
        first = data["check_1"]
        g.equal(
            (first["pass"], first["scheduler_processes"], first["expected"], first["faults"]),
            (True, tp_size, tp_size, []),
            f"{run}: check_1",
        )
        for header in data["headers"]:
            g.equal(set(header), HEADER_KEYS, f"{run}: header keys")
            g.equal(header["sglang_version"], version, f"{run}: sglang_version")
            g.equal(header["class"], "RecorderScheduler", f"{run}: scheduler class")
            g.equal(header["mro"][:2], ["RecorderScheduler", "Scheduler"], f"{run}: the recorder subclasses Scheduler")
            g.equal(header["install"], "hook", f"{run}: install")
            g.equal(header["page_size"], page, f"{run}: page_size")
            g.equal(header["ranks"]["tp_size"], tp_size, f"{run}: tp_size")
            g.equal(header["ranks"]["attn_dp_enabled"], False, f"{run}: DP attention off")
            g.equal(header["enable_overlap"], True, f"{run}: overlap scheduling on")
            g.equal(header["input_blocker"], False, f"{run}: no input blocker")
    g.equal(sum(1 for run in RUNS if RUNS[run][0] == "0.5.22"), 8, "runs at 0.5.22")
    g.equal(sum(1 for run in RUNS if RUNS[run][0] == "0.5.23"), 9, "runs at 0.5.23")
    shapes = [(RUNS[run][2], RUNS[run][3]) for run in RUNS]
    g.equal(
        (shapes.count((1, 1)), shapes.count((64, 1)), shapes.count((1, 2))),
        (7, 8, 2),
        "runs with the model path alone, with --page-size 64, with --tp-size 2",
    )


def group_not_recorded(g: Group) -> None:
    """No result file names a GPU, an image, a torch build, a radix-tree core or a graph size."""
    for run in RUNS:
        raw = (HERE / "results" / run / "results.json").read_bytes().lower()
        hits = [word.decode() for word in NOT_RECORDED if word in raw]
        g.equal(hits, [], f"{run}: words that would name hardware, image, torch, tree core or graph sizes")


def group_limits(g: Group) -> None:
    """The limits the scheduler headers report, and the two records the tests use."""
    first = load("v0522-q06-p1")["headers"][0]
    g.equal(tuple(first[name] for name in LIMIT_FIELDS), LIMITS_PAGE_1, "v0522-q06-p1 headers[0]: limits at page 1")
    second = load("v0522-q06-p64")["headers"][0]
    g.equal(tuple(second[name] for name in LIMIT_FIELDS), LIMITS_PAGE_64, "v0522-q06-p64 headers[0]: limits")
    for run, (_version, _model, page, _tp_size, pool, _checks) in RUNS.items():
        for header in load(run)["headers"]:
            g.equal(header["max_req_input_len"], MAX_REQ_INPUT_LEN, f"{run}: max_req_input_len")
            g.equal(header["max_running_requests"], MAX_RUNNING_REQUESTS, f"{run}: max_running_requests")
            g.equal(header["max_prefill_tokens"], MAX_PREFILL_TOKENS, f"{run}: max_prefill_tokens")
            g.equal(header["chunked_prefill_size"], CHUNKED_PREFILL_SIZE, f"{run}: chunked_prefill_size")
            g.equal(header["max_total_num_tokens"], pool, f"{run}: max_total_num_tokens")
            g.equal(header["max_total_num_tokens"] % page, 0, f"{run}: the pool is a whole number of pages")
    # The pool differs by tag, model and TP size; the two records above are v0.5.22 with Qwen3-0.6B at TP 1.
    pools = {(RUNS[run][0], RUNS[run][1], RUNS[run][2], RUNS[run][3]): RUNS[run][4] for run in RUNS}
    g.equal(pools[("0.5.23", "Qwen3-0.6B", 1, 1)], 1160904, "pool at 0.5.23, Qwen3-0.6B, page 1")
    g.equal(pools[("0.5.23", "Qwen3-0.6B", 64, 1)], 1160896, "pool at 0.5.23, Qwen3-0.6B, page 64")
    g.equal(len(set(pools.values())), 10, "distinct pool sizes over the 17 runs")


def group_check_2(g: Group) -> None:
    """check_2: one call of 2 and of 128 requests (32 in one run) is one extend batch of all of them."""
    runs = runs_with("check_2")
    g.equal(len(runs), 10, "runs with check_2")
    with_128 = []
    for run in runs:
        check = load(run)["check_2"]
        g.equal(check["pass"], True, f"{run}: check_2 pass")
        sizes = [case["batch"] for case in check["cases"]]
        g.equal(sizes, [2, 32] if run == "v0523-q06-p1-a" else [2, 128], f"{run}: batch sizes")
        if 128 in sizes:
            with_128.append(run)
        for case in check["cases"]:
            g.equal(case["pass"], True, f"{run} batch {case['batch']}: pass")
            g.equal(case["modes"], ["EXTEND"], f"{run} batch {case['batch']}: modes")
            g.equal(len(case["extend_batches"]), 1, f"{run} batch {case['batch']}: number of extend batches")
            g.equal(case["extend_batches"][0][1], case["batch"], f"{run} batch {case['batch']}: extend batch size")
    g.equal(len(with_128), 9, "runs with the 128-request case")
    g.equal(conditions(runs), TP_1_AND_2, "check_2: both tags, models and page sizes at TP 1, and Qwen3-8B at TP 2")
    # The 128-request case is missing only at 0.5.23 with Qwen3-0.6B at page size 1, where 32 requests ran.
    g.equal(conditions(with_128), TP_1_AND_2 - {("0.5.23", "Qwen3-0.6B", 1, 1)}, "conditions of the 128-request case")
    # 64 new tokens per request is a constant of the saved driver (group "saved driver"); 128 of them are C.
    g.equal(128 * 64, CHUNKED_PREFILL_SIZE, "128 * 64 against chunked_prefill_size")


def group_check_9_prefill(g: Group) -> None:
    """check_9.prefill: 136 requests of 64 new tokens ran as 128 (8192 tokens) and then 8 (512 tokens)."""
    runs = runs_with("check_9")
    g.equal(len(runs), 6, "runs with check_9")
    g.equal(
        conditions(runs),
        {(v, "Qwen3-0.6B", p, 1) for v in ("0.5.22", "0.5.23") for p in (1, 64)},
        "check_9 ran on Qwen3-0.6B at both tags and both page sizes, at TP 1",
    )
    for run in runs:
        check = load(run)["check_9"]
        g.equal(check["pass"], None, f"{run}: check_9 is informational")
        g.equal(check["step_budget"], CHUNKED_PREFILL_SIZE, f"{run}: step_budget")
        g.equal(check["page_size"], RUNS[run][2], f"{run}: page_size")
        prefill = check["prefill"]
        g.equal(
            (prefill["batch"], prefill["largest_batch_that_fits"], prefill["new_tokens_each"]),
            (136, 128, 64),
            f"{run}: prefill batch, largest batch that fits, new tokens each",
        )
        batches = prefill["extend_batches"]
        g.equal([(size, tokens) for _iter, size, tokens in batches], [(128, 8192), (8, 512)], f"{run}: extend batches")
        g.equal(batches[1][0] - batches[0][0], 1, f"{run}: the two extend batches are consecutive forward passes")
    g.equal((128 * 64, 8 * 64), (CHUNKED_PREFILL_SIZE, 512), "128 * 64 is C and 8 * 64 is 512")
    g.true(CHUNKED_PREFILL_SIZE < 129 * 64, "129 requests of 64 tokens are above C")


def group_check_9_decode_136(g: Group) -> None:
    """check_9.decode in the four -over files: 136 requests at page 64; skipped at page 1."""
    for run in ("v0522-q06-p64-over", "v0523-q06-p64-over"):
        decode = load(run)["check_9"]["decode"]
        # The older shape: one run, no "runs" list, no context. "sequence_ok" asks for exactly one extend batch.
        g.equal(sorted(decode), ["batch", "batches", "largest_batch_that_fits", "sequence_ok"], f"{run}: decode keys")
        g.equal((decode["batch"], decode["largest_batch_that_fits"]), (136, 128), f"{run}: decode batch")
        want = [["EXTEND", 128], ["EXTEND", 8]] + [["DECODE", 136]] * 6
        g.equal(decode["batches"], want, f"{run}: two extend batches, then six decode batches of all 136")
        g.equal(decode["sequence_ok"], False, f"{run}: sequence_ok is false because there are two extend batches")
        g.equal(argument(load(run)["argv"], "--over-budget-extra"), None, f"{run}: default extra of 8 requests")
    for run in ("v0522-q06-p1-over", "v0523-q06-p1-over"):
        decode = load(run)["check_9"]["decode"]
        g.equal(decode, {"skipped": "needs 8200 requests, above the running limit of 4096"}, f"{run}: decode skipped")


def group_check_9_decode_384(g: Group) -> None:
    """check_9.decode in the two -over2 files: 384 requests, five runs each, and their decode intervals."""
    last, before = [], []
    for run in ("v0522-q06-p64-over2", "v0523-q06-p64-over2"):
        data = load(run)
        g.equal(argument(data["argv"], "--over-budget-extra"), "256", f"{run}: --over-budget-extra")
        g.equal(argument(data["argv"], "--repeats"), "5", f"{run}: --repeats")
        decode = data["check_9"]["decode"]
        g.equal(sorted(decode), ["batch", "largest_batch_that_fits", "runs"], f"{run}: decode keys")
        g.equal((decode["batch"], decode["largest_batch_that_fits"]), (128 + 256, 128), f"{run}: decode batch")
        g.equal(len(decode["runs"]), 5, f"{run}: number of runs")
        for index, item in enumerate(decode["runs"]):
            want = [["EXTEND", 128]] * 3 + [["DECODE", 384]] * 6
            g.equal(item["batches"], want, f"{run} run {index}: three extend batches, then six decode batches")
            g.equal(item["decode_batches_hold_all"], True, f"{run} run {index}: decode batches hold all 384")
            g.equal(len(item["decode_inter_output_ms"]), 5, f"{run} run {index}: five decode intervals")
            last.append(item["decode_inter_output_ms"][-1])
            before.extend(item["decode_inter_output_ms"][:-1])
    g.equal((len(last), len(before)), (10, 40), "ten last intervals and forty before them")
    g.equal((min(last), max(last)), (1.895, 2.503), "last interval, smallest and largest, ms")
    g.equal((tenths_down(min(last)), tenths_up(max(last))), (18, 26), "last interval is 1.8 to 2.6 ms")
    g.equal((min(before), max(before)), (17.138, 20.11), "the four intervals before it, smallest and largest, ms")
    g.equal((tenths_down(min(before)), tenths_up(max(before))), (171, 202), "the four before it are 17.1 to 20.2 ms")
    g.equal((384 * 64, 3 * CHUNKED_PREFILL_SIZE), (24576, 24576), "384 * 64 is three times C")


def group_check_3(g: Group) -> None:
    """check_3: cached-token counts of prefill prefixes and of a seeded decode run of two requests."""
    runs = runs_with("check_3")
    g.equal(len(runs), 10, "runs with check_3")
    g.equal(sorted(RUNS[run][2] for run in runs), [1] * 6 + [64] * 4, "six runs at page size 1, four at 64")
    g.equal(conditions(runs), TP_1_AND_2, "check_3: both tags, models and page sizes at TP 1, and Qwen3-8B at TP 2")
    for run in runs:
        page = RUNS[run][2]
        check = load(run)["check_3"]
        aligned, unaligned, cached, extend = {1: (128, 135, 1020, 2), 64: (512, 512, 960, 122)}[page]
        want = {
            "prefill aligned prefix, warm shot": aligned,
            "prefill aligned prefix, measured shot": aligned,
            "prefill unaligned prefix": unaligned,
            "decode seed (cold)": 0,
            "decode measured run": cached,
        }
        g.equal(check["pass"], True, f"{run}: check_3 pass")
        g.equal(check["page_size"], page, f"{run}: page_size")
        g.equal([case["case"] for case in check["cases"]], list(want), f"{run}: cases")
        for case in check["cases"]:
            g.equal(case["cached_tokens"], [want[case["case"]]] * 2, f"{run}: cached tokens, {case['case']}")
            g.equal((case["expected"], case["pass"]), (want[case["case"]], True), f"{run}: expected, {case['case']}")
        g.equal(check["decode_prompt_tokens"], [1021, 1021], f"{run}: decode prompt tokens")
        g.equal(check["decode_extend_tokens"], [extend], f"{run}: extend tokens of the batch of two requests")
        # Per request: 1021 - 1020 = 1 new token at page size 1, 1021 - 960 = 61 at page size 64.
        g.equal(
            (1021 - cached, extend // 2, extend % 2), ({1: 1, 64: 61}[page], 1021 - cached, 0), f"{run}: per request"
        )
        g.equal(cached, (1021 - 1) // page * page, f"{run}: cached is the prompt less one token, floored to pages")


def group_check_5(g: Group) -> None:
    """check_5: what a seed request computes when its prompt, or a longer one, was sent before."""
    runs = runs_with("check_5")
    g.equal(len(runs), 8, "runs with check_5")
    g.equal(sorted(RUNS[run][2] for run in runs), [1] * 4 + [64] * 4, "four runs at page size 1, four at 64")
    g.equal(conditions(runs), TP_1, "check_5: both tags, both models and both page sizes, all at TP 1")
    for run in runs:
        page = RUNS[run][2]
        check = load(run)["check_5"]
        rows = check["rows"]
        g.equal((check["pass"], check["points"], len(rows)), (True, 20, 20), f"{run}: pass and number of points")
        g.equal([(row["batch"], row["context"]) for row in rows], REUSE_POINTS, f"{run}: points in order")
        g.equal(all(row["run_sequence_ok"] for row in rows), True, f"{run}: every seven-token run is 1 + 6 batches")
        cold = [4093, 4093, 4092, 4093] if run in ONE_TOKEN_MATCH else [4093] * 4
        g.equal(rows[0]["seed_computed_tokens"], cold, f"{run}: tokens computed by the four cold seeds")
        g.equal(rows[0]["seed_was_cached"], [False] * 4, f"{run}: the first point is cold")
        for row in rows[1:]:
            prompt = row["context"] - 3
            want = prompt - (prompt - 1) // page * page
            where = f"{run} batch {row['batch']} context {row['context']}"
            g.equal(row["seed_was_cached"], [True] * row["batch"], f"{where}: the prompt was sent before")
            g.equal(row["seed_computed_tokens"], [want] * row["batch"], f"{where}: tokens computed")
        # A prompt sent again after a seven-token run of the same prompt: batch 2 and batch 1, here at contexts
        # 4096 (a prompt of 4093 tokens) and 1024 (1021 tokens). 1 token is computed at page size 1, 61 at 64.
        new = {1: 1, 64: 61}[page]
        for context, prompt, cached in (
            (4096, 4093, {1: 4092, 64: 4032}[page]),
            (1024, 1021, {1: 1020, 64: 960}[page]),
        ):
            again = [row for row in rows if row["context"] == context and row["batch"] < 4]
            g.equal([row["batch"] for row in again], [2, 1], f"{run}: rows that send the {prompt}-token prompt again")
            for row in again:
                g.equal(row["seed_computed_tokens"], [new] * row["batch"], f"{run}: {prompt}-token prompt sent again")
            g.equal(prompt - new, cached, f"{run}: cached tokens of the {prompt}-token prompt sent again")
        total = {1: 16420, 64: 19108}[page] - (1 if run in ONE_TOKEN_MATCH else 0)
        g.equal(check["seed_tokens_with_reuse"], total, f"{run}: seed tokens computed over the 20 points")
        g.equal(check["seed_tokens_without_reuse"], 56484, f"{run}: seed tokens if nothing were cached")
    g.equal(
        sorted(run for run in runs if run in ONE_TOKEN_MATCH), sorted(ONE_TOKEN_MATCH), "runs with a one-token match"
    )


def decode_cases() -> list[tuple[str, dict]]:
    return [(run, case) for run in runs_with("check_4") for case in load(run)["check_4"]["cases"]]


def group_check_4_runs(g: Group) -> None:
    """check_4: a seven-token run is one extend batch and six decode batches of all requests."""
    runs = runs_with("check_4")
    g.equal(len(runs), 10, "runs with check_4")
    g.equal(conditions(runs), TP_1_AND_2, "check_4: both tags, models and page sizes at TP 1, and Qwen3-8B at TP 2")
    cases = decode_cases()
    big = [run for run, _case in cases if RUNS[run][1] == "Qwen3-8B"]
    g.equal((len(cases), len(big), len(cases) - len(big)), (20, 12, 8), "cases: all, Qwen3-8B, Qwen3-0.6B")
    g.equal(sum(1 for run in big if RUNS[run][3] == 2), 4, "Qwen3-8B cases with --tp-size 2")
    g.equal({RUNS[run][3] for run, _case in cases if RUNS[run][1] == "Qwen3-0.6B"}, {1}, "Qwen3-0.6B cases are TP 1")
    for run in runs:
        data = load(run)
        check = data["check_4"]
        g.equal(check["pass"], True, f"{run}: check_4 pass")
        g.equal([case["batch"] for case in check["cases"]], [2, 8], f"{run}: batch sizes")
        repeats = {"Qwen3-0.6B": None, "Qwen3-8B": "7" if RUNS[run][3] == 1 else "5"}[RUNS[run][1]]
        g.equal(argument(data["argv"], "--repeats"), repeats, f"{run}: --repeats (absent means the default of 5)")
        if "check_6" in data:
            g.equal(len(data["check_6"]["kept_ms_stream_off"]), int(repeats or 5), f"{run}: repeats seen in check_6")
        for case in check["cases"]:
            where = f"{run} batch {case['batch']}"
            g.equal((case["context"], case["pass"]), (1024, True), f"{where}: context and pass")
            g.equal(case["modes_seen"], [SEVEN_TOKEN_RUN], f"{where}: batch modes of every repeat")
            g.equal(len(case["median_inter_output_ms_by_step"]), 6, f"{where}: six decode step times")
            g.equal(len(case["median_launch_to_output_ms_by_step"]), 6, f"{where}: six launch-to-output times")


def group_decode_note(g: Group) -> None:
    """check_4: the step figures of the decode deviation note in sglbench_artifact.py, against step 5."""

    def rel(case: dict, step: int) -> float:
        steps = case["median_inter_output_ms_by_step"]
        return pct(steps[step - 1], steps[4])

    def sample(case: dict) -> float:
        return pct(case["median_kept_ms"], case["median_inter_output_ms_by_step"][4])

    cases = decode_cases()
    big = [case for run, case in cases if RUNS[run][1] == "Qwen3-8B"]
    small = [(run, case) for run, case in cases if RUNS[run][1] == "Qwen3-0.6B"]

    # Qwen3-8B. "Short" means more than 1 percent below step 5.
    first = [rel(case, 1) for case in big]
    short = [-value for value in first if value < -1]
    rest = [abs(value) for value in first if value >= -1]
    g.equal(len(short), 10, "8B: cases in which step 1 is short")
    g.equal((tenths_down(min(short)), tenths_up(max(short))), (61, 106), "8B: step 1 is 6.1 to 10.6 percent shorter")
    g.equal((len(rest), tenths_up(max(rest))), (2, 2), "8B: step 1 is within 0.2 percent in the other two")
    g.equal(tenths_up(max(abs(rel(case, 2)) for case in big)), 5, "8B: step 2 is within 0.5 percent")
    g.equal(tenths_up(max(abs(rel(case, 6)) for case in big)), 3, "8B: step 6 is within 0.3 percent")
    kept = [abs(rel(case, step)) for case in big for step in (3, 4)]
    g.equal(tenths_up(max(kept)), 5, "8B: steps 3 and 4 are within 0.5 percent")
    g.equal(tenths_up(max(abs(sample(case)) for case in big)), 3, "8B: the sample is within 0.3 percent")

    # Qwen3-0.6B. Steps 2 and 3 count as short when more than 5 percent below step 5, step 6 when more than 1.
    first = [rel(case, 1) for _run, case in small]
    g.equal(
        (len(first), math.floor(min(first)), math.ceil(max(first))),
        (8, 30, 71),
        "0.6B: step 1 is 30 to 71 percent longer",
    )
    second = [-rel(case, 2) for _run, case in small if rel(case, 2) < -5]
    g.equal(
        (len(second), math.floor(min(second)), math.ceil(max(second))),
        (6, 9, 19),
        "0.6B: step 2 is 9 to 19 percent shorter in 6 cases",
    )
    third = [-rel(case, 3) for _run, case in small if rel(case, 3) < -5]
    g.equal(
        (len(third), math.floor(min(third)), math.ceil(max(third))),
        (6, 6, 15),
        "0.6B: step 3 is 6 to 15 percent shorter in 6 cases",
    )
    sixth = sorted(round(-rel(case, 6), 1) for _run, case in small if rel(case, 6) < -1)
    g.equal(sixth, [3.0, 24.7], "0.6B: step 6 is 3.0 and 24.7 percent shorter in 2 cases")
    steady = [abs(sample(case)) for _run, case in small if abs(sample(case)) <= 0.5]
    g.equal((len(steady), tenths_up(max(steady))), (6, 5), "0.6B: the sample is within 0.5 percent in 6 cases")
    low = sorted((round(-sample(case), 1), run, case["batch"]) for run, case in small if sample(case) < -0.5)
    g.equal(low, [(2.1, "v0522-q06-p1", 2), (6.9, "v0522-q06-p64", 2)], "0.6B: the sample is 2.1 and 6.9 percent low")
    fourth = sorted((run, case["batch"]) for run, case in small if rel(case, 4) < -1)
    g.equal(fourth, [(run, batch) for _low, run, batch in low], "0.6B: step 4 is short in the same two cases")
    quoted = next(case for run, case in small if run == "v0522-q06-p64" and case["batch"] == 2)
    g.equal(quoted["median_inter_output_ms_by_step"], [2.015, 1.401, 1.419, 1.442, 1.549, 1.551], "the quoted steps")
    g.equal(quoted["median_kept_ms"], 1.442, "the quoted sample")

    highest = max(sample(case) for _run, case in cases)
    g.equal((highest > 0, tenths_up(highest)), (True, 3), "all 20 cases: the sample is at most 0.3 percent high")


def group_launch_to_output(g: Group) -> None:
    """check_4: how long before its output a decode batch was launched, in units of the step-5 interval."""
    big, small = [], []
    for run, case in decode_cases():
        step = case["median_inter_output_ms_by_step"][4]
        ratios = [value / step for value in case["median_launch_to_output_ms_by_step"][2:]]
        (big if RUNS[run][1] == "Qwen3-8B" else small).extend(ratios)
    g.equal((len(big), len(small)), (48, 32), "ratios for decode steps 3 to 6")
    g.equal((tenths_down(min(big)), tenths_up(max(big))), (18, 20), "8B: launched 1.8 to 2.0 step intervals before")
    g.equal((tenths_down(min(small)), tenths_up(max(small))), (13, 18), "0.6B: 1.3 to 1.8 step intervals before")


def group_informational(g: Group) -> None:
    """check_6 and check_8: the kept decode time with streaming on, and with a cache filled past the pool."""
    runs = runs_with("check_6")
    g.equal(len(runs), 6, "runs with check_6")
    ratios = []
    for run in runs:
        check = load(run)["check_6"]
        g.equal((check["pass"], check["batch"], check["context"]), (None, 8, 1024), f"{run}: check_6 conditions")
        g.equal(check["on_over_off"], round(check["median_on_ms"] / check["median_off_ms"], 4), f"{run}: on_over_off")
        ratios.append(check["on_over_off"])
    g.equal((min(ratios), max(ratios)), (0.9626, 1.0006), "streaming on over off, smallest and largest")
    runs = runs_with("check_8")
    g.equal(len(runs), 4, "runs with check_8")
    ratios = []
    for run in runs:
        check = load(run)["check_8"]
        g.equal((check["pass"], check["batch"], check["context"]), (None, 8, 1024), f"{run}: check_8 conditions")
        g.equal(check["pool_tokens"], RUNS[run][4], f"{run}: pool_tokens")
        g.equal(check["filled_tokens"], 966656, f"{run}: tokens sent to fill the cache")
        g.true(check["filled_tokens"] >= 1.2 * check["pool_tokens"], f"{run}: filled at least 1.2 times the pool")
        ratios.append(check["full_over_empty"])
    g.equal((min(ratios), max(ratios)), (0.9996, 1.0005), "full over empty, smallest and largest")
    pools = [RUNS[run][4] for run in runs]
    g.equal((min(pools), max(pools)), (800256, 803762), "pool sizes of the four runs, smallest and largest")
    g.equal(conditions(runs), {key for key in TP_1 if key[1] == "Qwen3-8B"}, "check_8 ran on Qwen3-8B at TP 1")


def group_saved_driver(g: Group) -> None:
    """Constants of phase0_spike.py that the statements above rest on and no result file records."""
    tree = ast.parse((HERE / "phase0_spike.py").read_text(encoding="utf-8"))
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def has(scope: str | None, statement: str) -> bool:
        want = ast.dump(ast.parse(statement).body[0])
        nodes = tree.body if scope is None else ast.walk(functions[scope])
        return any(ast.dump(node) == want for node in nodes if isinstance(node, ast.stmt))

    expected = (
        (None, "DECODE_TOKENS = 7", "a decode run asks for seven output tokens"),
        (None, "KEPT = (2, 5)", "the kept steps are 3 to 5"),
        ("decode_run", "length = context - 3", "a decode prompt is the context less three tokens"),
        ("decode_run", "seed = stage(ctx, prompts, 1, f'{tag}-seed', need=1)", "the seed asks for one token"),
        ("analyse_decode", "kept = sorted(inter[KEPT[0] : KEPT[1]])", "the kept step times are sorted"),
        (
            "analyse_decode",
            "result.update(inter_output_ms=inter, launch_to_output_ms=launch, kept_ms=kept[len(kept) // 2])",
            "the sample of one run is the upper median of the kept steps",
        ),
        ("check_batch_call", "new = max(64, ctx.page)", "check_2 sends 64 new tokens per request"),
        ("check_page_formulas", "page, unit = ctx.page, max(ctx.page, 16)", "check_3: the prefix unit"),
        ("check_page_formulas", "cached, new = 8 * unit, max(64, ctx.page)", "check_3: the aligned prefix"),
        ("check_page_formulas", "odd = cached + 7", "check_3: the unaligned prefix is seven tokens longer"),
        ("check_over_budget", "new = max(64, page)", "check_9 sends 64 new tokens per request"),
        ("check_over_budget", "batch = fits + ctx.args.over_budget_extra", "check_9: the decode batch"),
        (
            "check_over_budget",
            "_, run = decode_run(ctx, slots, 256, f'c9dr{repeat}')",
            "check_9 decodes at context 256",
        ),
        ("parse", "parser.add_argument('--repeats', type=int, default=5)", "the default number of repeats is 5"),
        ("parse", "parser.add_argument('--decode-context', type=int, default=1024)", "the default decode context"),
        ("parse", "parser.add_argument('--max-batch', type=int, default=32)", "the default largest batch is 32"),
    )
    for scope, statement, what in expected:
        g.true(has(scope, statement), f"phase0_spike.py, {scope or 'module'}: {statement} ({what})")
    # What follows from those constants at page sizes 1 and 64.
    for page, aligned, unaligned, floored in ((1, 128, 135, 135), (64, 512, 519, 512)):
        unit = max(page, 16)
        g.equal((8 * unit, 8 * unit + 7), (aligned, unaligned), f"page {page}: prefix lengths of check_3")
        g.equal((8 * unit + 7) // page * page, floored, f"page {page}: the unaligned prefix floored to pages")
    g.equal((1024 - 3, 256 - 3, 4096 - 3), (1021, 253, 4093), "prompt tokens at contexts 1024, 256 and 4096")
    g.equal(
        (253 - (253 - 1) // 64 * 64, -(-61 // 64) * 64), (61, 64), "context 256 at page 64: 61 new tokens, one page"
    )


def group_provenance(g: Group) -> None:
    """provenance.json: its run records agree with the files, and 15 of 17 files are older than the saved driver."""
    provenance = json.loads((HERE / "provenance.json").read_text(encoding="utf-8"))
    records = provenance["result_files"]
    g.equal(sorted(records), sorted(RUNS), "runs with a record")
    for run, (version, model, page, tp_size, _pool, checks) in RUNS.items():
        record = records[run]
        got = (record["sglang_version"], record["model"], record["page_size"], record["tp_size"], record["checks"])
        g.equal(got, (version, f"Qwen/{model}", page, tp_size, checks), f"{run}: run record")
        g.equal(record["recorder_stamp"], load(run)["recorder_dir"].rsplit("/", 1)[1], f"{run}: recorder stamp")
        g.equal(record["bytes"], (HERE / "results" / run / "results.json").stat().st_size, f"{run}: size in bytes")
    # File times of the saved copies; they are metadata that provenance.json holds and git does not keep.
    driver = provenance["scripts"]["phase0_spike.py"]["saved_file_time_utc"]
    later = sorted(run for run in RUNS if records[run]["saved_file_time_utc"] > driver)
    g.equal(later, ["v0522-q06-p64-over2", "v0522-q8-tp2"], "result files newer than the saved driver")
    g.equal(len(RUNS) - len(later), 15, "result files older than the saved driver")
    # The same split by the stamp each run wrote into recorder_dir when it started.
    stamp = driver.replace("-", "").replace(":", "").replace("T", "-").rstrip("Z")
    started_later = sorted(run for run in RUNS if records[run]["recorder_stamp"] > stamp)
    g.equal(started_later, later, "runs whose recorder_dir stamp is later than the saved driver")
    # The times README.md gives. Read as UTC, each stamp is 32 to 87 seconds older than the saved file.
    stamps = sorted(records[run]["recorder_stamp"] for run in RUNS)
    g.equal((stamps[0], stamps[-1]), ("20261114-035605", "20261114-041305"), "first and last recorder stamp")
    saved = sorted(records[run]["saved_file_time_utc"] for run in RUNS)
    g.equal((saved[0], saved[-1]), ("2026-11-14T03:57:15Z", "2026-11-14T04:14:16Z"), "first and last file time")
    delays = []
    for run in RUNS:
        started = datetime.datetime.strptime(records[run]["recorder_stamp"], "%Y%m%d-%H%M%S")
        written = datetime.datetime.strptime(records[run]["saved_file_time_utc"], "%Y-%m-%dT%H:%M:%SZ")
        delays.append(int((written - started).total_seconds()))
    g.equal((min(delays), max(delays)), (32, 87), "seconds from the stamp of a run to its saved file")
    g.equal(driver, "2026-11-14T04:12:19Z", "file time of the saved driver")
    recorder = provenance["scripts"]["phase0_recorder.py"]["saved_file_time_utc"]
    g.equal(recorder, "2026-11-14T01:14:28Z", "file time of the saved recorder")
    for name, marks in (("phase0_spike.py", 6), ("phase0_recorder.py", 1)):
        script = provenance["scripts"][name]
        g.equal(script["committed_sha256"], sha256(HERE / name), f"{name}: sha256 of the committed script")
        g.equal(
            script["syntax_tree_equal_to_saved"], True, f"{name}: recorded as equal in syntax tree to the saved file"
        )
        text = (HERE / name).read_text(encoding="utf-8")
        g.equal(sum(1 for line in text.splitlines() if "  # noqa: " in line), marks, f"{name}: lines with a lint mark")


GROUPS = (
    ("files", group_files),
    ("run conditions", group_conditions),
    ("not recorded", group_not_recorded),
    ("limits", group_limits),
    ("check_2", group_check_2),
    ("check_3", group_check_3),
    ("check_4 runs", group_check_4_runs),
    ("check_4 decode note", group_decode_note),
    ("check_4 launch to output", group_launch_to_output),
    ("check_5", group_check_5),
    ("check_6 and check_8", group_informational),
    ("check_9 prefill", group_check_9_prefill),
    ("check_9 decode, 136 requests", group_check_9_decode_136),
    ("check_9 decode, 384 requests", group_check_9_decode_384),
    ("saved driver", group_saved_driver),
    ("provenance.json", group_provenance),
)


def main() -> int:
    total = failed = 0
    for name, function in GROUPS:
        group = Group(name)
        try:
            function(group)
        except Exception as error:
            group.count += 1
            group.failures.append(f"the group stopped early: {error!r}")
        total += group.count
        failed += len(group.failures)
        if group.failures:
            print(f"FAIL  {name}: {len(group.failures)} of {group.count} assertions failed")
            for failure in group.failures:
                print(f"        {failure}")
        else:
            print(f"ok    {name}: {group.count} assertions")
    print(f"{total} assertions in {len(GROUPS)} groups, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
