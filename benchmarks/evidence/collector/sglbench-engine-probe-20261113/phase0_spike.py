#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# The driver of the engine probe, as saved; it is the last revision (README.md in this folder, "Limits"). Added
# for the repository: these comment lines, the layout that "ruff format" gives, and lint marks on six lines,
# each with its reason. No statement was changed: the syntax tree equals that of the saved file, whose sha256
# is in provenance.json.
"""Phase 0 spike for the AISim SGLang self-benchmark.

Throwaway script: it changes no repo. It builds ``sgl.Engine`` with a recorder
in every scheduler process (see phase0_recorder.py) and runs the checks that
only a real engine can settle.

Usage:
    python phase0_spike.py --out DIR [spike options] -- [SGLang server args]
Example:
    python phase0_spike.py --out /tmp/p0 -- --model-path Qwen/Qwen3-0.6B --page-size 64

Checks, selected with --checks (default 1,2,3,4,5,6,8):
    1  a recorder header from every scheduler process
    2  one batch call gives one extend batch
    3  cached-token counts match the page-size formulas
    4  decode run: one extend batch, then six decode batches; per-step times
    5  seed reuse across decode points without flushing
    6  kept decode time with streaming off and on      (informational)
    8  kept decode time with an empty and a full cache (informational)
    9  what happens to a batch above the step budget   (informational, not in the default)

DP attention, run separately with at least two GPUs and ``--attn-dp-size 2``:
    --dp-check        stage a point with block and unblock; ranks launch together
    --dp-decode-cost  steady decode cadence; run once with --blocker-env and
                      once without, then compare the two medians
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECODE_TOKENS = 7  # one extend batch, then six decode batches
KEPT = (2, 5)  # decode steps 3 to 5, as a slice over zero-based step indexes


def log(message: str) -> None:
    print(f"[phase0] {message}", flush=True)


# --------------------------------------------------------------------------- logs


class Logs:
    """Incremental reader for the recorder's per-process JSON-lines files."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._pos: dict[Path, int] = {}
        self._partial: dict[Path, str] = {}
        self.headers: dict[Path, dict] = {}
        self.faults: list[dict] = []
        self._by_rid: dict[tuple[Path, str], list[dict]] = {}
        self.batch_count: dict[Path, int] = {}

    def poll(self) -> None:
        for path in sorted(self.directory.glob("sched-*.jsonl")):
            # Lint mark UP015, kept as saved: dropping the mode argument would change the statement.
            with open(path, "r", encoding="utf-8") as handle:  # noqa: UP015
                handle.seek(self._pos.get(path, 0))
                data = handle.read()
                self._pos[path] = handle.tell()
            data = self._partial.pop(path, "") + data
            lines = data.split("\n")
            if lines[-1]:
                self._partial[path] = lines[-1]
            for line in lines[:-1]:
                if not line:
                    continue
                record = json.loads(line)
                kind = record.get("type")
                if kind == "header":
                    self.headers[path] = record
                elif kind == "fault":
                    self.faults.append({"file": path.name, **record})
                elif kind == "batch":
                    self.batch_count[path] = self.batch_count.get(path, 0) + 1
                    for rid in record.get("rids") or ():
                        self._by_rid.setdefault((path, rid), []).append(record)

    def leaders(self) -> dict[int, Path]:
        """One log per DP rank: the scheduler process that receives requests."""
        found: dict[int, Path] = {}
        for path, header in self.headers.items():
            ranks = header.get("ranks") or {}
            if "error" in ranks:
                found.setdefault(0, path)
                continue
            if (ranks.get("attn_tp_rank") or 0) or (ranks.get("pp_rank") or 0):
                continue
            if ranks.get("attn_cp_rank") or 0:
                continue
            dp_rank = (ranks.get("attn_dp_rank") or 0) if ranks.get("attn_dp_enabled") else 0
            found.setdefault(int(dp_rank), path)
        return found

    def batches_for(self, path: Path, rids: list[str]) -> list[dict]:
        seen: dict[int, dict] = {}
        for rid in rids:
            for record in self._by_rid.get((path, rid), ()):
                seen[id(record)] = record
        return sorted(seen.values(), key=lambda item: (item.get("forward_iter") or 0))


# ------------------------------------------------------------------------ content


class Content:
    """Per-slot token sequences. A shorter prompt is a prefix of a longer one."""

    def __init__(self, seed: int, low: int, high: int):
        self.seed, self.low, self.high = seed, low, high
        self._cache: dict[str, list[int]] = {}

    def prefix(self, slot: str, length: int) -> list[int]:
        sequence = self._cache.get(slot, [])
        if len(sequence) < length:
            rng = random.Random(f"sglbench:{self.seed}:{slot}")
            sequence = [rng.randrange(self.low, self.high) for _ in range(length)]
            self._cache[slot] = sequence
        return sequence[:length]

    def suffix(self, slot: str, tag: str, length: int, avoid: int | None) -> list[int]:
        rng = random.Random(f"sglbench-suffix:{self.seed}:{slot}:{tag}")
        tokens = [rng.randrange(self.low, self.high) for _ in range(length)]
        if avoid is not None and tokens[0] == avoid:
            tokens[0] = self.low + (tokens[0] - self.low + 1) % (self.high - self.low)
        return tokens


# -------------------------------------------------------------------------- stage


class Context:
    def __init__(self, engine, logs: Logs, args, content: Content):
        self.engine, self.logs, self.args, self.content = engine, logs, args, content
        self.run = uuid.uuid4().hex[:8]
        self.leader: Path | None = None
        self.header: dict = {}

    @property
    def page(self) -> int:
        return int(self.header.get("page_size") or 1)

    def floor_page(self, tokens: int) -> int:
        return tokens // self.page * self.page

    def ceil_page(self, tokens: int) -> int:
        return -(-tokens // self.page) * self.page

    def step_budget(self) -> int:
        limits = [self.header.get("chunked_prefill_size"), self.header.get("max_prefill_tokens")]
        limits = [int(value) for value in limits if isinstance(value, (int, float)) and value > 0]
        return min(limits) if limits else 1 << 30

    def max_batch(self, extend_per_request: int) -> int:
        running = self.header.get("max_running_requests")
        running = int(running) if isinstance(running, (int, float)) and running > 0 else 32
        by_budget = max(1, self.step_budget() // max(1, self.ceil_page(extend_per_request)))
        return max(1, min(running, self.args.max_batch, by_budget))


class Stage:
    def __init__(self, rids, outputs, batches, wall):
        self.rids, self.outputs, self.batches, self.wall = rids, outputs, batches, wall

    def meta(self, key: str) -> list:
        return [(output.get("meta_info") or {}).get(key) for output in self.outputs]


def stage(ctx: Context, prompts, max_new: int, tag: str, stream=False, need=None) -> Stage:
    """Send one batch call and collect the batches that held its requests."""
    rids = [f"{ctx.run}-{tag}-{index}" for index in range(len(prompts))]
    params = {"max_new_tokens": max_new, "temperature": 0.0, "ignore_eos": True}
    started = time.monotonic()
    if stream:
        last: dict[int, dict] = {}
        for chunk in ctx.engine.generate(input_ids=prompts, sampling_params=params, rid=rids, stream=True):
            last[int(chunk.get("index", 0))] = chunk
        outputs = [last[index] for index in range(len(prompts))]
    else:
        outputs = ctx.engine.generate(input_ids=prompts, sampling_params=params, rid=rids)
        if isinstance(outputs, dict):
            outputs = [outputs]
    wall = time.monotonic() - started

    need = max_new if need is None else need
    deadline = time.monotonic() + ctx.args.stage_timeout
    while True:
        ctx.logs.poll()
        batches = ctx.logs.batches_for(ctx.leader, rids)
        # Lint mark C420, kept as saved: dict.fromkeys would change the statement.
        counts = {rid: 0 for rid in rids}  # noqa: C420
        for record in batches:
            for rid in record.get("rids") or ():
                if rid in counts:
                    counts[rid] += 1
        if all(count >= need for count in counts.values()) or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    time.sleep(0.05)  # let a trailing batch of the overlap loop reach the log
    ctx.logs.poll()
    return Stage(rids, outputs, ctx.logs.batches_for(ctx.leader, rids), wall)


def decode_run(ctx: Context, slots, context: int, tag: str, stream=False):
    """Seed each slot's first (context - 3) tokens, then run the measured request."""
    length = context - 3
    prompts = [ctx.content.prefix(slot, length) for slot in slots]
    seed = stage(ctx, prompts, 1, f"{tag}-seed", need=1)
    run = stage(ctx, prompts, DECODE_TOKENS, f"{tag}-run", stream=stream)
    return seed, run


def analyse_decode(run: Stage, batch: int) -> dict:
    modes = [record["mode"] for record in run.batches]
    whole = all(record["bs"] == batch and set(record["rids"]) == set(run.rids) for record in run.batches)
    expected = ["EXTEND"] + ["DECODE"] * (DECODE_TOKENS - 1)
    result = {"modes": modes, "all_requests_in_every_batch": whole}
    result["sequence_ok"] = modes == expected and whole
    if modes[: len(expected)] == expected:
        records = run.batches[: len(expected)]
        inter = [(records[step]["out_ts"] - records[step - 1]["out_ts"]) * 1e3 for step in range(1, len(records))]
        launch = [(records[step]["out_ts"] - records[step]["launch_ts"]) * 1e3 for step in range(1, len(records))]
        kept = sorted(inter[KEPT[0] : KEPT[1]])
        result.update(inter_output_ms=inter, launch_to_output_ms=launch, kept_ms=kept[len(kept) // 2])
    return result


# ------------------------------------------------------------------------- checks


def check_headers(ctx: Context) -> dict:
    ctx.logs.poll()
    headers = list(ctx.logs.headers.values())
    classes = sorted({header.get("class") for header in headers})
    expected = ctx.args.expected_schedulers
    ok = len(headers) == expected and classes == ["RecorderScheduler"]
    return {
        "pass": ok,
        "scheduler_processes": len(headers),
        "expected": expected,
        "classes": classes,
        "install": sorted({header.get("install") for header in headers}),
        "sglang_version": sorted({header.get("sglang_version") for header in headers}),
        "page_size": ctx.page,
        "enable_overlap": ctx.header.get("enable_overlap"),
        "faults": ctx.logs.faults,
    }


def check_batch_call(ctx: Context) -> dict:
    new = max(64, ctx.page)
    sizes = sorted({2, ctx.max_batch(new)})
    cases = []
    for batch in sizes:
        slots = [f"c2-{batch}-{index}" for index in range(batch)]
        prompts = [ctx.content.prefix(slot, new) for slot in slots]
        staged = stage(ctx, prompts, 1, f"c2b{batch}")
        extend = [record for record in staged.batches if record["mode"] == "EXTEND"]
        ok = len(extend) == 1 and set(extend[0]["rids"]) == set(staged.rids) and extend[0]["bs"] == batch
        cases.append(
            {
                "batch": batch,
                "pass": ok,
                "extend_batches": [(record["forward_iter"], record["bs"]) for record in extend],
                "modes": [record["mode"] for record in staged.batches],
            }
        )
    return {"pass": all(case["pass"] for case in cases), "cases": cases}


def check_page_formulas(ctx: Context) -> dict:
    page, unit = ctx.page, max(ctx.page, 16)
    cached, new = 8 * unit, max(64, ctx.page)
    out: dict = {"page_size": page, "cases": []}

    def case(name, got, want):
        out["cases"].append(
            {"case": name, "cached_tokens": got, "expected": want, "pass": all(value == want for value in got)}
        )

    # Prefill with a page-aligned prefix: both shots must match exactly the prefix.
    slots = ["c3a-0", "c3a-1"]
    prefixes = [ctx.content.prefix(slot, cached) for slot in slots]
    stage(ctx, prefixes, 1, "c3a-seed", need=1)
    for shot in ("warm", "measured"):
        prompts = [
            prefixes[i] + ctx.content.suffix(slot, shot, new, ctx.content.prefix(slot, cached + 1)[-1])
            for i, slot in enumerate(slots)
        ]
        staged = stage(ctx, prompts, 1, f"c3a-{shot}")
        case(f"prefill aligned prefix, {shot} shot", staged.meta("cached_tokens"), cached)

    # Prefill with an unaligned prefix: the match rounds down to whole pages.
    slots = ["c3u-0", "c3u-1"]
    odd = cached + 7
    prefixes = [ctx.content.prefix(slot, odd) for slot in slots]
    stage(ctx, prefixes, 1, "c3u-seed", need=1)
    prompts = [
        prefixes[i] + ctx.content.suffix(slot, "odd", new, ctx.content.prefix(slot, odd + 1)[-1])
        for i, slot in enumerate(slots)
    ]
    staged = stage(ctx, prompts, 1, "c3u-shot")
    case("prefill unaligned prefix", staged.meta("cached_tokens"), ctx.floor_page(odd))

    # Decode: prompt of (context - 3) tokens, seeded first.
    context = ctx.args.decode_context
    slots = ["c3d-0", "c3d-1"]
    seed, run = decode_run(ctx, slots, context, "c3d")
    case("decode seed (cold)", seed.meta("cached_tokens"), 0)
    case("decode measured run", run.meta("cached_tokens"), ctx.floor_page(context - 4))
    out["decode_prompt_tokens"] = run.meta("prompt_tokens")
    out["decode_extend_tokens"] = [record["extend_num_tokens"] for record in run.batches if record["mode"] == "EXTEND"]
    out["pass"] = all(item["pass"] for item in out["cases"])
    return out


def check_decode_steps(ctx: Context) -> dict:
    context = ctx.args.decode_context
    sizes = sorted({2, min(8, ctx.max_batch(ctx.page))})
    cases = []
    for batch in sizes:
        slots = [f"c4-{batch}-{index}" for index in range(batch)]
        runs = []
        for repeat in range(ctx.args.repeats):
            _, run = decode_run(ctx, slots, context, f"c4b{batch}r{repeat}")
            runs.append(analyse_decode(run, batch))
        timed = [item for item in runs if "inter_output_ms" in item]
        row = {
            "batch": batch,
            "context": context,
            "pass": all(item["sequence_ok"] for item in runs),
            "modes_seen": sorted({" ".join(item["modes"]) for item in runs}),
        }
        if timed:
            steps = range(DECODE_TOKENS - 1)
            row["median_inter_output_ms_by_step"] = [
                round(statistics.median(item["inter_output_ms"][step] for item in timed), 3) for step in steps
            ]
            row["median_launch_to_output_ms_by_step"] = [
                round(statistics.median(item["launch_to_output_ms"][step] for item in timed), 3) for step in steps
            ]
            row["median_kept_ms"] = round(statistics.median(item["kept_ms"] for item in timed), 3)
        cases.append(row)
    return {"pass": all(case["pass"] for case in cases), "cases": cases}


def check_seed_reuse(ctx: Context) -> dict:
    limit = ctx.header.get("max_req_input_len")
    limit = int(limit) if isinstance(limit, (int, float)) and limit > 0 else 4096
    deepest = min(ctx.args.reuse_max_context, limit - 16)
    contexts = []
    value = deepest
    while value >= 32 and len(contexts) < 8:
        contexts.append(value)
        value //= 2
    top = min(4, ctx.max_batch(ctx.page))
    points = [(batch, context) for batch in sorted({top, max(1, top // 2), 1}, reverse=True) for context in contexts][
        :20
    ]
    rows, seen_depth = [], {}
    started = time.monotonic()
    for index, (batch, context) in enumerate(points):
        slots = [f"c5-{slot}" for slot in range(batch)]
        seed, run = decode_run(ctx, slots, context, f"c5p{index}")
        # Lint mark B905, kept as saved: a strict= argument would change the call.
        computed = [
            (prompt or 0) - (cached or 0)
            for prompt, cached in zip(seed.meta("prompt_tokens"), seed.meta("cached_tokens"))  # noqa: B905
        ]
        reused = [seen_depth.get(slot, 0) >= context - 3 for slot in slots]
        for slot in slots:
            seen_depth[slot] = max(seen_depth.get(slot, 0), context - 3)
        rows.append(
            {
                "batch": batch,
                "context": context,
                "seed_computed_tokens": computed,
                "seed_was_cached": reused,
                "seed_wall_s": round(seed.wall, 4),
                "run_wall_s": round(run.wall, 4),
                "run_sequence_ok": analyse_decode(run, batch)["sequence_ok"],
            }
        )
    elapsed = time.monotonic() - started
    bound = max(ctx.page, 1)
    # Lint mark B905, kept as saved: a strict= argument would change the call.
    ok = all(
        tokens <= bound
        for row in rows
        for tokens, was in zip(row["seed_computed_tokens"], row["seed_was_cached"])  # noqa: B905
        if was
    )
    with_reuse = sum(sum(row["seed_computed_tokens"]) for row in rows)
    without = sum(row["batch"] * (row["context"] - 3) for row in rows)
    return {
        "pass": ok and all(row["run_sequence_ok"] for row in rows),
        "points": len(rows),
        "seed_tokens_with_reuse": with_reuse,
        "seed_tokens_without_reuse": without,
        "elapsed_s": round(elapsed, 3),
        "seconds_per_point": round(elapsed / max(1, len(rows)), 4),
        "rows": rows,
    }


def kept_times(ctx: Context, slots, context: int, tag: str, stream=False) -> list[float]:
    times = []
    for repeat in range(ctx.args.repeats):
        _, run = decode_run(ctx, slots, context, f"{tag}r{repeat}", stream=stream)
        result = analyse_decode(run, len(slots))
        if "kept_ms" in result:
            times.append(result["kept_ms"])
    return times


def check_streaming(ctx: Context) -> dict:
    batch = min(8, ctx.max_batch(ctx.page))
    slots = [f"c6-{index}" for index in range(batch)]
    context = ctx.args.decode_context
    off = kept_times(ctx, slots, context, "c6off", stream=False)
    on = kept_times(ctx, slots, context, "c6on", stream=True)
    out = {"pass": None, "batch": batch, "context": context, "kept_ms_stream_off": off, "kept_ms_stream_on": on}
    if off and on:
        out["median_off_ms"] = round(statistics.median(off), 3)
        out["median_on_ms"] = round(statistics.median(on), 3)
        out["on_over_off"] = round(out["median_on_ms"] / out["median_off_ms"], 4)
    return out


def check_eviction(ctx: Context) -> dict:
    pool = ctx.header.get("max_total_num_tokens")
    try:
        pool = int(ctx.engine.get_server_info().get("max_total_num_tokens") or pool)
    except Exception:
        pass
    if not pool or pool > ctx.args.fill_limit:
        return {
            "pass": None,
            "skipped": f"KV pool of {pool} tokens is above --fill-limit; "
            "pass --max-total-tokens to SGLang or raise --fill-limit",
        }
    batch = min(8, ctx.max_batch(ctx.page))
    context = ctx.args.decode_context
    flushed = ctx.engine.flush_cache()
    empty = kept_times(ctx, [f"c8e-{index}" for index in range(batch)], context, "c8e")

    limit = ctx.header.get("max_req_input_len")
    limit = int(limit) if isinstance(limit, (int, float)) and limit > 0 else 4096
    length = min(2048, limit - 16)
    filled, index, deadline = 0, 0, time.monotonic() + ctx.args.fill_timeout
    while filled < 1.2 * pool and time.monotonic() < deadline:
        slots = [f"fill-{index}-{slot}" for slot in range(4)]
        stage(ctx, [ctx.content.prefix(slot, length) for slot in slots], 1, f"fill{index}", need=1)
        filled += 4 * length
        index += 1
    full = kept_times(ctx, [f"c8f-{index}" for index in range(batch)], context, "c8f")
    out = {
        "pass": None,
        "pool_tokens": pool,
        "filled_tokens": filled,
        "flush_result": repr(flushed),
        "batch": batch,
        "context": context,
        "kept_ms_empty": empty,
        "kept_ms_full": full,
    }
    if empty and full:
        out["median_empty_ms"] = round(statistics.median(empty), 3)
        out["median_full_ms"] = round(statistics.median(full), 3)
        out["full_over_empty"] = round(out["median_full_ms"] / out["median_empty_ms"], 4)
    return out


def check_over_budget(ctx: Context) -> dict:
    """Informational: what SGLang does with a batch above the prefill step budget."""
    page, budget = ctx.page, ctx.step_budget()
    running = ctx.header.get("max_running_requests")
    running = int(running) if isinstance(running, (int, float)) and running > 0 else 32
    out: dict = {"pass": None, "step_budget": budget, "page_size": page}

    new = max(64, page)
    fits = max(1, budget // ctx.ceil_page(new))
    batch = fits + 8
    if batch <= running:
        slots = [f"c9p-{index}" for index in range(batch)]
        staged = stage(ctx, [ctx.content.prefix(slot, new) for slot in slots], 1, "c9p", need=1)
        out["prefill"] = {
            "batch": batch,
            "largest_batch_that_fits": fits,
            "new_tokens_each": new,
            "extend_batches": [
                (r["forward_iter"], r["bs"], r["extend_num_tokens"]) for r in staged.batches if r["mode"] == "EXTEND"
            ],
        }
    fits = max(1, budget // max(page, 1))
    batch = fits + ctx.args.over_budget_extra
    if batch <= running:
        slots = [f"c9d-{index}" for index in range(batch)]
        runs = []
        for repeat in range(ctx.args.repeats):
            _, run = decode_run(ctx, slots, 256, f"c9dr{repeat}")
            decode = [r for r in run.batches if r["mode"] == "DECODE"]
            runs.append(
                {
                    "batches": [(r["mode"], r["bs"]) for r in run.batches][:16],
                    "decode_batches_hold_all": all(r["bs"] == batch for r in decode),
                    "decode_inter_output_ms": [
                        round((b["out_ts"] - a["out_ts"]) * 1e3, 3)
                        # Lint marks B905 and RUF007, kept as saved: strict= or pairwise() would change the call.
                        for a, b in zip(decode, decode[1:])  # noqa: B905, RUF007
                    ],
                }
            )
        out["decode"] = {"batch": batch, "largest_batch_that_fits": fits, "runs": runs}
    else:
        out["decode"] = {"skipped": f"needs {batch} requests, above the running limit of {running}"}
    return out


# ------------------------------------------------------------------- DP attention


def _dp_plan(ctx: Context, leaders, per_rank: int, length: int, tag: str):
    plan = []
    for rank in sorted(leaders):
        for index in range(per_rank):
            slot = f"{tag}-r{rank}-{index}"
            plan.append((rank, f"{ctx.run}-{slot}", ctx.content.prefix(slot, length)))
    return plan


def check_dp_staging(ctx: Context) -> dict:
    import asyncio

    from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType

    leaders = ctx.logs.leaders()
    if len(leaders) < 2:
        return {"pass": False, "error": f"expected at least two DP ranks, found {sorted(leaders)}"}
    manager = ctx.engine.tokenizer_manager
    plan = _dp_plan(ctx, leaders, 2, max(64, ctx.page), "dp")
    params = {"max_new_tokens": 1, "temperature": 0.0, "ignore_eos": True}

    async def staged():
        loop = asyncio.get_running_loop()
        manager._dispatch_to_scheduler(BlockReqInput(req_type=BlockReqType.BLOCK))
        tasks = []
        try:
            for rank, rid, prompt in plan:
                tasks.append(
                    asyncio.ensure_future(
                        ctx.engine.async_generate(
                            input_ids=prompt, sampling_params=params, rid=rid, routed_dp_rank=rank
                        )
                    )
                )
            deadline = loop.time() + 60
            while True:
                states = [manager.rid_to_state.get(rid) for _, rid, _ in plan]
                if all(state is not None and getattr(state, "dispatched", False) for state in states):
                    break
                if any(task.done() for task in tasks):
                    break
                if loop.time() > deadline:
                    raise TimeoutError("requests were not marked dispatched within 60 s")
                await asyncio.sleep(0.001)
        finally:
            manager._dispatch_to_scheduler(BlockReqInput(req_type=BlockReqType.UNBLOCK))
        return await asyncio.gather(*tasks)

    ctx.engine.loop.run_until_complete(staged())
    time.sleep(0.5)
    ctx.logs.poll()
    ranks, launches, iters = [], [], []
    for rank, path in sorted(leaders.items()):
        rids = [rid for item_rank, rid, _ in plan if item_rank == rank]
        extend = [r for r in ctx.logs.batches_for(path, rids) if r["mode"] == "EXTEND"]
        ok = len(extend) == 1 and set(extend[0]["rids"]) == set(rids)
        if extend:
            launches.append(extend[0]["launch_ts"])
            iters.append(extend[0]["forward_iter"])
        ranks.append(
            {
                "dp_rank": rank,
                "pass": ok,
                "extend_batches": [(r["forward_iter"], r["bs"], r["global_num_tokens"]) for r in extend],
            }
        )
    spread = (max(launches) - min(launches)) * 1e3 if len(launches) == len(leaders) else None
    return {
        "pass": all(item["pass"] for item in ranks) and spread is not None and spread < 50.0,
        "ranks": ranks,
        "launch_spread_ms": None if spread is None else round(spread, 3),
        "forward_iters": iters,
        "input_blocker_enabled": ctx.header.get("input_blocker"),
    }


def check_dp_decode_cost(ctx: Context) -> dict:
    import asyncio

    leaders = ctx.logs.leaders()
    params = {"max_new_tokens": 48, "temperature": 0.0, "ignore_eos": True}

    async def many(plan):
        return await asyncio.gather(
            *[
                ctx.engine.async_generate(input_ids=prompt, sampling_params=params, rid=rid, routed_dp_rank=rank)
                for rank, rid, prompt in plan
            ]
        )

    samples = []
    for repeat in range(ctx.args.repeats):
        # New request ids each repeat, so the log lookup returns this repeat's batches only.
        plan = _dp_plan(ctx, leaders, 4, ctx.args.decode_context, f"dpc{repeat}")
        ctx.engine.loop.run_until_complete(many(plan))
        time.sleep(0.3)
        ctx.logs.poll()
        rids = [rid for rank, rid, _ in plan if rank == min(leaders)]
        decode = [
            r
            for r in ctx.logs.batches_for(leaders[min(leaders)], rids)
            if r["mode"] == "DECODE" and r["bs"] == len(rids)
        ]
        # Lint marks B905 and RUF007, kept as saved: strict= or pairwise() would change the call.
        steps = [(b["out_ts"] - a["out_ts"]) * 1e3 for a, b in zip(decode[15:40], decode[16:41])]  # noqa: B905, RUF007
        if steps:
            samples.append(statistics.median(steps))
        ctx.engine.flush_cache()
    out = {
        "pass": None,
        "input_blocker_enabled": ctx.header.get("input_blocker"),
        "steady_decode_ms_by_repeat": [round(value, 3) for value in samples],
    }
    if samples:
        out["median_steady_decode_ms"] = round(statistics.median(samples), 3)
    return out


CHECKS = {
    "1": ("recorder headers", check_headers),
    "2": ("one batch call, one extend batch", check_batch_call),
    "3": ("page-size formulas", check_page_formulas),
    "4": ("decode steps", check_decode_steps),
    "5": ("seed reuse", check_seed_reuse),
    "6": ("streaming off and on", check_streaming),
    "8": ("empty and full cache", check_eviction),
    "9": ("batch above the step budget", check_over_budget),
}


# --------------------------------------------------------------------------- main


def parse(argv: list[str]):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--checks", default="1,2,3,4,5,6,8")
    parser.add_argument(
        "--install", choices=("hook", "setattr"), default="hook", help="how the recorder class replaces Scheduler"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-low", type=int, default=1000)
    parser.add_argument("--vocab-high", type=int, default=20000)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--decode-context", type=int, default=1024)
    parser.add_argument("--reuse-max-context", type=int, default=4096)
    parser.add_argument("--stage-timeout", type=float, default=120.0)
    parser.add_argument("--fill-limit", type=int, default=2_000_000)
    parser.add_argument("--fill-timeout", type=float, default=600.0)
    parser.add_argument(
        "--over-budget-extra",
        type=int,
        default=8,
        help="check 9: requests above the largest decode batch that fits one step",
    )
    parser.add_argument("--dp-check", action="store_true")
    parser.add_argument("--dp-decode-cost", action="store_true")
    parser.add_argument(
        "--blocker-env", action="store_true", help="set SGLANG_ENABLE_COLOCATED_BATCH_GEN=1 (implied by --dp-check)"
    )
    if "--" in argv:
        split = argv.index("--")
        own, rest = parser.parse_args(argv[:split]), argv[split + 1 :]
    else:
        own, rest = parser.parse_known_args(argv)
    return own, rest


def main(argv: list[str]) -> int:
    own, sglang_argv = parse(argv)
    out = Path(own.out).resolve()
    recorder_dir = out / "recorder" / time.strftime("%Y%m%d-%H%M%S")
    recorder_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(HERE))
    import phase0_recorder

    os.environ[phase0_recorder.ENV_DIR] = str(recorder_dir)
    os.environ[phase0_recorder.ENV_INSTALL] = own.install
    if own.dp_check or own.blocker_env:
        os.environ["SGLANG_ENABLE_COLOCATED_BATCH_GEN"] = "1"

    # sglang.Engine is a lazy-import proxy and cannot be subclassed; use the real class.
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(sglang_argv)
    own.expected_schedulers = int(getattr(server_args, "tp_size", 1) or 1) * int(
        getattr(server_args, "pp_size", 1) or 1
    )

    class SpikeEngine(Engine):
        run_scheduler_process_func = staticmethod(phase0_recorder.recorder_entry)

    log(f"building the engine; recorder logs in {recorder_dir}")
    engine = SpikeEngine(server_args=server_args)
    results: dict = {"argv": argv, "sglang_argv": sglang_argv, "recorder_dir": str(recorder_dir)}
    try:
        logs = Logs(recorder_dir)
        logs.poll()
        ctx = Context(engine, logs, own, Content(own.seed, own.vocab_low, own.vocab_high))
        leaders = logs.leaders()
        results["headers"] = [dict(file=path.name, **header) for path, header in logs.headers.items()]
        if not leaders:
            results["fatal"] = "no recorder header was written; the class replacement did not apply"
            log(results["fatal"])
        else:
            ctx.leader = leaders[min(leaders)]
            ctx.header = logs.headers[ctx.leader]
            log(
                f"sglang {ctx.header.get('sglang_version')}, page size {ctx.page}, "
                f"overlap {ctx.header.get('enable_overlap')}, {len(logs.headers)} scheduler process(es)"
            )
            if own.dp_check or own.dp_decode_cost:
                selected = [("dp", "DP attention staging", check_dp_staging)] if own.dp_check else []
                if own.dp_decode_cost:
                    selected.append(("dp-cost", "DP steady decode", check_dp_decode_cost))
            else:
                stage(ctx, [ctx.content.prefix("warmup", 256)], 5, "warmup")
                selected = [(key, *CHECKS[key]) for key in own.checks.split(",") if key in CHECKS]
            for key, name, function in selected:
                log(f"check {key}: {name}")
                started = time.monotonic()
                try:
                    outcome = function(ctx)
                except Exception:
                    outcome = {"pass": False, "error": traceback.format_exc()}
                outcome["seconds"] = round(time.monotonic() - started, 2)
                results[f"check_{key}"] = {"name": name, **outcome}
                log(f"check {key}: pass={outcome.get('pass')} ({outcome['seconds']} s)")
            logs.poll()
            results["faults"] = logs.faults
    finally:
        try:
            engine.shutdown()
        except Exception:
            pass
    (out / "results.json").write_text(json.dumps(results, indent=2, default=repr) + "\n")
    log(f"wrote {out / 'results.json'}")
    failed = [key for key, value in results.items() if key.startswith("check_") and value.get("pass") is False]
    if results.get("fatal") or results.get("faults") or failed:
        log(f"FAILED: {failed or results.get('fatal') or 'recorder faults'}")
        return 1
    log("all selected checks passed or are informational")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
