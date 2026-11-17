# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal durable-output contract for the unified CLI."""

from __future__ import annotations

import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import yaml

from .detail import format_prediction_details
from .power import format_power_summary

if TYPE_CHECKING:
    from .sweeper.result import SweepResult

_RECOMMENDATION_NAME = re.compile(r"^[0-9]{4}\.yaml$")
_COMPACT_JSON = json.JSONEncoder(sort_keys=True, separators=(",", ":"))


# Files a finished run leaves behind. Their presence means the directory holds
# results that a retry must not silently replace.
_RESULT_FILES = frozenset(
    {
        "prediction.json",
        "report.html",
        "recommendation.json",
        "recommendation.csv",
        "requests.jsonl",
        "telemetry-requests.jsonl",
        "telemetry-samples.jsonl",
        "telemetry-lifecycle.jsonl",
    }
)
# Diagnostics that runs write whether or not they succeed.
_DIAGNOSTIC_FILES = frozenset(
    {
        "fpm-coverage.json",
        "resource-plan.json",
        "resource-runtime.json",
        "execution-events.jsonl",
        "afd-replay-spec.json",
        "afd-qualification.json",
    }
)
_KNOWN_FILES = _RESULT_FILES | _DIAGNOSTIC_FILES


# Terminal statuses that ``supervision.run_process`` records in resource-runtime.json
# for a run that did not complete. Anything else, including a missing or
# unreadable report, means the directory is preserved.
_FAILED_RUN_STATUSES = frozenset({"failed", "resource_limited", "timed_out", "cancelled"})
_RUNTIME_REPORT = "resource-runtime.json"

_Fingerprint = tuple[int, int, int, int, bool]


def _fingerprint(path: Path) -> _Fingerprint:
    info = path.lstat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, stat.S_ISDIR(info.st_mode))


def _read_runtime_report(path: Path) -> dict[str, Any] | None:
    """Read a runtime report only if it is a regular file; never block on a FIFO or follow a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            report = json.loads(source.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    finally:
        os.close(descriptor)
    return report if isinstance(report, dict) else None


def _runtime_status(report: dict[str, Any] | None) -> str | None:
    """The status of a schema-1 runtime report, or None for anything malformed."""

    if report is None:
        return None
    version = report.get("schema_version")
    # bool is an int subclass, and True == 1; a JSON true is not schema version 1.
    if type(version) is not int or version != 1:
        return None
    status = report.get("status")
    return status if isinstance(status, str) else None


def _failed_runtime_status(path: Path) -> str | None:
    """Return the status of a valid runtime report that records a failed run, else None."""

    status = _runtime_status(_read_runtime_report(path))
    return status if status in _FAILED_RUN_STATUSES else None


def _unfinished_run_leftovers(root: Path) -> dict[str, _Fingerprint] | None:
    """Fingerprint the files of a failed AISim run, or return None to preserve the directory.

    A directory qualifies only when every entry is a regular AISim diagnostic
    file (or an empty ``recommendations/`` directory), none is a symlink, and a
    valid ``resource-runtime.json`` records one of the failed-run statuses.
    """

    leftovers: dict[str, _Fingerprint] = {}
    for entry in root.iterdir():
        try:
            mode = entry.lstat().st_mode
        except OSError:
            return None
        if entry.name == "recommendations" and stat.S_ISDIR(mode):
            if any(entry.iterdir()):
                return None
        elif entry.name not in _DIAGNOSTIC_FILES or not stat.S_ISREG(mode):
            return None
        leftovers[entry.name] = _fingerprint(entry)
    if _RUNTIME_REPORT not in leftovers or _failed_runtime_status(root / _RUNTIME_REPORT) is None:
        return None
    return leftovers


def _clear_unfinished_run(root: Path, leftovers: dict[str, _Fingerprint]) -> None:
    """Remove the fingerprinted leftovers, stopping at the first file that changed.

    Each file is checked again just before it is removed, and the runtime report
    goes last, so a directory another process starts using keeps its evidence.
    No output-directory lock exists yet; see the PR notes on the remaining race.
    """

    for name in sorted(leftovers, key=lambda item: item == _RUNTIME_REPORT):
        target = root / name
        try:
            unchanged = _fingerprint(target) == leftovers[name]
        except FileNotFoundError:
            unchanged = False
        if not unchanged:
            raise ValueError(
                f"output directory {root} changed while AISim was preparing it ({name}); "
                "nothing else was removed. Choose a new --output-dir"
            )
        if leftovers[name][4]:
            target.rmdir()
        else:
            target.unlink()
    if any(root.iterdir()):
        raise ValueError(f"output directory {root} changed while AISim was preparing it; choose a new --output-dir")


def prepare_output_directory(path: str | Path, *, overwrite: bool) -> Path:
    """Create an output directory without deleting unrelated user files.

    A directory that holds only the diagnostics of a failed AISim run (a runtime
    report with status failed, resource_limited, timed_out or cancelled) is
    cleared and reused, so the obvious retry works without ``--overwrite``.
    Finished results, files that AISim did not write, and diagnostics without a
    failed-run runtime report still require a new directory or ``--overwrite``.
    """

    root = Path(path)
    if root.exists() and not root.is_dir():
        raise ValueError(f"output path {root} exists and is not a directory")
    if root.exists() and any(root.iterdir()) and not overwrite:
        leftovers = _unfinished_run_leftovers(root)
        if leftovers is None:
            raise ValueError(_non_empty_message(root))
        _clear_unfinished_run(root, leftovers)
        sys.stderr.write(f"aisim: reusing {root}: removed diagnostics left by a failed run\n")
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in sorted(_KNOWN_FILES):
            target = root / name
            if target.is_file() or target.is_symlink():
                target.unlink()
        recommendations = root / "recommendations"
        if recommendations.is_dir():
            for target in recommendations.iterdir():
                if _RECOMMENDATION_NAME.fullmatch(target.name) and (target.is_file() or target.is_symlink()):
                    target.unlink()
    return root


def _non_empty_message(root: Path) -> str:
    names = {entry.name for entry in root.iterdir()}
    unknown = sorted(name for name in names if name not in _KNOWN_FILES and name != "recommendations")
    advice = "choose a new --output-dir to keep them, or pass --overwrite to replace known AISim outputs"
    if unknown:
        shown = ", ".join(unknown[:3]) + (", ..." if len(unknown) > 3 else "")
        return (
            f"output directory {root} is not empty and contains files AISim did not write ({shown}); "
            "choose a new --output-dir (--overwrite removes only known AISim outputs)"
        )
    # The same safe read as eligibility: a FIFO or symlink here must not block or be followed.
    status = _runtime_status(_read_runtime_report(root / _RUNTIME_REPORT))
    has_results = bool(names & _RESULT_FILES) or (
        (root / "recommendations").is_dir() and any((root / "recommendations").iterdir())
    )
    if has_results and isinstance(status, str) and status != "completed":
        held = f"results from an earlier AISim run that ended with status {status!r}"
    elif has_results or status == "completed":
        held = "results from a finished AISim run"
    elif _RUNTIME_REPORT not in names:
        held = "AISim diagnostics without a runtime report that records a failed run"
    else:
        held = "AISim diagnostics whose runtime report does not record a failed run"
    return f"output directory {root} is not empty and holds {held}; {advice}"


def _open_result(path: Path, *, exclusive: bool):
    """Open a result file for writing.

    With ``exclusive`` (the CLI, after ``prepare_output_directory``), a file that
    appeared after preparation makes the write fail instead of being replaced.
    """

    if not exclusive:
        return path.open("w", encoding="utf-8")
    try:
        return path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"{path} appeared after the output directory was prepared; refusing to replace it") from error


def write_prediction_report(root: Path, report: dict[str, Any], *, exclusive: bool = False) -> Path:
    path = root / "prediction.json"
    # json.dump writes the chunks json.dumps would join, so a large per-request
    # report is never held as one string. A failed encoding leaves no report, as before.
    output = _open_result(path, exclusive=exclusive)
    identity = _file_identity(output)
    try:
        with output:
            json.dump(report, output, indent=2, sort_keys=True)
            output.write("\n")
    except BaseException:
        _unlink_if_same(path, identity)
        raise
    return path


def write_fpm_coverage(root: Path, coverage: dict[str, Any]) -> Path:
    """Persist native lookup evidence even when the replay fails."""

    path = root / "fpm-coverage.json"
    path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_requests(root: Path, records: list[dict[str, Any]], *, exclusive: bool = False) -> Path:
    path = root / "requests.jsonl"
    with _open_result(path, exclusive=exclusive) as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")
    return path


def _file_identity(output: Any) -> tuple[int, int]:
    status = os.fstat(output.fileno())
    return status.st_dev, status.st_ino


def _unlink_if_same(path: Path, identity: tuple[int, int]) -> None:
    """Roll back a file this call created, but never one another writer put in its place."""

    try:
        status = path.lstat()
    except FileNotFoundError:
        return
    if (status.st_dev, status.st_ino) == identity:
        path.unlink(missing_ok=True)


def write_telemetry(root: Path, telemetry: dict[str, Any], *, exclusive: bool = False) -> dict[str, Any]:
    """Write all-phase telemetry streams and return their compact report manifest."""

    streams = ("requests", "samples", "lifecycle")
    for stream in streams:
        records = telemetry.get(stream)
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError(f"telemetry.{stream} must be a list of JSON mappings")
    paths = {stream: root / f"telemetry-{stream}.jsonl" for stream in streams}
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for stream, path in paths.items():
            with _open_result(path, exclusive=exclusive) as output:
                created.append((path, _file_identity(output)))
                for record in telemetry[stream]:
                    json.dump(record, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
                    output.write("\n")
    except BaseException:
        for path, identity in created:
            _unlink_if_same(path, identity)
        raise
    return {
        **{key: value for key, value in telemetry.items() if key not in streams},
        "artifacts": {stream: path.name for stream, path in paths.items()},
        "record_counts": {stream: len(telemetry[stream]) for stream in streams},
    }


def write_recommendation_result(root: Path, result: SweepResult, *, exclusive: bool = False) -> Path:
    """Write the lossless recommendation ledger and derived candidate views."""

    path = root / "recommendation.json"
    with _open_result(path, exclusive=exclusive) as output:
        output.write(result.to_json() + "\n")
    return path


def write_recommendation_csv(root: Path, result: SweepResult, *, exclusive: bool = False) -> Path:
    """Write the complete candidate ledger as an analysis-friendly CSV."""

    path = root / "recommendation.csv"
    with _open_result(path, exclusive=exclusive) as output:
        output.write(result.to_csv())
    return path


def write_recommendations(root: Path, configs: list[Mapping[str, Any]], *, exclusive: bool = False) -> list[Path]:
    directory = root / "recommendations"
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, config in enumerate(configs, start=1):
        path = directory / f"{index:04d}.yaml"
        with _open_result(path, exclusive=exclusive) as output:
            output.write(
                yaml.safe_dump(
                    dict(config),
                    sort_keys=False,
                    allow_unicode=True,
                )
            )
        paths.append(path)
    return paths


def format_prediction_stdout(
    summary: dict[str, Any],
    output_format: str,
    *,
    details: dict[str, Any] | None = None,
    power_diagnostics: dict[str, Any] | None = None,
    diagnostics_top_n: int = 12,
) -> str:
    from .replay.reporting import format_power_diagnostics, format_report_table

    if output_format == "json":
        return _COMPACT_JSON.encode(_prediction_json_payload(summary, details, power_diagnostics))
    if details is not None:
        rendered = (
            format_prediction_stdout(summary, output_format)
            + "\n\n"
            + format_prediction_details(details, energy_top_n=diagnostics_top_n)
        )
        if power_diagnostics is not None and "energy" not in details["sections"]:
            rendered += "\n\n" + format_power_diagnostics(power_diagnostics, top_n=diagnostics_top_n)
        return rendered
    if power_diagnostics is not None:
        return (
            format_prediction_stdout(summary, output_format)
            + "\n\n"
            + format_power_diagnostics(power_diagnostics, top_n=diagnostics_top_n)
        )
    provenance_lines = [
        f"KV cache ({role}): {precision['dtype']} (source: {precision['source']})"
        for role, precision in summary.get("kv_cache_precision", {}).items()
    ]
    for role, provenance in summary.get("prediction_provenance", {}).items():
        if reason := provenance.get("unavailable_reason"):
            provenance_lines.append(f"Timing ({role}): unavailable ({reason})")
            continue
        version = provenance.get("data_version")
        if version is None:
            version = f"unavailable ({provenance['data_version_unavailable_reason']})"
        line = (
            f"Timing ({role}): estimator={provenance['estimator']}; "
            f"source={provenance['timing_source']}; data_version={version}"
        )
        if (mode := provenance.get("database_mode")) is not None:
            line += f"; database_mode={mode}"
        provenance_lines.append(line)
    invalid_banner = ""
    if summary.get("prediction_status") == "invalid":
        invalid_banner = f"INVALID prediction ({summary['stop_reason']}): {summary['diagnostic']}\n"
    if summary.get("metric_semantics") == "analytical_epd_overlay":
        lines = ["AISim analytical EPD (aggregate estimates; no encoder queue simulation)"]
        for name in (
            "mean_ttft_ms",
            "mean_tpot_ms",
            "mean_e2e_latency_ms",
            "output_throughput_tok_s",
            "completed_requests",
            "duration_ms",
            "gpu_hours",
            "encoder_gpus",
            "total_gpus",
        ):
            lines.append(f"{name}: {summary.get(name, 'N/A')}")
        lines.append(format_power_summary(summary))
        lines.extend(provenance_lines)
        lines.append("duration_ms is a rate-derived accounting interval, not an EPD event timeline.")
        return invalid_banner + "\n".join(lines)

    table = invalid_banner + format_report_table(summary)
    if provenance_lines:
        table += "\n" + "\n".join(provenance_lines)
    if summary.get("agentic_qualification") == "functional_only":
        return "AgentX functional replay only; not an AgentX benchmark result.\n" + table
    return table


def write_prediction_stdout(
    output: TextIO,
    summary: dict[str, Any],
    output_format: str,
    *,
    details: dict[str, Any] | None = None,
    power_diagnostics: dict[str, Any] | None = None,
    diagnostics_top_n: int = 12,
) -> None:
    """Write format_prediction_stdout's text without building JSON output as one string."""

    if output_format == "json":
        _write_compact_json(output, _prediction_json_payload(summary, details, power_diagnostics))
        return
    output.write(
        format_prediction_stdout(
            summary,
            output_format,
            details=details,
            power_diagnostics=power_diagnostics,
            diagnostics_top_n=diagnostics_top_n,
        )
    )


def _prediction_json_payload(
    summary: dict[str, Any],
    details: dict[str, Any] | None,
    power_diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    if power_diagnostics is not None:
        return {
            "summary": summary,
            "power_diagnostics": power_diagnostics,
            **({"details": details} if details is not None else {}),
        }
    return summary if details is None else {"summary": summary, "details": details}


def _write_compact_json(output: TextIO, value: Any, _enclosing: frozenset[int] = frozenset()) -> None:
    """Write _COMPACT_JSON.encode(value) piecewise.

    Objects are walked key by key and lists element by element, so a large
    per-request list is never one string. Every leaf and list element goes
    through the same encoder, keeping the bytes identical.
    """

    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        if id(value) in _enclosing:
            raise ValueError("Circular reference detected")
        enclosing = _enclosing | {id(value)}
        output.write("{")
        for index, key in enumerate(sorted(value)):
            if index:
                output.write(",")
            output.write(_COMPACT_JSON.encode(key))
            output.write(":")
            _write_compact_json(output, value[key], enclosing)
        output.write("}")
    elif isinstance(value, list):
        output.write("[")
        for index, item in enumerate(value):
            if index:
                output.write(",")
            output.write(_COMPACT_JSON.encode(item))
        output.write("]")
    else:
        output.write(_COMPACT_JSON.encode(value))


def format_recommendation_stdout(rows: list[dict[str, Any]], output_format: str) -> str:
    if output_format == "json":
        return json.dumps(rows, sort_keys=True, separators=(",", ":"))
    if not rows:
        return "No feasible candidate found."
    lines = ["AISim recommendations"]
    for row in rows:
        objective = row.get("objectives") or {"score": row.get("score")}
        metrics = ", ".join(f"{key}={value:.4g}" for key, value in objective.items())
        power = " " + format_power_summary(row)
        lines.append(f"{row['rank']}: {metrics} used_gpus={row['used_gpus']}{power} config={row['config_path']}")
    return "\n".join(lines)
