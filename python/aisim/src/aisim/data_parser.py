# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsers for ``aisim data``, ``aisim fetch`` and ``aisim init``.

Kept free of dataset-library imports so building the CLI parser stays
runtime-free; :mod:`aisim.data_cli` holds the command implementation.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_CATEGORIES = ("op-latency", "fpm", "e2e", "eval")
#: Not a dataset category: the config.json files of Hugging Face models AISim does not bundle.
MODEL_CONFIG_CATEGORY = "model-config"
PURPOSES = ("prediction", "validation", "results", "full-evidence")
# Library entry purposes that hold the data a predictor reads, per category.
# ``full-evidence`` applies no purpose filter. ``validation`` and ``results``
# select evidence tiers that the library cannot filter yet.
_PREDICTION_PURPOSE = {"op-latency": "operation-tables", "fpm": "execution-profile"}


def _add_source_args(parser: Any) -> None:
    parser.add_argument(
        "--repo",
        help="Hugging Face dataset repository, owner/name (default: $AISIM_DATASET, the saved init config, "
        "then ai-dynamo/aisim-data)",
    )
    parser.add_argument(
        "--revision",
        help="branch, tag or commit SHA (default: $AISIM_DATASET, the saved init config, then the shipped pin; "
        "'main' when the package ships no pin)",
    )
    parser.add_argument("--cache-dir", help="cache root (default: $AISIM_DATA_CACHE or the user cache)")


def _add_fetch_args(parser: Any) -> None:
    parser.add_argument(
        "--category",
        default="op-latency",
        help=f"one of: {', '.join((*SUPPORTED_CATEGORIES, MODEL_CONFIG_CATEGORY))} (default: op-latency; fpm, e2e and "
        f"eval need selectors; {MODEL_CONFIG_CATEGORY} caches Hugging Face model metadata, not dataset files)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="whole category; op-latency only. This is also what no selector means for op-latency",
    )
    parser.add_argument(
        "--model",
        help=f"model selector; for {MODEL_CONFIG_CATEGORY}, Hugging Face model IDs, comma-separated "
        "(default: the manifest AISim ships)",
    )
    parser.add_argument("--hardware")
    parser.add_argument("--framework")
    parser.add_argument("--framework-version")
    parser.add_argument(
        "--configuration",
        action="append",
        default=[],
        metavar="ID",
        help="configuration ID; repeat or comma-separate for several",
    )
    parser.add_argument("--operation-family")
    parser.add_argument("--purpose", choices=PURPOSES, default="prediction")
    _add_source_args(parser)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan; downloads only the small catalog and manifest files"
    )
    parser.add_argument("--offline", action="store_true", help="make no network requests")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=f"{MODEL_CONFIG_CATEGORY} only: resolve the model's branch again instead of reusing the cached commit",
    )
    parser.add_argument(
        "--no-limit", action="store_true", help="ignore the opt-in AISIM_DATA_MAX_BYTES limit for this download"
    )


def add_data_parser(subparsers: Any) -> None:
    data = subparsers.add_parser("data", help="Download, inspect and validate the AISim dataset.")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    fetch_parser = data_commands.add_parser(
        "fetch",
        help="Download dataset files into the local cache (no arguments: the whole op-latency database).",
        description=(
            "Resolve a dataset revision to one commit SHA, plan the files a selection needs, and download and "
            "verify them into the cache. With no arguments this downloads the whole op-latency database. fpm, "
            "e2e and eval are never part of that default: they need --configuration or --hardware (plus "
            "--model). Also available as 'aisim fetch'."
        ),
    )
    _add_fetch_args(fetch_parser)
    status = data_commands.add_parser(
        "status", help="Show the pin, cache path, size on disk, cached versus missing files and the init mode."
    )
    status.add_argument("--repo")
    status.add_argument("--revision")
    status.add_argument("--cache-dir")
    clear = data_commands.add_parser(
        "clear", help="Remove cached dataset revisions and the saved init config.", description="Asks to confirm."
    )
    clear.add_argument("--cache-dir")
    clear.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
    validate = data_commands.add_parser(
        "validate",
        help="Check a dataset directory, one manifest, or a Hugging Face dataset revision.",
        description=(
            "Check names, manifests, digests and catalog consistency. PATH is a dataset root, a category "
            "directory or one manifest.json and is checked locally with no network. --repo (and --revision) "
            "fetches that Hugging Face revision first. Prints one JSON object; exit status is 0 when clean "
            "and 1 on findings."
        ),
    )
    validate.add_argument("path", nargs="?", help="dataset root, a category directory, or one manifest.json")
    validate.add_argument("--repo", help="Hugging Face dataset repository to fetch and validate (needs network)")
    validate.add_argument(
        "--revision",
        help="branch, tag or commit SHA (default: as for fetch, the saved init source, else the shipped pin)",
    )
    validate.add_argument("--cache-dir", help="cache root for --repo")
    add_init_parser(subparsers)
    fetch_alias = subparsers.add_parser(
        "fetch", help="Alias of 'aisim data fetch'.", description="Alias of 'aisim data fetch'."
    )
    _add_fetch_args(fetch_alias)
    fetch_alias.set_defaults(data_command="fetch")


def add_init_parser(subparsers: Any) -> None:
    init = subparsers.add_parser(
        "init",
        help="Choose how AISim gets op-latency data: the whole database now, or on demand.",
        description=(
            "In a terminal, asks: 1) op database now, 2) op data on demand. Without a terminal, pass --all, "
            "--on-demand or --hardware. The choice is saved in the cache root and 'aisim data clear' removes it. "
            "--all and --hardware also cache the config.json of the Hugging Face models AISim lists in its model "
            "manifest (--no-models skips them); a gated or denied model is skipped with a warning."
        ),
    )
    group = init.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="download the whole op-latency database now")
    group.add_argument(
        "--on-demand",
        action="store_true",
        help="download nothing now; a run fetches only from a pinned source (--repo/--revision, AISIM_DATASET), "
        "else it uses the default data (the dataset revision pinned in the release)",
    )
    group.add_argument("--hardware", metavar="H", help="download op data for one hardware platform now")
    _add_source_args(init)
    init.add_argument(
        "--offline", action="store_true", help="make no network requests (downloads come from the cache only)"
    )
    init.add_argument(
        "--no-models",
        action="store_true",
        help="do not cache the Hugging Face model configs of the manifest (--all and --hardware cache them)",
    )
    init.add_argument("--no-limit", action="store_true", help="ignore the opt-in AISIM_DATA_MAX_BYTES limit")
    init.set_defaults(data_command="init")
