#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject known AI identities in authorship metadata for newly introduced commits."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from email.utils import parseaddr

# Match identities, not arbitrary mentions of tools in commit prose. Keep human
# names such as Claude Shannon and employee addresses at AI companies valid.
BOT_NAME = re.compile(
    r"^(?:(?:openai[ -]+)?codex|claude(?:[ -]+(?:code|opus|sonnet|haiku|fable))?"
    r"|(?:github[ -]+)?copilot|cursor(?:[ -]+agent)?|devin|aider"
    r"|gemini(?:[ -]+(?:cli|code[ -]+assist))?|jules|swe-agent)"
    r"(?:$|\[bot\]$|[ -]+(?:bot|agent)$|[ -]+\d.*$)",
    re.IGNORECASE,
)
BOT_EMAIL = re.compile(
    r"^(?:(?:noreply|no-reply)@(?:anthropic\.com|openai\.com|cursor\.com|cursor\.sh|cognition\.ai)"
    r"|codex@openai\.com|claude@anthropic\.com|copilot@github\.com"
    r"|(?:\d+\+)?(?:codex|claude|copilot|copilot-swe-agent|github-copilot"
    r"|cursor-agent|devin-ai-integration|gemini-code-assist|google-labs-jules)"
    r"(?:\[bot\])?@users\.noreply\.github\.com)$",
    re.IGNORECASE,
)
SIGNATURE = re.compile(r"^\s*(?:co-authored-by|signed-off-by)\s*:\s*(.+)$", re.IGNORECASE)


def is_ai_identity(identity: str) -> bool:
    # GitHub bot email local parts contain literal [bot], which RFC email
    # parsers can discard. Git identities conventionally use angle brackets.
    match = re.fullmatch(r"\s*(.*?)\s*<([^<>]+)>\s*", identity)
    name, email = match.groups() if match else parseaddr(identity)
    return bool(BOT_NAME.fullmatch(name.strip()) or BOT_EMAIL.fullmatch(email.strip()))


def violations(author: str, message: str) -> list[str]:
    errors = []
    if is_ai_identity(author):
        errors.append(f"author: {author}")
    # Scan every line: squash messages can contain signatures from multiple
    # commits outside the final Git trailer block.
    for line in message.splitlines():
        match = SIGNATURE.match(line)
        if match and is_ai_identity(match[1]):
            errors.append(line.strip())
    return list(dict.fromkeys(errors))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Exclusive base commit")
    parser.add_argument("--head", required=True, help="Inclusive head commit")
    args = parser.parse_args()
    # Resolve to immutable commit IDs before using either argument in a range.
    head = git("rev-parse", "--verify", "--end-of-options", f"{args.head}^{{commit}}").strip()
    if re.fullmatch(r"0+", args.base):
        revision = head  # Initial branch push: all reachable commits are new.
    else:
        base = git("rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}").strip()
        revision = f"{base}..{head}"
    commits = git("rev-list", revision).splitlines()
    failed = False
    for commit in commits:
        author, message = git("show", "-s", "--format=%an <%ae>%n%B", commit).split("\n", 1)
        for error in violations(author, message):
            print(f"{commit}: forbidden AI attribution: {error}", file=sys.stderr)
            failed = True
    if failed:
        print(
            "Use the responsible human author and remove AI authorship/sign-off lines; "
            "amend or rebase the affected commits and push again.",
            file=sys.stderr,
        )
        return 1
    print(f"Checked {len(commits)} commit(s): no known AI authorship found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
