# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity matching and real Git range regression tests for the CI policy."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from check import is_ai_identity, violations

SCRIPT = Path(__file__).with_name("check.py")


class AuthorshipTests(unittest.TestCase):
    def test_known_ai_identities(self):
        for identity in (
            "Codex <codex@openai.com>",
            "OpenAI Codex <bot@example.com>",
            "Other name <codex@openai.com>",
            "Claude <bot@example.com>",
            "Claude Fable 5.1 <noreply@anthropic.com>",
            "Claude Opus 5 (1M context) <noreply@anthropic.com>",
            "CLAUDE CODE <bot@example.com>",
            "Bot <123+claude[bot]@users.noreply.github.com>",
            "copilot-swe-agent[bot] <123+copilot-swe-agent[bot]@users.noreply.github.com>",
            "GitHub Copilot <bot@example.com>",
            "Cursor Agent <bot@example.com>",
            "Devin <bot@example.com>",
            "Gemini CLI <bot@example.com>",
            "Jules <bot@example.com>",
        ):
            with self.subTest(identity=identity):
                self.assertTrue(is_ai_identity(identity))

    def test_human_and_non_ai_automation(self):
        for identity in (
            "Claude Shannon <claude@example.com>",
            "Jane Smith <jane@anthropic.com>",
            "Jane Smith <jane@openai.com>",
            "GitHub <noreply@github.com>",
            "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
            "Harry Lee <harrli@nvidia.com>",
        ):
            with self.subTest(identity=identity):
                self.assertFalse(is_ai_identity(identity))

    def test_embedded_and_case_insensitive_signatures(self):
        message = (
            "fix: use Claude API\n\n"
            "  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\n"
            "\nAnother squashed commit\n\n"
            "signed-off-by: Codex <codex@openai.com>\n"
            "Co-authored-by: Jane Smith <jane@example.com>\n"
        )
        self.assertEqual(len(violations("Jane Smith <jane@example.com>", message)), 2)
        self.assertEqual(violations("Jane Smith <jane@example.com>", "fix: Claude / Codex support"), [])

    def test_author_and_range_with_real_commits(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}

            def git(*args):
                return subprocess.check_output(["git", *args], cwd=directory, env=env, text=True).strip()

            def commit(name, message):
                git(
                    "-c",
                    "user.name=Human",
                    "-c",
                    "user.email=human@example.com",
                    "-c",
                    "commit.gpgsign=false",
                    "commit",
                    "--allow-empty",
                    f"--author={name} <author@example.com>",
                    "-m",
                    message,
                )
                return git("rev-parse", "HEAD")

            def check(base, head):
                return subprocess.run(
                    [sys.executable, str(SCRIPT), "--base", base, "--head", head],
                    cwd=directory,
                    env=env,
                    text=True,
                    capture_output=True,
                )

            git("init", "-q")
            old = commit("Codex", "historical violation")
            human = commit("Human", "fix: add Claude support")
            self.assertEqual(check(old, human).returncode, 0)
            bot = commit("Claude", "automated author")
            result = check(human, bot)
            self.assertEqual(result.returncode, 1)
            self.assertIn(bot, result.stderr)
            self.assertNotIn(old, result.stderr)
            coauthor = commit("Human", "fix: test\n\nCo-authored-by: Codex <codex@openai.com>")
            self.assertEqual(check(bot, coauthor).returncode, 1)
            self.assertEqual(check("0" * 40, coauthor).returncode, 1)
            self.assertNotEqual(check("missing-ref", human).returncode, 0)


if __name__ == "__main__":
    unittest.main()
