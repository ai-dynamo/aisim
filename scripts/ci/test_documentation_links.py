# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral coverage for local documentation destination checks."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.ci.check_documentation_links import check_file, documentation_files, local_links


class DocumentationLinksTest(unittest.TestCase):
    def test_default_scan_finds_broken_links_in_agent_guides_and_skill_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            guides = [
                "REVIEW.md",
                "crates/example/AGENTS.md",
                ".agents/skills/example/references/verification.md",
                "skills/example/SKILL.md",
                "python/aisim/.claude/rules/example.md",
            ]
            for name in [*guides, "scratch.md"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("[example](missing.md)")
            failures = [failure for page in documentation_files(root) for failure in check_file(page, root)]
            self.assertCountEqual(failures, [f"{name}:1: missing local destination: missing.md" for name in guides])

    def test_extracts_inline_images_references_and_balanced_destinations(self):
        cases = {
            "[guide](guide.md#setup)": "guide.md",
            "![plot](plot.png)": "plot.png",
            '[named](<my guide.md> "Title")': "my guide.md",
            "[nested](name_(one).md)": "name_(one).md",
            '[ref]: ../guide.md "Title"': "../guide.md",
            "[encoded](my%20guide.md)": "my guide.md",
            r"[escaped](name_\(one\).md)": "name_(one).md",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                self.assertEqual(local_links(source), [(1, target)])
        self.assertEqual(
            local_links("[use][ref]\n\n[ref]: guide.md"),
            [(1, "guide.md"), (3, "guide.md")],
        )

    def test_ignores_code_comments_and_nonlocal_destinations(self):
        text = (
            "<!-- [hidden](absent.md) -->\n"
            "```md\n[example](absent.md)\n```\n"
            "~~~md\n[example](absent.md)\n~~~\n"
            "`[inline](absent.md)` [web](https://example.com/a)\n"
            "[mail](mailto:docs@example.com) [anchor](#heading) [cdn](//example.com/a)\n"
            "\n[real](real.md)\n"
        )
        self.assertEqual(local_links(text), [(11, "real.md")])

    def test_ignores_indented_code_but_checks_nested_list_links(self):
        text = "Example:\n\n    [example](absent.md)\n\n- Parent\n  - [real](real.md)\n"
        self.assertEqual(local_links(text), [(6, "real.md")])

    def test_ignores_code_in_blockquotes_and_lists(self):
        text = (
            "> ~~~md\n> [example](absent.md)\n> ~~~\n\n"
            "- Example:\n\n    ~~~md\n    [example](absent.md)\n    ~~~\n\n"
            "[real](real.md)\n"
        )
        self.assertEqual(local_links(text), [(11, "real.md")])

    def test_checks_multiline_link_labels(self):
        self.assertEqual(
            local_links("[Sweep Configuration\nProviders](provider.md)"),
            [(1, "provider.md")],
        )

    def test_checks_code_formatted_link_labels(self):
        self.assertEqual(local_links("[`file.py`](file.py)"), [(1, "file.py")])

    def test_reports_deleted_target_and_repository_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            docs = root / "docs"
            docs.mkdir()
            target = root / "source.py"
            target.write_text("")
            page = docs / "guide.md"
            page.write_text("[source](../source.py) [folder](../docs/)\n")
            self.assertEqual(check_file(page, root), [])
            target.unlink()
            self.assertEqual(
                check_file(page, root),
                ["docs/guide.md:1: missing local destination: ../source.py"],
            )
            page.write_text("[escape](../../)")
            self.assertEqual(
                check_file(page, root),
                ["docs/guide.md:1: missing local destination: ../../"],
            )


if __name__ == "__main__":
    unittest.main()
