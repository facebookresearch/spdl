# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from spdl.autoresearch import load_knowledge, load_prompt, load_prompt_directory


class LoadPromptTest(unittest.TestCase):
    def test_loads_existing_prompt(self) -> None:
        """A valid prompt name returns non-empty template content."""
        result = load_prompt("analyze", KNOWLEDGE="test-knowledge")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_substitutes_placeholders(self) -> None:
        """All __KEY__ placeholders are replaced with the supplied values."""
        result = load_prompt(
            "headspace",
            KNOWLEDGE="INJECTED_KNOWLEDGE",
            PIPELINE_SCRIPT="/tmp/test.py",
            PIPELINE_CODE="def main(): pass",
        )

        self.assertIn("INJECTED_KNOWLEDGE", result)
        self.assertIn("/tmp/test.py", result)
        self.assertIn("def main(): pass", result)
        self.assertNotIn("__KNOWLEDGE__", result)
        self.assertNotIn("__PIPELINE_SCRIPT__", result)
        self.assertNotIn("__PIPELINE_CODE__", result)

    def test_missing_prompt_exits(self) -> None:
        """Requesting a nonexistent prompt template triggers SystemExit."""
        with self.assertRaises(SystemExit):
            load_prompt("nonexistent_prompt_that_does_not_exist")

    def test_headspace_prompt_requires_stop_after(self) -> None:
        """The headspace prompt instructs the agent to include stop_after=500."""
        prompt = load_prompt(
            "headspace",
            KNOWLEDGE="",
            PIPELINE_SCRIPT="/tmp/pipeline.py",
            PIPELINE_CODE="def main():\n    pass\n",
        )

        self.assertIn("stop_after=500", prompt)
        self.assertIn("must include `stop_after=500`", prompt)

    def test_all_phase_prompts_loadable(self) -> None:
        """Every phase prompt shipped with the package loads without error."""
        phase_prompts = [
            "analyze",
            "apply_changes",
            "apply_startup_repair",
            "assess",
            "headspace",
            "instrument",
            "plan_next",
        ]
        for name in phase_prompts:
            with self.subTest(prompt=name):
                result = load_prompt(name, KNOWLEDGE="k")
                self.assertIsInstance(result, str)
                self.assertGreater(len(result), 0)


class LoadPromptDirectoryTest(unittest.TestCase):
    def test_loads_knowledge_directory(self) -> None:
        """The knowledge directory contains at least one .md file."""
        result = load_prompt_directory("knowledge")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_loads_supervisor_directory(self) -> None:
        """The supervisor directory contains at least one .md file."""
        result = load_prompt_directory("supervisor")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_loads_platform_directory(self) -> None:
        """The platform directory contains at least one .md file."""
        result = load_prompt_directory("platform")

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_nonexistent_directory_returns_empty(self) -> None:
        """A missing directory returns an empty string instead of raising."""
        result = load_prompt_directory("nonexistent_dir")

        self.assertEqual(result, "")

    def test_deterministic_order(self) -> None:
        """Repeated loads produce identical output (sorted path order)."""
        first = load_prompt_directory("knowledge")
        second = load_prompt_directory("knowledge")

        self.assertEqual(first, second)


class LoadKnowledgeTest(unittest.TestCase):
    def test_returns_nonempty_string(self) -> None:
        """The combined knowledge + platform content is non-empty."""
        result = load_knowledge()

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_includes_knowledge_and_platform_content(self) -> None:
        """The result contains the full text of both knowledge and platform directories."""
        result = load_knowledge()
        knowledge_only = load_prompt_directory("knowledge")
        platform_only = load_prompt_directory("platform")

        for section in (knowledge_only, platform_only):
            if section:
                self.assertIn(section, result)
