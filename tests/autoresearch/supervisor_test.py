# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import unittest

from spdl.autoresearch._common._supervisor import (
    _ClaudeSupervisor,
    _CodexSupervisor,
)


class SupervisorTest(unittest.TestCase):
    def test_claude_supervisor_builds_system_prompt_command(self) -> None:
        """Claude supervisor passes system prompt and initial request as separate args."""
        command = _ClaudeSupervisor().command("SYSTEM", "INITIAL")

        self.assertEqual(["claude", "--system-prompt", "SYSTEM", "INITIAL"], command)

    def test_codex_supervisor_merges_system_and_request_into_single_prompt(
        self,
    ) -> None:
        """Codex supervisor combines system prompt and request into one argument."""
        command = _CodexSupervisor().command("SYSTEM", "INITIAL")

        self.assertEqual("codex", command[0])
        self.assertEqual(2, len(command))
        self.assertIn("SYSTEM", command[1])
        self.assertIn("## User Request", command[1])
        self.assertIn("INITIAL", command[1])
