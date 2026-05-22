# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from spdl.autoresearch._common._supervisor import (
    _ClaudeSupervisor,
    _CodexSupervisor,
    _SupervisorAvailability,
)
from spdl.tools.autoresearch import cli

__all__: list[str] = []


class _Supervisor:
    name = "codex"

    def is_available(self) -> _SupervisorAvailability:
        return _SupervisorAvailability(True)

    def command(self, system_prompt: str, user_request: str) -> list[str]:
        return ["supervisor", system_prompt, user_request]


class _CliTest(unittest.TestCase):
    def test_path_like_positional_is_workdir(self) -> None:
        ns = cli._parse_args(["/tmp/autoresearch", "please", "run"])

        workdir, request = cli._split_workdir_and_request(ns)

        self.assertEqual("/tmp/autoresearch", workdir)
        self.assertEqual("please run", request)

    def test_non_path_positional_is_user_request(self) -> None:
        ns = cli._parse_args(["Run", "autoresearch", "here"])

        workdir, request = cli._split_workdir_and_request(ns)

        self.assertIsNone(workdir)
        self.assertEqual("Run autoresearch here", request)

    def test_engine_command_uses_supervisor_agent_as_workflow_default(self) -> None:
        ns = cli._parse_args(
            [
                "/tmp/autoresearch",
                "--agent",
                "codex",
                "--platform",
                "local",
                "--pipeline-script",
                "pipeline.py",
                "--source-dir",
                "src",
                "--build-command",
                "build image",
                "--base-launch-command",
                "launch $IMAGE",
            ]
        )

        command = cli._build_engine_command(ns, "/tmp/autoresearch", _Supervisor())

        self.assertTrue(command[1].endswith("spdl/tools/autoresearch/run.py"))
        self.assertIn("--agent", command)
        self.assertEqual("codex", command[command.index("--agent") + 1])
        self.assertEqual("local", command[command.index("--platform") + 1])
        self.assertIn("launch $IMAGE", command)

    def test_engine_command_override_can_use_environment_specific_launcher(
        self,
    ) -> None:
        ns = cli._parse_args(
            [
                "/tmp/autoresearch",
                "--engine-command",
                "env-run autoresearch-engine --",
            ]
        )

        command = cli._build_engine_command(ns, "/tmp/autoresearch", _Supervisor())

        self.assertEqual(
            ["env-run", "autoresearch-engine", "--"],
            command[:3],
        )

    def test_workflow_agent_can_override_supervisor_agent(self) -> None:
        ns = cli._parse_args(
            [
                "--agent",
                "codex",
                "--workflow-agent",
                "claude",
            ]
        )

        command = cli._build_engine_command(ns, "/tmp/autoresearch", _Supervisor())

        self.assertEqual("claude", command[command.index("--agent") + 1])

    def test_supervisor_context_lists_missing_inputs(self) -> None:
        ns = cli._parse_args(["--agent", "claude"])

        context = cli._build_supervisor_context(ns, None, _Supervisor())

        self.assertIn("Missing required first-run configuration", context)
        self.assertIn("- workdir", context)
        self.assertIn("- pipeline script", context)
        self.assertIn("Engine command template", context)

    def test_supervisor_prompt_loads_supervisor_category(self) -> None:
        ns = cli._parse_args([])

        prompt = cli._build_supervisor_prompt(ns, None, _Supervisor())

        self.assertIn("Autoresearch: Automated SPDL Pipeline Optimization", prompt)

    def test_initial_prompt_is_generated_when_request_is_empty(self) -> None:
        ns = cli._parse_args(
            [
                "/tmp/autoresearch",
                "--pipeline-script",
                "pipeline.py",
                "--source-dir",
                "src",
                "--build-command",
                "build image",
                "--base-launch-command",
                "launch $IMAGE",
            ]
        )

        prompt = cli._build_initial_prompt(ns, "/tmp/autoresearch", _Supervisor(), "")

        self.assertIn("Start supervising this autoresearch run", prompt)
        self.assertIn("Engine command", prompt)
        self.assertIn("--base-launch-command", prompt)

    def test_initial_prompt_preserves_explicit_user_request(self) -> None:
        ns = cli._parse_args(["/tmp/autoresearch"])

        prompt = cli._build_initial_prompt(
            ns,
            "/tmp/autoresearch",
            _Supervisor(),
            "Inspect the current run first.",
        )

        self.assertEqual("Inspect the current run first.", prompt)

    def test_claude_supervisor_receives_system_and_initial_prompts(self) -> None:
        command = _ClaudeSupervisor().command("SYSTEM", "INITIAL")

        self.assertEqual(["claude", "--system-prompt", "SYSTEM", "INITIAL"], command)

    def test_codex_supervisor_receives_system_and_initial_prompts(self) -> None:
        command = _CodexSupervisor().command("SYSTEM", "INITIAL")

        self.assertEqual("codex", command[0])
        self.assertEqual(2, len(command))
        self.assertIn("SYSTEM", command[1])
        self.assertIn("## User Request", command[1])
        self.assertIn("INITIAL", command[1])


if __name__ == "__main__":
    unittest.main()
